import json

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from ostutor.api.app import create_app  # noqa: E402
from ostutor.narration.llm import LLMClient, LLMConfig  # noqa: E402
from ostutor.narration.pipeline import template_narrate  # noqa: E402
from ostutor.slice.explorer import analyze  # noqa: E402
from ostutor.slice.programs import PROGRAMS  # noqa: E402

DEADLOCK = ("mutex a, b\nthread T1:\n  lock a\n  lock b\n  unlock b\n  unlock a\n"
            "thread T2:\n  lock b\n  lock a\n  unlock a\n  unlock b\n")
RR = {"topic": "cpu", "algorithm": "rr",
      "params": {"procs": [["P1", 0, 5], ["P2", 1, 3], ["P3", 2, 1]], "quantum": 2}}


@pytest.fixture()
def client():
    return TestClient(create_app())


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_analyze_code_returns_verdict_witness_replay_and_checked_narration(client):
    r = client.post("/analyze_code", json={"code": DEADLOCK})
    assert r.status_code == 200
    d = r.json()
    assert d["verdict"]["label"] == "deadlock" and d["verdict"]["states_explored"] == 19
    assert d["verdict"]["completeness"] == "complete_for_bound"
    assert len(d["verdict"]["witness"]["steps"]) == 2
    assert d["replay"]["ok"] is True and d["narration_source"] == "template"
    assert d["narration"]["summary"].startswith("Verdict: `deadlock`") and d["checked"]


def test_analyze_code_handles_semaphores_conditions_and_starvation(client):
    lost = "mutex m\ncond c\nthread Waiter:\n lock m\n wait c, m\n unlock m\nthread Signaller:\n lock m\n signal c\n unlock m\n"
    assert client.post("/analyze_code", json={"code": lost}).json()["verdict"]["label"] == "lost_signal"
    starve = "mutex m\nthread A:\n forever:\n  lock m\n  unlock m\nthread B:\n forever:\n  lock m\n  unlock m\nthread C:\n forever:\n  lock m\n  unlock m\n"
    weak = client.post("/analyze_code", json={"code": starve, "fairness": "weak"}).json()
    strong = client.post("/analyze_code", json={"code": starve, "fairness": "strong"}).json()
    assert weak["verdict"]["label"] == "starvation" and weak["replay"]["ok"]
    assert strong["verdict"]["label"] == "safe"


def test_parse_errors_are_422_with_a_line_number(client):
    r = client.post("/analyze_code", json={"code": "mutex a\nthread T1:\n  sleep 5\n"})
    assert r.status_code == 422
    d = r.json()["detail"]
    assert d["error"] == "parse" and d["line"] == 3 and "timing" in d["message"]


def test_forced_tier_a_over_budget_is_422_but_auto_degrades(client):
    body = {"code": DEADLOCK, "budget": 10, "tier": "A"}
    assert client.post("/analyze_code", json=body).status_code == 422
    ok = client.post("/analyze_code", json={"code": DEADLOCK, "budget": 10}).json()
    assert ok["verdict"]["completeness"] == "sampled"


@pytest.mark.parametrize("bad", [{"code": DEADLOCK, "tier": "Z"}, {"code": DEADLOCK, "fairness": "x"},
                                 {"code": DEADLOCK, "budget": 1}, {"nocode": 1}])
def test_request_validation(client, bad):
    assert client.post("/analyze_code", json=bad).status_code == 422


def test_narrate_without_a_key_falls_back_to_the_template(client, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    d = client.post("/analyze_code", json={"code": DEADLOCK, "narrate": True}).json()
    assert d["narration_source"] == "template_fallback" and "no API key" in d["narration_note"]


def test_narrate_with_an_injected_llm_is_checked_and_used():
    program = PROGRAMS["deadlock_two_locks"]
    canned = json.dumps(template_narrate(program, analyze(program)))
    llm = LLMClient(LLMConfig(api_key="k"), transport=lambda *a: (
        200, {"choices": [{"message": {"content": canned}}], "usage": {"total_tokens": 9}}, {}))
    c = TestClient(create_app(llm=llm))
    d = c.post("/analyze_code", json={"code": DEADLOCK, "narrate": True}).json()
    assert d["narration_source"] == "llm" and llm.calls == 1
    again = c.post("/analyze_code", json={"code": DEADLOCK, "narrate": True}).json()
    assert again["narration_source"] == "cache" and llm.calls == 1  # second call costs nothing


def test_simulate_returns_the_correct_trace(client):
    r = client.post("/simulate", json={"problem": RR})
    assert r.status_code == 200 and r.json()["answer"] == "P1 P1 P2 P2 P3 P1 P1 P2 P1".split()
    disk = {"topic": "disk", "algorithm": "sstf", "params": {
        "requests": [98, 183, 37, 122, 14, 124, 65, 67], "head": 53}}
    assert client.post("/simulate", json={"problem": disk}).json()["metrics"][
        "total_head_movement"] == 236


@pytest.mark.parametrize("problem", [
    {"topic": "cpu", "algorithm": "rr", "params": {"procs": [["A", 0, -1]]}},
    {"topic": "paging", "algorithm": "lru", "params": {"refs": [1], "frames": 0}},
    {"topic": "nope", "algorithm": "x"}, {"algorithm": "rr"},
    {"topic": "banker", "algorithm": "safety", "params": {"available": [1], "maximum": [[1, 1]],
                                                          "allocation": [[0, 0]]}}])
def test_bad_problems_are_422_not_500(client, problem):
    r = client.post("/simulate", json={"problem": problem})
    assert r.status_code == 422 and r.json()["detail"]["error"] == "problem"


def test_diagnose_names_the_misconception_and_the_first_divergence(client):
    r = client.post("/diagnose", json={"problem": RR, "answer": "P1 P1 P2 P2 P1 P1 P3 P2 P1"})
    d = r.json()
    assert d["diagnosis"]["method"] == "mutant_match"
    assert d["diagnosis"]["misconception_id"] == "rr_preempted_before_new_arrival"
    assert d["diagnosis"]["first_divergence"] == {"step": 5, "expected": "P3", "got": "P1"}
    assert d["correct_answer"][4] == "P3"
    assert client.post("/diagnose", json={"problem": RR, "answer": ""}).status_code == 422


def test_ask_routes_code_to_the_explorer(client):
    d = client.post("/ask", json={"message": DEADLOCK}).json()
    assert d["route"]["layer"] == "analyze_code" and d["result"]["verdict"]["label"] == "deadlock"


def test_ask_answers_concept_questions_from_the_course_material(client):
    d = client.post("/ask", json={"message": "How does chmod 755 work?"}).json()
    assert d["route"]["layer"] == "rag" and d["result"]["found"]
    assert "chmod 755" in d["result"]["answer"] and d["result"]["citations"]
    off = client.post("/ask", json={"message": "who won the football world cup"}).json()
    assert off["result"]["found"] is False and off["result"]["answer"].startswith("Not found")


def test_ask_asks_for_structured_input_when_a_problem_is_detected_but_missing(client):
    d = client.post("/ask", json={"message": "Compute the FCFS average waiting time for "
                                              "bursts 24 3 3"}).json()
    assert d["route"]["layer"] == "simulate" and d["needs_structured_input"] is True
    ok = client.post("/ask", json={"message": "Compute the round robin Gantt chart, quantum 2",
                                   "problem": RR}).json()
    assert ok["result"]["answer"][0] == "P1"
    dg = client.post("/ask", json={"message": "My Gantt chart for round robin is P1 P1 P2 P2 P1 "
                                              "P1 P3 P2 P1, is this right?", "problem": RR,
                                   "answer": "P1 P1 P2 P2 P1 P1 P3 P2 P1"}).json()
    assert dg["route"]["layer"] == "diagnose"
    assert dg["result"]["diagnosis"]["misconception_id"] == "rr_preempted_before_new_arrival"


def test_oversized_input_is_rejected(client):
    assert client.post("/analyze_code", json={"code": "x" * 30_000}).status_code == 422
