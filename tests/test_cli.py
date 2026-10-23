"""In-process CLI tests (fast, and counted by coverage)."""
import json
from pathlib import Path

from ostutor.cli import main
from ostutor.dsl.parser import parse
from ostutor.explorer import analyze
from ostutor.narration.llm import FixtureStore, LLMClient, LLMConfig
from ostutor.narration.pipeline import explain, template_narrate

DATA = Path(__file__).parent.parent / "data" / "programs"
DEADLOCK = DATA / "deadlock_two_locks.ost"


def test_analyze_text_output(capsys):
    assert main(["analyze", str(DEADLOCK)]) == 0
    out = capsys.readouterr().out
    assert "label        : deadlock" in out and "reproduced" in out
    assert "[template]" in out and "checker validates facts" in out


def test_analyze_json_output(capsys):
    assert main(["analyze", str(DATA / "safe_with_lock.ost"), "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["verdict"]["label"] == "safe" and data["narration_source"] == "template"


def test_missing_file_and_parse_error_exit_2(capsys, tmp_path):
    assert main(["analyze", str(tmp_path / "nope.ost")]) == 2
    bad = tmp_path / "b.ost"
    bad.write_text("thread T1:\n  lock zz\n")
    assert main(["analyze", str(bad)]) == 2
    assert "not declared" in capsys.readouterr().err


def test_tiny_budget_escalates_to_sampling_and_says_so(capsys):
    assert main(["analyze", str(DEADLOCK), "--budget", "3"]) == 0
    out = capsys.readouterr().out
    assert "sampled" in out and "label        : deadlock" in out


def test_state_budget_exit_3_when_tier_a_is_forced(capsys):
    assert main(["analyze", str(DEADLOCK), "--budget", "3", "--tier", "A"]) == 3
    assert "Search stopped" in capsys.readouterr().err


def test_db_cache_makes_second_run_a_hit(capsys, tmp_path):
    db = str(tmp_path / "c.sqlite")
    assert main(["analyze", str(DEADLOCK), "--db", db]) == 0
    assert main(["analyze", str(DEADLOCK), "--db", db]) == 0
    assert (tmp_path / "c.sqlite").exists()


def test_llm_path_runs_fully_offline_from_a_recorded_fixture(capsys, tmp_path):
    """Record one fake LLM answer, then replay it through the real CLI with no key."""
    program = parse(DEADLOCK.read_text()).program
    verdict = analyze(program)
    canned = json.dumps(template_narrate(program, verdict))

    def transport(url, headers, body, timeout):
        return 200, {"choices": [{"message": {"content": canned}}],
                     "usage": {"total_tokens": 321}}, {}

    recorder = LLMClient(LLMConfig(api_key="k"), FixtureStore(tmp_path, "record"),
                         transport=transport)
    assert explain(program, verdict, recorder).source == "llm"

    code = main(["analyze", str(DEADLOCK), "--llm", "--fixtures", str(tmp_path),
                 "--fixture-mode", "replay"])
    out = capsys.readouterr().out
    assert code == 0 and "checked (LLM)" in out and "llm calls    : 0" in out


def test_llm_flag_without_key_degrades(capsys):
    assert main(["analyze", str(DEADLOCK), "--llm"]) == 0
    assert "template fall-back" in capsys.readouterr().out


def test_ping_without_key_fails_cleanly(capsys, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # so no developer .env is picked up
    assert main(["ping"]) == 1
    out = capsys.readouterr().out
    assert "NOT SET" in out and "FAILED" in out


def test_models_without_key_fails_cleanly(capsys, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert main(["models"]) == 1
    assert "no API key" in capsys.readouterr().out


def test_models_lists_ids_and_warns_when_current_model_is_missing(capsys, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "k")
    monkeypatch.setenv("OSTUTOR_MODEL", "retired-model")
    monkeypatch.setattr("ostutor.cli.list_models", lambda cfg: ["openai/gpt-oss-120b", "x/y"])
    assert main(["models"]) == 0
    out = capsys.readouterr().out
    assert "openai/gpt-oss-120b" in out and "WARNING: 'retired-model'" in out


def test_warm_records_every_example_and_a_second_run_costs_nothing(capsys, tmp_path):
    """`warm` is what makes the demo offline-safe: it must pre-record one checked narration
    per example program, and then be satisfied entirely from the recordings."""
    import ostutor.cli as cli
    programs = sorted(DATA.glob("*.ost"))
    assert len(programs) >= 12

    def transport(url, headers, body, timeout):  # a well-behaved fake model
        prompt = body["messages"][1]["content"]
        label = next(ln.split(": ")[1] for ln in prompt.splitlines() if ln.startswith("label:"))
        for p in programs:
            parsed = parse(p.read_text())
            v = analyze(parsed.program, loop_unroll=parsed.loop_unroll)
            if v.label == label and f"states_explored: {v.states_explored}" in prompt:
                from ostutor.narration.evidence import encode_evidence
                if encode_evidence(parsed.program, v) in prompt:
                    canned = json.dumps(template_narrate(parsed.program, v))
                    return 200, {"choices": [{"message": {"content": canned}}],
                                 "usage": {"total_tokens": 10}}, {}
        raise AssertionError("prompt did not match any example program")

    real = cli.LLMClient
    fx, db = tmp_path / "fx", str(tmp_path / "c.sqlite")

    class Fake(real):  # route the CLI's client through the fake transport
        def __init__(self, cfg, fixtures=None):
            super().__init__(cfg, fixtures=fixtures, transport=transport)
            self.config.api_key = "test-key"

    cli.LLMClient = Fake
    try:
        assert main(["warm", "--programs", str(DATA), "--db", db, "--fixtures", str(fx)]) == 0
        out = capsys.readouterr().out
        assert out.count("checked (LLM)") == len(programs)
        assert len(list(fx.glob("*.json"))) == len(programs)
    finally:
        cli.LLMClient = real
    # second run: no key, no network, answers come from the cache/fixtures
    assert main(["warm", "--programs", str(DATA), "--db", db, "--fixtures", str(fx)]) == 0
    out2 = capsys.readouterr().out
    assert out2.count("checked (cached LLM)") == len(programs)


def test_warm_without_a_key_reports_the_fallbacks_and_fails(capsys, tmp_path):
    code = main(["warm", "--programs", str(DATA), "--db", str(tmp_path / "c.sqlite"),
                 "--fixtures", str(tmp_path / "fx")])
    out = capsys.readouterr().out
    assert code == 1 and "template fall-back" in out and "fell back to the template" in out
