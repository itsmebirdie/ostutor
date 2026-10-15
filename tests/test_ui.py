"""UI tests. The view-model needs no Streamlit; the app tests use Streamlit's AppTest."""
import json
from pathlib import Path

import pytest

from ostutor.dsl.parser import parse
from ostutor.slice.explorer import analyze
from ostutor.ui import viewmodel as vm

DATA = Path(__file__).parent.parent / "data" / "programs"
ROOT = Path(__file__).parent.parent


def load(name):
    p = parse((DATA / f"{name}.ost").read_text())
    return p.program, p.loop_unroll


# ---------------------------------------------------------------- view-model
def test_interleavings_matches_report_figures():
    assert vm.interleavings([10, 10]) == 184_756
    assert vm.interleavings([6, 6, 6]) == 17_153_136  # appendix A.3 of the build plan


def test_scaling_rows_are_measured_and_consistent_with_the_report():
    rows = vm.scaling_rows(10)
    assert [r["steps_per_thread"] for r in rows] == list(range(1, 11))
    last = rows[-1]
    assert last["interleavings"] == 184_756 and last["states"] == 121  # (n+1)^2
    assert all(r["states"] == (r["steps_per_thread"] + 1) ** 2 for r in rows)
    assert all(a["ratio"] < b["ratio"] for a, b in zip(rows, rows[1:]))  # gap widens


def test_witness_rows_for_deadlock_and_safe():
    program, _ = load("deadlock_two_locks")
    rows = vm.witness_rows(analyze(program))
    assert [(r["Thread"], r["Operation"]) for r in rows] == [("T1", "lock a"), ("T2", "lock b")]
    safe, _ = load("safe_with_lock")
    assert vm.witness_rows(analyze(safe)) == []


def test_state_graph_dot_colours_and_witness_path():
    program, unroll = load("deadlock_two_locks")
    dot = vm.state_graph_dot(program, unroll)
    assert dot.startswith("digraph") and dot.rstrip().endswith("}")
    assert dot.count("fillcolor") == 19  # one node per explored state
    assert "#ffcdd2" in dot  # the deadlock state is coloured MUST_FAIL
    assert dot.count("penwidth=3") >= 3  # deadlock node + two witness edges
    assert 'label="T1"' in dot and 'label="T2"' in dot


def test_state_graph_is_none_when_too_large():
    program, unroll = load("dining_3_deadlock")
    assert vm.state_graph_dot(program, unroll, max_nodes=10) is None
    assert vm.state_graph_dot(program, unroll) is not None  # 87 states fits the default


def test_state_graph_for_safe_program_has_no_red():
    program, unroll = load("safe_with_lock")
    dot = vm.state_graph_dot(program, unroll)
    assert "#ffcdd2" not in dot and "#b71c1c" not in dot


def test_rr_view_matches_report():
    procs = [("P1", 0, 5), ("P2", 1, 3), ("P3", 2, 1)]
    wrong = vm.example_student_answer(procs, 2)
    v = vm.rr_view(procs, 2, wrong)
    assert v["correct"] == "P1 P1 P2 P2 P3 P1 P1 P2 P1".split()
    assert v["divergence"] == 5 and v["diagnosis"].method == "mutant_match"
    ok = vm.rr_view(procs, 2, v["correct"])
    assert ok["divergence"] is None and ok["diagnosis"].method == "correct"


def test_parse_gantt_accepts_commas_and_spaces():
    assert vm.parse_gantt("P1, P1  P2,P3") == ["P1", "P1", "P2", "P3"]


# ---------------------------------------------------------------- the app itself
st_testing = pytest.importorskip("streamlit.testing.v1")


@pytest.fixture(autouse=True)
def _isolated_cache_and_fixtures(tmp_path, monkeypatch):
    """Never touch the developer's real demo_cache.sqlite / demo_fixtures during tests."""
    pytest.importorskip("streamlit")
    import ostutor.ui.app as app
    monkeypatch.setattr(app, "DB_PATH", str(tmp_path / "ui_test.sqlite"))
    monkeypatch.setattr(app, "FIXTURES_DIR", tmp_path / "ui_fixtures")


def new_app():
    at = st_testing.AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=60)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def click(at, label):
    next(b for b in at.button if b.label == label).click()
    at.run()
    assert not at.exception, [e.value for e in at.exception]


def test_app_loads_with_default_example_and_no_key():
    at = new_app()
    assert "deadlock_two_locks" in at.selectbox[0].value
    assert "T1" in at.text_area[0].value  # the example program is in the editor
    assert any("Analyze" in b.label for b in at.button)


def test_app_analyze_deadlock_shows_verdict_and_checker_rejections():
    at = new_app()
    click(at, "Analyze")
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Verdict"] == "DEADLOCK" and metrics["States explored"] == "19"
    assert metrics["Possible interleavings"] == "70"
    text = " ".join(m.value for m in at.markdown)
    assert "template" in text and "Critical decision point" in " ".join(
        w.value for w in at.warning)
    rejected = [e for e in at.error if e.value == "rejected"]
    assert len(rejected) == 4  # all four tamperings are caught


def test_app_live_edit_of_summary_is_checked():
    at = new_app()
    click(at, "Analyze")
    box = next(t for t in at.text_area if t.label == "Summary")
    box.set_value(box.value.replace("`deadlock`", "`safe`", 1)).run()
    assert not at.exception
    assert any("Checker rejected your edit" in e.value for e in at.error)


def test_app_switching_example_changes_the_program_and_verdict():
    at = new_app()
    at.selectbox[0].select("safe_with_lock").run()
    assert "lock m" in at.text_area[0].value
    click(at, "Analyze")
    assert {m.label: m.value for m in at.metric}["Verdict"] == "SAFE"


def test_app_parse_error_is_shown_not_crashed():
    at = new_app()
    at.text_area[0].set_value("mutex a\nthread T1:\n  wait a\n").run()
    click(at, "Analyze")
    assert any("line 3" in e.value and "condition variables" in e.value for e in at.error)


def test_app_llm_toggle_without_key_falls_back_to_template():
    at = new_app()
    at.toggle[0].set_value(True)
    at.toggle[1].set_value(False).run()  # no fixtures, no key
    click(at, "Analyze")
    assert any("template fall-back" in m.value for m in at.markdown)


def test_app_replays_recorded_fixture_with_no_key(tmp_path, monkeypatch):
    """The offline-demo promise: recorded LLM answers are shown as 'checked (LLM)'."""
    import ostutor.ui.app as app
    from ostutor.narration.llm import FixtureStore, LLMClient, LLMConfig
    from ostutor.narration.pipeline import explain
    from ostutor.slice.narrate import narrate

    program, unroll = load("deadlock_two_locks")
    verdict = analyze(program)
    canned = json.dumps(narrate(program, verdict))
    rec = LLMClient(LLMConfig(), FixtureStore(tmp_path, "record"),
                    transport=lambda *a: (200, {"choices": [{"message": {"content": canned}}],
                                                "usage": {"total_tokens": 1}}, {}))
    rec.config.api_key = "k"
    assert explain(program, verdict, rec).source == "llm"
    monkeypatch.setattr(app, "FIXTURES_DIR", tmp_path)

    at = new_app()
    at.toggle[0].set_value(True).run()  # LLM on, fixtures on (default)
    click(at, "Analyze")
    assert any("checked (LLM)" in m.value for m in at.markdown)


def test_app_scaling_and_round_robin_tabs_render():
    at = new_app()
    assert any("184,756" in c.value for c in at.caption)
    assert any("rr_preempted_before_new_arrival" in e.value for e in at.error)
