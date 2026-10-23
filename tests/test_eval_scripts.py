"""Guards for the evaluation scripts: the labelled set must agree with the engine, and the
fast scripts must run and satisfy the invariants their reports claim. (The classifier script
takes ~30 s and is exercised by the CI `eval` job instead.)"""
import json
import sys
from pathlib import Path

import pytest

EVAL = Path(__file__).parent.parent / "eval"
sys.path.insert(0, str(EVAL))

import common  # noqa: E402
import dataset  # noqa: E402

from ostutor.dsl.parser import parse  # noqa: E402
from ostutor.explorer import analyze  # noqa: E402


@pytest.fixture()
def results_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "RESULTS", tmp_path)
    return tmp_path


def test_dataset_has_between_30_and_50_programs_all_parse_and_cover_every_label():
    assert 30 <= len(dataset.DATASET) <= 50
    seen = set()
    for d in dataset.DATASET:
        parse(d["code"])
        exp = d["expected"]
        seen |= set(exp.values()) if isinstance(exp, dict) else {exp}
    assert seen == {"safe", "race", "deadlock", "lost_signal", "starvation"}


def test_dataset_labels_agree_with_the_engine_under_both_fairness_assumptions():
    for d in dataset.DATASET:
        p = parse(d["code"])
        exp = d["expected"] if isinstance(d["expected"], dict) else {
            "weak": d["expected"], "strong": d["expected"]}
        for fairness in ("weak", "strong"):
            v = analyze(p.program, loop_unroll=p.loop_unroll, fairness=fairness)
            assert v.label == exp[fairness], (d["name"], fairness, v.label)


def test_dataset_provenance_warning_is_present():
    assert "NOT labelled by two independent human annotators" in (EVAL / "dataset.py").read_text()


def test_faithfulness_report_states_the_invariants_and_the_blind_spot(results_dir):
    import run_faithfulness
    data = run_faithfulness.main()
    assert data["template_pass"] == data["template_total"] > 40
    assert all(v["rejected"] == v["total"] for v in data["tamper"].values())
    assert data["blind_spot_accepted"] == data["blind_spot_total"] > 0  # documented limitation
    md = (results_dir / "faithfulness.md").read_text()
    assert "cannot see" in md and "causal" in md


def test_diagnosis_report_is_labelled_synthetic_and_internally_consistent(results_dir):
    import run_diagnosis
    data = run_diagnosis.main()
    assert data["synthetic_only"] is True
    assert data["reachable"] == data["recognised"] == data["mutants"]
    assert data["correct_not_blamed"] == data["correct_total"]
    assert data["junk_matched"] == 0
    assert "not a diagnostic-accuracy result" in (results_dir / "diagnosis.md").read_text()


def test_router_and_rag_scripts_run_and_report_caveats(results_dir):
    import run_rag
    import run_router
    r = run_router.main()
    assert r["cases"] >= 30 and r["agree"] >= 0.9 * r["cases"]
    g = run_rag.main()
    assert g["hit3"] == g["n"] and g["off_topic"] == 8
    md = (results_dir / "rag.md").read_text()
    assert "Caveat" in md and "lexical" in md
    assert json.loads((results_dir / "rag.json").read_text())["n"] == g["n"]


def test_efficiency_script_reports_measured_scaling(results_dir):
    import run_efficiency
    data = run_efficiency.main()
    assert data["bytes_per_state"] > 0 and data["budget_states_per_second"] > 1000
    last = data["scaling"][-1]
    assert last[0] == 12 and last[2] == 169  # (n+1)^2 states for n = 12
