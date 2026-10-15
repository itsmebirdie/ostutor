"""Expected values below were derived BY HAND (see README), then checked against code.

Hand counts of reachable states (pc of T1, pc of T2):
  deadlock_two_locks: 19  (5 + 4 + 2 + 3 + 5 for T1 pc = 0,1,2,3,4)
  race_no_lock      : 9   (3 x 3, no locks restrict anything)
  safe_with_lock    : 16  (25 pairs minus the 9 where both threads are inside the lock)
"""
import math

import pytest

from ostutor.slice.explorer import Model, ProgramError, StateBudgetExceeded, analyze, analyze_full
from ostutor.slice.programs import PROGRAMS


def test_deadlock_verdict():
    v = analyze(PROGRAMS["deadlock_two_locks"])
    assert v.label == "deadlock"
    assert v.completeness == "complete_for_bound"
    assert v.states_explored == 19 < 20
    steps = [(s.thread, s.op, s.object) for s in v.witness.steps]
    assert steps == [("T1", "lock", "a"), ("T2", "lock", "b")]
    assert v.witness.steps[-1].blocked == ["T1", "T2"]
    assert v.detail["owners"] == {"a": "T1", "b": "T2"}


def test_deadlock_critical_point():
    cp = analyze(PROGRAMS["deadlock_two_locks"]).critical_point
    assert cp["decision_after_step"] == 1
    assert cp["step_index"] == 2
    assert cp["bad_choice"] == "T2:lock b"
    assert cp["safe_alternative"] == "T1:lock b"


def test_race_verdict():
    v = analyze(PROGRAMS["race_no_lock"])
    assert v.label == "race"
    assert v.detail["variable"] == "x"
    assert v.states_explored == 9
    assert len(v.witness.steps) == 1  # after T1 reads, T1-write vs T2-read conflict
    assert v.critical_point is None  # no schedule avoids it


def test_safe_verdict_reaches_all_terminals():
    v, ex, labels = analyze_full(PROGRAMS["safe_with_lock"])
    assert v.label == "safe"
    assert v.witness is None and v.critical_point is None
    assert v.states_explored == 16
    assert len(ex.terminals) == 1  # only (4, 4)
    assert set(labels.values()) == {"ALWAYS_SAFE"}


def test_state_labels_on_deadlock_program():
    v, ex, labels = analyze_full(PROGRAMS["deadlock_two_locks"])
    assert labels[ex.initial] == "MAY_FAIL"
    assert [s for s, l in labels.items() if l == "MUST_FAIL"] == ex.deadlocks
    assert len(ex.deadlocks) == 1


def test_interleavings_vs_states():
    prog = {"mutexes": [], "shared": ["x"],
            "threads": {"T1": [("read", "x")] * 10, "T2": [("read", "x")] * 10}}
    v = analyze(prog)
    assert v.label == "safe"  # reads never conflict
    assert v.states_explored == 121
    assert math.comb(20, 10) == 184_756


def test_budget_exceeded():
    with pytest.raises(StateBudgetExceeded):
        analyze(PROGRAMS["safe_with_lock"], budget=3)


def test_bad_programs_rejected():
    with pytest.raises(ProgramError):
        Model({"mutexes": [], "shared": [], "threads": {"T1": [("lock", "zz")]}})
    with pytest.raises(ProgramError):
        Model({"mutexes": [], "shared": [], "threads": {"T1": [("jump", "x")]}})


def test_verdict_has_contract_fields():
    d = analyze(PROGRAMS["deadlock_two_locks"]).to_dict()
    for key in ("label", "completeness", "bounds", "states_explored",
                "fairness_assumption", "witness", "critical_point", "program_hash"):
        assert key in d
    assert d["program_hash"].startswith("sha256:")
