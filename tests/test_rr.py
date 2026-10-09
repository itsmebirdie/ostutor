from ostutor.slice import rr

PROCS = [("P1", 0, 5), ("P2", 1, 3), ("P3", 2, 1)]
CORRECT = ["P1", "P1", "P2", "P2", "P3", "P1", "P1", "P2", "P1"]  # derived by hand
MUTANT = ["P1", "P1", "P2", "P2", "P1", "P1", "P3", "P2", "P1"]


def test_simulate_matches_hand_traces():
    assert rr.simulate(PROCS, 2) == CORRECT
    assert rr.simulate(PROCS, 2, "preempted_first") == MUTANT


def test_mutant_is_identified_with_first_divergence():
    d = rr.diagnose(MUTANT, PROCS, 2)
    assert d.method == "mutant_match"
    assert d.misconception_id == "rr_preempted_before_new_arrival"
    assert d.first_divergence == {"step": 5, "expected": "P3", "got": "P1"}


def test_correct_answer():
    assert rr.diagnose(CORRECT, PROCS, 2).method == "correct"


def test_random_slip_is_not_called_systematic():
    slip = list(CORRECT)
    slip[8] = "P2"
    d = rr.diagnose(slip, PROCS, 2)
    assert d.method == "edit_distance" and d.systematic is False
    assert d.first_divergence["step"] == 9


def test_validation():
    import pytest
    with pytest.raises(ValueError):
        rr.simulate([("P1", 0, -1)], 2)
    with pytest.raises(ValueError):
        rr.simulate(PROCS, 0)
