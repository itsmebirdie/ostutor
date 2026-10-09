import copy

import pytest

from ostutor.slice.explorer import analyze
from ostutor.slice.programs import PROGRAMS
from ostutor.slice.replay import replay


@pytest.mark.parametrize("name", sorted(PROGRAMS))
def test_every_witness_replays(name):
    v = analyze(PROGRAMS[name])
    if v.witness is None:
        pytest.skip("safe program has no witness")
    r = replay(PROGRAMS[name], v.witness, expected=v.label)
    assert r.ok and r.failure == v.label


def test_illegal_schedule_rejected():
    prog = PROGRAMS["deadlock_two_locks"]
    w = copy.deepcopy(analyze(prog).witness)
    w.steps[1].thread = "T1"  # T1 locking a again: wrong op / not the next op
    assert not replay(prog, w).ok


def test_blocked_thread_cannot_run():
    prog = PROGRAMS["deadlock_two_locks"]
    w = copy.deepcopy(analyze(prog).witness)
    w.steps[1].op, w.steps[1].object = "lock", "a"  # T2 locks a, but T1 holds a
    r = replay(prog, w)
    assert not r.ok and "not enabled" in r.reason or "does not match" in r.reason


def test_legal_but_harmless_prefix_is_not_a_failure():
    prog = PROGRAMS["deadlock_two_locks"]
    w = copy.deepcopy(analyze(prog).witness)
    w.steps = w.steps[:1]
    r = replay(prog, w)
    assert not r.ok and r.failure is None


def test_wrong_expected_label():
    prog = PROGRAMS["race_no_lock"]
    v = analyze(prog)
    assert not replay(prog, v.witness, expected="deadlock").ok
