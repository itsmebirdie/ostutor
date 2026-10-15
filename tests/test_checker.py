import copy

import pytest

from ostutor.slice.check import check, tamper_examples
from ostutor.slice.explorer import analyze
from ostutor.slice.narrate import narrate
from ostutor.slice.programs import PROGRAMS


@pytest.mark.parametrize("name", sorted(PROGRAMS))
def test_template_narration_always_passes(name):
    prog = PROGRAMS[name]
    v = analyze(prog)
    assert check(narrate(prog, v), v, prog).ok


@pytest.mark.parametrize("name", sorted(PROGRAMS))
@pytest.mark.parametrize("kind", ["invented step index", "unknown thread name",
                                  "wrong number", "wrong label"])
def test_tampered_narrations_rejected(name, kind):
    prog = PROGRAMS[name]
    v = analyze(prog)
    good = narrate(prog, v)
    bad = tamper_examples(good, v)[kind]
    assert bad != good
    res = check(bad, v, prog)
    assert not res.ok, f"{kind} was accepted"


def test_each_tamper_trips_the_expected_rule():
    prog = PROGRAMS["deadlock_two_locks"]
    v = analyze(prog)
    t = tamper_examples(narrate(prog, v), v)
    rules = {k: {x["rule"] for x in check(n, v, prog).violations} for k, n in t.items()}
    assert "step_index" in rules["invented step index"]
    assert "vocabulary" in rules["unknown thread name"]
    assert "number" in rules["wrong number"]
    assert "label" in rules["wrong label"]


def test_missing_verdict_rejected():
    prog = PROGRAMS["race_no_lock"]
    v = analyze(prog)
    n = copy.deepcopy(narrate(prog, v))
    n["summary"] = "Something happened."
    assert not check(n, v, prog).ok


def test_invented_mutex_in_backticks_rejected():
    prog = PROGRAMS["safe_with_lock"]
    v = analyze(prog)
    n = narrate(prog, v)
    n["summary"] += " `q` was held."
    assert not check(n, v, prog).ok


# ---- regression tests for the limitations found in the review report -------
def test_after_2_steps_on_a_one_step_race_witness_is_rejected():
    """Report limitation 1: '2' was accepted because it equals the thread count."""
    v = analyze(PROGRAMS["race_no_lock"])
    n = narrate(PROGRAMS["race_no_lock"], v)
    n["summary"] += " After 2 steps the race appears."
    res = check(n, v, PROGRAMS["race_no_lock"])
    assert not res.ok and any(x["rule"] == "number" for x in res.violations)


def test_thread_count_cannot_stand_in_for_a_state_count():
    v = analyze(PROGRAMS["deadlock_two_locks"])
    n = narrate(PROGRAMS["deadlock_two_locks"], v)
    n["summary"] += " Searching visited 2 states."
    assert not check(n, v, PROGRAMS["deadlock_two_locks"]).ok


def test_correct_counts_in_free_text_are_accepted():
    p = PROGRAMS["deadlock_two_locks"]
    v = analyze(p)
    n = narrate(p, v)
    n["summary"] += " The program has 2 threads and 2 mutexes; the witness has 2 steps."
    assert check(n, v, p).ok


def test_bare_invented_name_after_a_keyword_is_rejected():
    """Report limitation 2: names outside backticks."""
    p = PROGRAMS["deadlock_two_locks"]
    v = analyze(p)
    n = narrate(p, v)
    n["summary"] += " Thread T1 then waits on mutex c forever."
    res = check(n, v, p)
    assert not res.ok and any("unknown name c" in x["detail"] for x in res.violations)


def test_prose_after_keywords_is_not_a_false_positive():
    p = PROGRAMS["deadlock_two_locks"]
    v = analyze(p)
    n = narrate(p, v)
    n["summary"] += " Each thread holds a lock while it waits for the other lock."
    assert check(n, v, p).ok
