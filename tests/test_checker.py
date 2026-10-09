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
