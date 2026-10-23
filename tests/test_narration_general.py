"""Parser extensions and general narration: every template must pass the checker, and
every tampered version must be rejected (for programs with a witness)."""
import re
from pathlib import Path

import pytest

from ostutor.dsl.parser import ParseError, canonical_hash, parse
from ostutor.explorer import analyze
from ostutor.narration.evidence import encode_evidence, estimate_tokens
from ostutor.narration.pipeline import template_narrate
from ostutor.slice.check import check, tamper_examples

DATA = Path(__file__).parent.parent / "data" / "programs"
ALL = sorted(p.stem for p in DATA.glob("*.ost"))
EXPECTED = {
    "deadlock_two_locks": "deadlock", "race_no_lock": "race", "safe_with_lock": "safe",
    "dining_3_deadlock": "deadlock", "dining_3_ordered": "safe", "loop_counter_locked": "safe",
    "dining_5_deadlock": "deadlock", "producer_consumer_ok": "safe",
    "producer_consumer_bad_order": "deadlock", "lost_signal": "lost_signal",
    "starvation_unfair_mutex": "starvation", "starvation_blocked_forever": "starvation",
}


def load(name):
    p = parse((DATA / f"{name}.ost").read_text())
    return p.program, p.loop_unroll


def test_every_example_file_has_a_hand_derived_expectation():
    assert set(ALL) == set(EXPECTED)


@pytest.mark.parametrize("name", ALL)
def test_template_narration_passes_the_checker_for_every_program(name):
    program, unroll = load(name)
    v = analyze(program, loop_unroll=unroll)
    assert v.label == EXPECTED[name]
    n = template_narrate(program, v)
    res = check(n, v, program)
    assert res.ok, (name, res.violations, n)
    assert n["summary"].startswith(f"Verdict: `{v.label}`")


@pytest.mark.parametrize("name", ALL)
def test_checker_rejects_tampered_narration_for_every_program(name):
    program, unroll = load(name)
    v = analyze(program, loop_unroll=unroll)
    n = template_narrate(program, v)
    tampered = tamper_examples(n, v)
    assert len(tampered) == 4
    for kind, bad in tampered.items():
        if kind.startswith("invented step") and not v.witness:
            continue  # nothing to invent against: the safe programs have no witness steps
        assert not check(bad, v, program).ok, (name, kind)


@pytest.mark.parametrize("name", ALL)
def test_evidence_stays_within_the_token_budget(name):
    program, unroll = load(name)
    v = analyze(program, loop_unroll=unroll)
    assert estimate_tokens(encode_evidence(program, v)) <= 1000


def test_new_narration_mentions_the_right_mechanisms():
    p, u = load("lost_signal")
    n = template_narrate(p, analyze(p))
    text = " ".join(c["text"] for c in n["claims"]) + n["summary"]
    assert "the signal is lost" in text and "whereas letting" in text
    p, u = load("producer_consumer_bad_order")
    text = template_narrate(p, analyze(p))["summary"]
    assert "semaphore `full`" in text.replace("blocked on ", "") or "`full`" in text
    p, u = load("starvation_unfair_mutex")
    text = template_narrate(p, analyze(p, fairness="weak"))["summary"]
    assert "weak fairness" in text and re.search(r"from step \d+ to step \d+", text)


def test_checker_catches_an_invented_semaphore_or_condition_name():
    p, u = load("lost_signal")
    v = analyze(p)
    n = template_narrate(p, v)
    n["summary"] += " Then `ghost` is signalled."
    assert not check(n, v, p).ok


def test_checker_knows_semaphore_and_condition_names():
    p, u = load("producer_consumer_bad_order")
    v = analyze(p)
    n = template_narrate(p, v)
    n["summary"] += " The semaphore `full` is never posted while `m` is held."
    assert check(n, v, p).ok


# ------------------------------------------------------------------ parser extensions
def test_parser_builds_semaphores_conditions_and_wait_expansion():
    p = parse("mutex m\nsemaphore s = 2\ncond c\nthread T:\n  lock m\n  wait c, m\n"
              "  signal c\n  broadcast c\n  down s\n  up s\n  unlock m\n").program
    assert p["semaphores"] == {"s": 2} and p["conds"] == ["c"]
    assert p["threads"]["T"] == [("lock", "m"), ("cwait", "c", "m"), ("cresume", "c", "m"),
                                 ("signal", "c"), ("broadcast", "c"), ("down", "s"),
                                 ("up", "s"), ("unlock", "m")]


def test_semaphore_defaults_to_zero_and_legacy_programs_have_no_extra_keys():
    assert parse("semaphore s\nthread T:\n  down s\n").program["semaphores"] == {"s": 0}
    legacy = parse("mutex m\nthread T:\n  lock m\n  unlock m\n").program
    assert set(legacy) == {"mutexes", "shared", "threads"}  # hash of old programs unchanged


def test_forever_becomes_a_back_edge():
    p = parse("mutex m\nthread T:\n  lock m\n  forever:\n    unlock m\n    lock m\n").program
    assert p["threads"]["T"] == [("lock", "m"), ("unlock", "m"), ("lock", "m"), ("jump", 1)]


@pytest.mark.parametrize("src,line,fragment", [
    ("semaphore s\nthread T:\n  down q\n", 3, "semaphore 'q' is not declared"),
    ("cond c\nthread T:\n  signal d\n", 3, "condition 'd' is not declared"),
    ("cond c\nthread T:\n  wait c, m\n", 3, "mutex 'm' is not declared"),
    ("mutex m\nthread T:\n  wait m\n", 3, "wait cond, mutex"),
    ("mutex m\nthread T:\n  repeat 2:\n    forever:\n      lock m\n", 4, "top level"),
    ("mutex m\nthread T:\n  forever:\n    lock m\n  unlock m\n", 5, "never reached"),
    ("mutex m\nthread T:\n  forever:\n", 3, "indented body"),
    ("semaphore a\nmutex a\n", 2, "declared twice"),
    ("semaphore s = x\n", 1, "semaphore s = 1"),
    ("mutex m\nthread T:\n  acquire m\n", 3, "down s"),
])
def test_new_syntax_errors_are_friendly(src, line, fragment):
    with pytest.raises(ParseError) as e:
        parse(src)
    assert e.value.line == line and fragment in e.value.message


def test_canonical_hash_ignores_renaming_of_semaphores_and_conditions():
    a = parse("mutex m\nsemaphore s = 1\ncond c\nthread T:\n  down s\n  lock m\n  signal c\n"
              "  unlock m\n  up s\n").program
    b = parse("mutex lk\nsemaphore permits = 1\ncond ready\nthread W:\n  down permits\n"
              "  lock lk\n  signal ready\n  unlock lk\n  up permits\n").program
    assert canonical_hash(a) == canonical_hash(b)
    c = parse("mutex m\nsemaphore s = 2\ncond c\nthread T:\n  down s\n  lock m\n  signal c\n"
              "  unlock m\n  up s\n").program
    assert canonical_hash(a) != canonical_hash(c)  # initial count matters
