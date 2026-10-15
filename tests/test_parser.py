from pathlib import Path

import pytest

from ostutor.dsl.parser import ParseError, canonical_hash, parse
from ostutor.slice.explorer import analyze
from ostutor.slice.programs import PROGRAMS
from ostutor.slice.replay import replay

DATA = Path(__file__).parent.parent / "data" / "programs"

# labels reasoned by hand before running (dining: all three grab the left fork)
EXPECTED = {"deadlock_two_locks": "deadlock", "race_no_lock": "race",
            "safe_with_lock": "safe", "dining_3_deadlock": "deadlock",
            "dining_3_ordered": "safe", "loop_counter_locked": "safe"}


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_example_files_get_hand_verified_labels(name):
    p = parse((DATA / f"{name}.ost").read_text())
    v = analyze(p.program, loop_unroll=p.loop_unroll)
    assert v.label == EXPECTED[name]
    if v.witness:
        assert replay(p.program, v.witness, expected=v.label).ok


@pytest.mark.parametrize("name", ["deadlock_two_locks", "race_no_lock", "safe_with_lock"])
def test_parser_matches_hard_coded_programs(name):
    p = parse((DATA / f"{name}.ost").read_text())
    ref = PROGRAMS[name]
    assert p.program["threads"] == ref["threads"]
    assert p.program["mutexes"] == ref["mutexes"] and p.program["shared"] == ref["shared"]


def test_dining_deadlock_witness_is_three_steps():
    v = analyze(parse((DATA / "dining_3_deadlock.ost").read_text()).program)
    assert len(v.witness.steps) == 3  # every philosopher takes the left fork


def test_loop_is_unrolled_and_bound_recorded():
    p = parse("mutex m\nthread T1:\n  repeat 3:\n    lock m\n    unlock m\n")
    assert p.program["threads"]["T1"] == [("lock", "m"), ("unlock", "m")] * 3
    assert p.loop_unroll == 3
    v = analyze(p.program, loop_unroll=p.loop_unroll)
    assert v.bounds["loop_unroll"] == 3


def test_while_and_for_loops():
    src = "shared x\nthread T1:\n  while true:\n    read x\nthread T2:\n  for i in range(3):\n    write x\n"
    p = parse(src, default_unroll=2)
    assert p.program["threads"]["T1"] == [("read", "x")] * 2
    assert p.program["threads"]["T2"] == [("write", "x")] * 3


def test_equivalent_programs_share_a_canonical_hash():
    a = parse("mutex a, b\nthread T1:\n lock a\n lock b\n unlock b\n unlock a\n"
              "thread T2:\n lock b\n lock a\n unlock a\n unlock b\n").program
    b = parse("# renamed, different spacing\nmutex p,   q\n\nthread Alice:\n"
              "        lock p\n        lock q\n        unlock q\n        unlock p\n"
              "thread Bob:\n  lock q   // second\n  lock p\n  unlock p\n  unlock q\n").program
    assert canonical_hash(a) == canonical_hash(b)
    c = parse("mutex a, b\nthread T1:\n lock a\n lock b\n unlock b\n unlock a\n"
              "thread T2:\n lock a\n lock b\n unlock b\n unlock a\n").program
    assert canonical_hash(a) != canonical_hash(c)


@pytest.mark.parametrize("src,line,fragment", [
    ("mutex a\nthread T1:\n  lock b\n", 3, "mutex 'b' is not declared"),
    ("shared x\nthread T1:\n  read y\n", 3, "shared variable 'y' is not declared"),
    ("mutex a\nthread T1:\n  wait a\n", 3, "condition variables"),
    ("semaphore s 1\n", 1, "semaphores"),
    ("thread T1:\n  if x:\n    read x\n", 2, "conditionals"),
    ("mutex a\nthread T1\n  lock a\n", 2, "thread T1:"),
    ("mutex a\nthread T1:\nlock a\n", 2, "no statements"),
    ("mutex a\nthread T1:\n  repeat 99:\n    lock a\n", 3, "between 1 and"),
    ("mutex a\nthread T1:\n  frobnicate a\n", 3, "cannot understand"),
    ("mutex a\nmutex a\n", 2, "declared twice"),
])
def test_friendly_errors_with_line_numbers(src, line, fragment):
    with pytest.raises(ParseError) as e:
        parse(src)
    assert e.value.line == line and fragment in e.value.message


def test_empty_program_rejected():
    with pytest.raises(ParseError):
        parse("mutex a\n")
