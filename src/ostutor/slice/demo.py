"""CLI demo.

  python -m ostutor.slice.demo deadlock_two_locks
  python -m ostutor.slice.demo deadlock_two_locks --tamper
  python -m ostutor.slice.demo --all
  python -m ostutor.slice.demo --scaling
  python -m ostutor.slice.demo --rr
"""
from __future__ import annotations

import argparse
import math
import sys

from ostutor.slice import rr
from ostutor.slice.check import check, tamper_examples
from ostutor.slice.explorer import analyze, explore
from ostutor.slice.narrate import narrate
from ostutor.slice.programs import PROGRAMS
from ostutor.slice.replay import replay


def _rule(title: str) -> None:
    print(f"\n=== {title} ===")


def run_program(name: str, tamper: bool = False) -> bool:
    program = PROGRAMS[name]
    v = analyze(program)
    _rule(f"{name}")
    print(f"label        : {v.label}")
    print(f"completeness : {v.completeness}  (bounds: {v.bounds})")
    print(f"states       : {v.states_explored}")
    lengths = [len(o) for o in program["threads"].values()]
    total = math.factorial(sum(lengths))
    for n in lengths:
        total //= math.factorial(n)
    print(f"interleavings: {total}  (state graph is what the search visits)")

    if v.witness:
        print("witness (shortest):")
        for s in v.witness.steps:
            blocked = f"   blocked after: {', '.join(s.blocked)}" if s.blocked else ""
            print(f"  {s.i}. {s.thread}: {s.op} {s.object}{blocked}")
        r = replay(program, v.witness, expected=v.label)
        print(f"replay       : {'reproduced ' + str(r.failure) if r.ok else 'FAILED - ' + r.reason}")
    cp = v.critical_point
    if cp:
        print(f"critical point: after step {cp['decision_after_step']}  "
              f"bad choice {cp['bad_choice']}  vs  safe alternative {cp['safe_alternative']}")
    elif v.label == "race":
        print("critical point: none (every schedule hits the race; it is in the code)")

    narration = narrate(program, v)
    res = check(narration, v, program)
    print("narration:")
    for c in narration["claims"]:
        print(f"  [step {c['step']}] {c['text']}")
    print(f"  summary: {narration['summary']}")
    print(f"checker      : {'CHECKED (facts only, not causal reasoning)' if res.ok else 'REJECTED ' + str(res.violations)}")

    ok = res.ok
    if tamper:
        print("\n-- tampering with the narration --")
        for kind, bad in tamper_examples(narration, v).items():
            r2 = check(bad, v, program)
            status = "rejected" if not r2.ok else "ACCEPTED (bug!)"
            why = r2.violations[0]["detail"] if r2.violations else ""
            print(f"  {kind:22s} -> {status}: {why}")
            ok = ok and not r2.ok
    return ok


def run_scaling() -> None:
    _rule("interleavings vs states: two threads, 10 steps each")
    prog = {"mutexes": [], "shared": ["x"],
            "threads": {"T1": [("read", "x")] * 10, "T2": [("read", "x")] * 10}}
    ex = explore(prog)
    print(f"interleavings: {math.comb(20, 10):,}")
    print(f"states visited: {len(ex.order):,}")


def run_rr() -> None:
    _rule("Round Robin mutant diagnosis (quantum 2)")
    procs = [("P1", 0, 5), ("P2", 1, 3), ("P3", 2, 1)]
    correct = rr.simulate(procs, 2)
    student = rr.simulate(procs, 2, "preempted_first")
    print("correct :", " ".join(correct))
    print("student :", " ".join(student))
    d = rr.diagnose(student, procs, 2)
    assert d.first_divergence is not None, "the mutant answer must diverge"
    print(f"method {d.method}; misconception {d.misconception_id}")
    print(f"first divergence at time unit {d.first_divergence['step']}: "
          f"expected {d.first_divergence['expected']}, got {d.first_divergence['got']}")
    print(f"note: {d.confidence_note}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("program", nargs="?", choices=sorted(PROGRAMS))
    ap.add_argument("--all", action="store_true", help="run all three programs")
    ap.add_argument("--tamper", action="store_true", help="show the checker rejecting bad narrations")
    ap.add_argument("--scaling", action="store_true", help="interleavings vs states")
    ap.add_argument("--rr", action="store_true", help="Round Robin mutant diagnosis")
    a = ap.parse_args(argv)
    if not (a.program or a.all or a.scaling or a.rr):
        ap.print_help()
        return 2
    ok = True
    names = sorted(PROGRAMS) if a.all else ([a.program] if a.program else [])
    for n in names:
        ok = run_program(n, a.tamper) and ok
    if a.scaling:
        run_scaling()
    if a.rr:
        run_rr()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
