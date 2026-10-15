"""replay(program, witness): independently re-execute a witness schedule.

This is the explorer's integrity check: it shares only the tiny Model state
machine with the search, not the search, the labelling or the witness code.
"""
from __future__ import annotations

from dataclasses import dataclass

from ostutor.common.types import Trace
from ostutor.slice.explorer import Model


@dataclass
class ReplayResult:
    ok: bool  # schedule was legal AND ended in a failure
    failure: str | None  # "deadlock" | "race" | None
    reason: str


def replay(program: dict, witness: Trace, expected: str | None = None) -> ReplayResult:
    m = Model(program)
    state = m.initial()
    for n, st in enumerate(witness.steps, start=1):
        if st.i != n:
            return ReplayResult(False, None, f"step numbering broken at {st.i}")
        if st.thread not in m.threads:
            return ReplayResult(False, None, f"step {n}: unknown thread {st.thread}")
        t = m.threads.index(st.thread)
        if t not in m.enabled(state):
            return ReplayResult(False, None, f"step {n}: {st.thread} is not enabled")
        if m.next_op(state, t) != (st.op, st.object):
            return ReplayResult(False, None, f"step {n}: operation does not match program")
        state = m.step(state, t)

    if not m.is_terminal(state) and not m.enabled(state):
        failure = "deadlock"
    elif m.races(state):
        failure = "race"
    else:
        return ReplayResult(False, None, "schedule is legal but ends without a failure")
    if expected is not None and failure != expected:
        return ReplayResult(False, failure, f"expected {expected}, reproduced {failure}")
    return ReplayResult(True, failure, "failure reproduced")
