"""Independent re-execution of a witness for the generalised engine.

Shares only the small GModel state machine with the search. Besides finite failures
(deadlock, lost_signal, race) it verifies starvation *lassos*: the cycle must return to
its starting state, the starved thread must never run in it, and the cycle must be fair
under the stated assumption.
"""
from __future__ import annotations

from ostutor.common.types import Trace
from ostutor.explorer.model import GModel
from ostutor.slice.replay import ReplayResult


def _run(m: GModel, steps, state, start_no=1):
    for n, st in enumerate(steps, start=start_no):
        if st.i != n:
            return None, f"step numbering broken at {st.i}"
        if st.thread not in m.threads:
            return None, f"step {n}: unknown thread {st.thread}"
        t = m.threads.index(st.thread)
        if not m.can_run(state, t):
            return None, f"step {n}: {st.thread} is not enabled"
        op = m.next_op(state, t)
        if (op[0], op[1]) != (st.op, st.object):
            return None, f"step {n}: operation does not match program"
        state = m.step(state, t)
    return state, ""


def replay(program: dict, witness: Trace, expected: str | None = None) -> ReplayResult:
    m = GModel(program)
    p = witness.params or {}
    if "cycle_start" in p:
        return _replay_lasso(m, witness, expected)
    state, err = _run(m, witness.steps, m.initial())
    if state is None:
        return ReplayResult(False, None, err)
    if not m.is_terminal(state) and not m.enabled(state):
        failure = "lost_signal" if m.lost_waiters(state) else "deadlock"
        if expected == "deadlock" and failure == "lost_signal":
            failure = "deadlock"  # a lost signal is also a deadlock
    elif m.races(state):
        failure = "race"
    else:
        return ReplayResult(False, None, "schedule is legal but ends without a failure")
    if expected is not None and failure != expected:
        return ReplayResult(False, failure, f"expected {expected}, reproduced {failure}")
    return ReplayResult(True, failure, "failure reproduced")


def _replay_lasso(m: GModel, w: Trace, expected: str | None) -> ReplayResult:
    p = w.params
    k, fairness, starved = p["cycle_start"], p["fairness"], p["starved"]
    if starved not in m.threads:
        return ReplayResult(False, None, "unknown starved thread")
    T = m.threads.index(starved)
    entry, err = _run(m, w.steps[:k], m.initial())
    if entry is None:
        return ReplayResult(False, None, err)
    cyc = w.steps[k:]
    if not cyc:
        return ReplayResult(False, None, "empty cycle")
    states, state = [entry], entry
    for n, st in enumerate(cyc, start=k + 1):
        nxt, err = _run(m, [st], state, start_no=n)
        if nxt is None:
            return ReplayResult(False, None, err)
        states.append(nxt)
        state = nxt
    if state != entry:
        return ReplayResult(False, None, "the cycle does not return to its starting state")
    loop = states[:-1]
    ran = {m.threads.index(st.thread) for st in cyc}
    if T in ran:
        return ReplayResult(False, None, "the starved thread runs inside the cycle")
    if any(m.next_op(s, T) is None for s in loop):
        return ReplayResult(False, None, "the starved thread has finished")
    en = [set(m.enabled(s)) for s in loop]
    if fairness == "weak":
        if all(T in e for e in en):
            return ReplayResult(False, None, "unfair: starved thread is enabled throughout")
        unfair = [u for u in range(len(m.threads)) if u != T and u not in ran
                  and all(u in e for e in en)]
    else:
        if any(T in e for e in en):
            return ReplayResult(False, None, "unfair: starved thread is enabled in the cycle")
        unfair = [u for u in range(len(m.threads)) if u != T and u not in ran
                  and any(u in e for e in en)]
    if unfair:
        return ReplayResult(False, None, f"unfair cycle: {m.threads[unfair[0]]} is starved too")
    if expected is not None and expected != "starvation":
        return ReplayResult(False, "starvation", f"expected {expected}, reproduced starvation")
    return ReplayResult(True, "starvation", "starvation cycle reproduced")
