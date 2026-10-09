"""Template narration (no LLM). Output follows the Narration contract:
{"claims": [{"step": int, "text": str}, ...], "summary": str}

Convention (the checker relies on it): every thread / mutex / variable name is
written in backticks, and the summary states the outcome as "Verdict: `label`".
"""
from __future__ import annotations

from ostutor.common.types import Verdict
from ostutor.slice.explorer import Model

_VERB = {"lock": "locks", "unlock": "unlocks", "read": "reads", "write": "writes"}


def _q(name: str) -> str:
    return f"`{name}`"


def narrate(program: dict, verdict: Verdict) -> dict:
    m = Model(program)
    claims = []
    state = m.initial()
    if verdict.witness:
        for st in verdict.witness.steps:
            t = m.threads.index(st.thread)
            state = m.step(state, t)
            text = f"{_q(st.thread)} {_VERB[st.op]} {_q(st.object)}."
            for waiter, (mu, owner) in sorted(m.blocked_on(state).items()):
                text += f" {_q(waiter)} is now blocked on {_q(mu)}, held by {_q(owner)}."
            claims.append({"step": st.i, "text": text})

    n = verdict.states_explored
    tail = (f" The search completed within its bounds (complete_for_bound) "
            f"after exploring {n} states.")
    label = verdict.label
    if label == "deadlock":
        waits = "; ".join(
            f"{_q(w)} waits for {_q(mu)} held by {_q(o)}"
            for w, (mu, o) in sorted(m.blocked_on(state).items()))
        summary = (f"Verdict: `deadlock`. After {len(claims)} steps no thread can run: "
                   f"{waits}.")
        cp = verdict.critical_point
        if cp:
            d = cp["decision_after_step"]
            when = "At the start" if d == 0 else f"After step {d}"
            summary += (f" {when}, {_q(cp['bad_thread'])} {_VERB[cp['bad_op']]} "
                        f"{_q(cp['bad_object'])} leaves the safe region")
            if cp.get("safe_thread"):
                summary += (f", while {_q(cp['safe_thread'])} {_VERB[cp['safe_op']]} "
                            f"{_q(cp['safe_object'])} keeps a safe completion reachable")
            summary += "."
    elif label == "race":
        d = verdict.detail
        t1, t2 = d["threads"]
        summary = (f"Verdict: `race`. {_q(t1)} is about to {d['pending'][t1].split()[0]} "
                   f"{_q(d['variable'])} while {_q(t2)} is about to "
                   f"{d['pending'][t2].split()[0]} it, no common lock protects the "
                   f"access, and at least one of them is a write.")
    else:
        summary = ("Verdict: `safe`. Every explored schedule runs to completion with "
                   "no deadlock and no race.")
    return {"claims": claims, "summary": summary + tail}
