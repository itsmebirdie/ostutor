"""Template narration for every explorer label and operation (no LLM).

Follows the Narration contract and the checker's conventions: every thread, mutex,
semaphore, condition and variable is in backticks, the summary starts
"Verdict: `label`", and the text contains no numbers except step references, the step
count, the state count and recorded bounds.
"""
from __future__ import annotations

from ostutor.common.types import Trace, Verdict
from ostutor.explorer.model import GModel

_PHRASE = {"lock": "locks", "unlock": "unlocks", "read": "reads", "write": "writes",
           "down": "takes a permit from semaphore", "up": "posts to semaphore",
           "cwait": "waits on condition", "cresume": "resumes on condition",
           "signal": "signals condition", "broadcast": "broadcasts on condition"}


_BASE = {"lock": "lock", "unlock": "unlock", "read": "read", "write": "write",
         "down": "take a permit from semaphore", "up": "post to semaphore",
         "cwait": "wait on condition", "cresume": "resume on condition",
         "signal": "signal condition", "broadcast": "broadcast on condition"}


def _q(name: str) -> str:
    return f"`{name}`"


def _stuck(reason: tuple) -> str:
    kind, obj, holder = reason
    if kind == "lock":
        return f"blocked on {_q(obj)}, held by {_q(holder)}"
    if kind == "semaphore":
        return f"blocked on semaphore {_q(obj)}"
    return f"waiting for a signal on {_q(obj)}"


def _walk(m: GModel, trace: Trace):
    """Yield (step, op, state_before, state_after) re-executing the witness."""
    state = m.initial()
    for st in trace.steps:
        t = m.threads.index(st.thread)
        op = m.next_op(state, t)
        after = m.step(state, t)
        yield st, op, state, after
        state = after


def _step_text(m: GModel, st, op, before, after) -> str:
    t = _q(st.thread)
    k = st.op
    if k in ("lock", "unlock", "read", "write", "down", "up"):
        text = f"{t} {_PHRASE[k]} {_q(st.object)}."
    elif k == "cwait":
        text = f"{t} releases {_q(op[2])} and waits on condition {_q(st.object)}."
    elif k == "cresume":
        text = f"{t} wakes up on condition {_q(st.object)} and re-acquires {_q(op[2])}."
    else:
        c = m.conds.index(st.object)
        waiting = before[3][c]
        if not waiting:
            text = (f"{t} signals condition {_q(st.object)}, but no thread is waiting, "
                    "so the signal is lost.")
        elif k == "signal":
            text = (f"{t} signals condition {_q(st.object)} and wakes "
                    f"{_q(m.threads[waiting[0]])}.")
        else:
            woken = ", ".join(_q(m.threads[x]) for x in waiting)
            text = f"{t} broadcasts on condition {_q(st.object)} and wakes {woken}."
    for waiter, reason in sorted(m.waits_for(after).items()):
        text += f" {_q(waiter)} is now {_stuck(reason)}."
    return text


def _tail(v: Verdict) -> str:
    n = v.states_explored
    if v.completeness == "sampled":
        return (" The search was sampled, not exhaustive, so a failure that was not found "
                "is not ruled out.")
    how = "within its bounds (complete_for_bound)" if v.completeness == "complete_for_bound" \
        else "using symmetry reduction (reduced_but_complete)"
    return f" The search completed {how} after exploring {n} states."


def narrate(program: dict, verdict: Verdict) -> dict:
    m = GModel(program)
    claims = []
    if verdict.witness:
        for st, op, before, after in _walk(m, verdict.witness):
            claims.append({"step": st.i, "text": _step_text(m, st, op, before, after)})
    label, d = verdict.label, verdict.detail

    if label in ("deadlock", "lost_signal"):
        waits = "; ".join(f"{_q(w)} is {_stuck((r['kind'], r['object'], r['holder']))}"
                          for w, r in sorted(d["waits"].items()))
        summary = (f"Verdict: `{label}`. After {len(claims)} steps no thread can run: "
                   f"{waits}.")
        for item in d.get("lost", []):
            summary += (f" A signal on {_q(item['cond'])} was sent while nobody was waiting, "
                        f"so {_q(item['thread'])} can never be woken.")
        cp = verdict.critical_point
        if cp:
            k = cp["decision_after_step"]
            when = "At the start" if k == 0 else f"After step {k}"
            summary += (f" {when}, letting {_q(cp['bad_thread'])} {_BASE[cp['bad_op']]} "
                        f"{_q(cp['bad_object'])} leaves the safe region")
            if cp.get("safe_thread"):
                summary += (f", whereas letting {_q(cp['safe_thread'])} "
                            f"{_BASE[cp['safe_op']]} {_q(cp['safe_object'])} keeps a safe "
                            "completion reachable")
            summary += "."
    elif label == "race":
        t1, t2 = d["threads"]
        summary = (f"Verdict: `race`. {_q(t1)} is about to {d['pending'][t1].split()[0]} "
                   f"{_q(d['variable'])} while {_q(t2)} is about to "
                   f"{d['pending'][t2].split()[0]} it, no common lock protects the "
                   "access, and at least one of them is a write.")
    elif label == "starvation":
        a, b = d["cycle_start"], d["cycle_end"]
        summary = (f"Verdict: `starvation`. Under {d['fairness']} fairness, the steps from "
                   f"step {a} to step {b} can repeat forever while {_q(d['starved'])} never "
                   "gets to run, although other threads keep making progress. This is a "
                   "warning that depends on the stated fairness assumption.")
    elif label == "none_found":
        runs = verdict.bounds["pct_runs"]
        summary = (f"Verdict: `none_found`. {runs} randomised schedules were tried and none "
                   "produced a deadlock, a race or a lost signal. This is evidence, not a "
                   "proof that the program is safe.")
    else:
        extra = ""
        if "fairness_checked" in d:
            extra = f" and no starvation under {d['fairness_checked']} fairness"
        summary = ("Verdict: `safe`. Every explored schedule runs to completion with no "
                   f"deadlock and no race{extra}." if not extra else
                   f"Verdict: `safe`. No deadlock, no race{extra} was found across every "
                   "reachable state.")
    if verdict.completeness == "sampled" and label != "none_found":
        summary += (" This schedule was found by randomised sampling, so it may not be the "
                    "shortest.")
    return {"claims": claims, "summary": summary + _tail(verdict)}
