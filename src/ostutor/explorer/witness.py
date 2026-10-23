"""Witness traces and the critical decision point (shortest-witness BFS lives in search)."""
from __future__ import annotations

from ostutor.common.types import Step, Trace
from ostutor.explorer.model import GModel
from ostutor.explorer.search import Exploration


def vocab(m: GModel) -> dict:
    return {"threads": list(m.threads), "objects": m.mutexes + m.sem_names + m.conds,
            "vars": list(m.shared)}


def stuck_threads(m: GModel, s) -> list[str]:
    return sorted(m.waits_for(s))


def build_trace(m: GModel, steps: list[tuple], params: dict) -> Trace:
    """steps: [(concrete state before, thread index)] -> Trace (concrete re-execution)."""
    out = []
    for i, (s, t) in enumerate(steps, start=1):
        op = m.ops[t][s[0][t]]
        s2 = m.step(s, t)
        out.append(Step(i, m.threads[t], op[0], op[1], m.delta(s, s2, t), stuck_threads(m, s2)))
    return Trace("concurrency", out, vocab(m), params)


def witness_from_chain(ex: Exploration, chain: list[tuple], params: dict) -> Trace:
    return build_trace(ex.model, [(ex.rep[c], t) for c, t in chain], params)


def critical_point(ex: Exploration, chain: list[tuple], target, labels: dict) -> dict | None:
    """Last state on the witness that still has a safe completion + the choice that left it."""
    m = ex.model
    states = [c for c, _ in chain] + [target]
    idx = None
    for k, s in enumerate(states[:-1]):
        if labels[s] != "MUST_FAIL":
            idx = k
    if idx is None:
        return None  # every schedule fails (for example a race in the code itself)
    prev, t = chain[idx]
    s = ex.rep[prev]
    kind, obj = m.ops[t][s[0][t]][:2]
    cp = {"step_index": idx + 1, "decision_after_step": idx,
          "bad_choice": f"{m.threads[t]}:{kind} {obj}",
          "bad_thread": m.threads[t], "bad_op": kind, "bad_object": obj,
          "safe_alternative": None}
    for t2, s2 in ex.edges[prev]:
        if labels[s2] != "MUST_FAIL":
            k2, o2 = m.ops[t2][s[0][t2]][:2]
            cp.update(safe_alternative=f"{m.threads[t2]}:{k2} {o2}", safe_thread=m.threads[t2],
                      safe_op=k2, safe_object=o2)
            break
    return cp
