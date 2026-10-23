"""Tier C: PCT-style randomised priority scheduling (Build Plan 8.2).

Per run the guarantee is at least 1/(n * k^(d-1)) of finding a bug of depth d
(n threads, k steps). Results are ALWAYS labelled `sampled`; "no bug found" is never
reported as safe.
"""
from __future__ import annotations

import random

from ostutor.common.types import Verdict, program_hash
from ostutor.explorer.model import GModel
from ostutor.explorer.witness import build_trace

MAX_STEPS_CYCLIC = 200


def _one_run(m: GModel, rng: random.Random, depth: int, k: int, max_steps: int, seen: set):
    n = len(m.threads)
    prio = list(range(depth, depth + n))
    rng.shuffle(prio)
    change = sorted(rng.sample(range(1, k + 1), min(depth - 1, k))) if depth > 1 else []
    s = m.initial()
    steps: list[tuple] = []
    for step_no in range(1, max_steps + 1):
        seen.add(s)
        if m.is_terminal(s):
            return None
        if m.races(s):
            return "race", steps, s
        en = m.enabled(s)
        if not en:
            return ("lost_signal" if m.lost_waiters(s) else "deadlock"), steps, s
        t = max(en, key=lambda x: prio[x])
        steps.append((s, t))
        s = m.step(s, t)
        if step_no in change:
            prio[t] = change.index(step_no) + 1  # drop below every initial priority
    return None


def analyze_pct(program: dict, runs: int = 2000, depth: int = 3, seed: int = 0,
                loop_unroll: int = 0) -> Verdict:
    m = GModel(program)
    total = sum(len(o) for o in m.ops)
    max_steps = MAX_STEPS_CYCLIC if m.cyclic else max(total, 1)
    k = max_steps
    rng = random.Random(seed)
    seen: set = set()
    found = None
    done = 0
    for done in range(1, runs + 1):  # noqa: B007 (used after the loop)
        found = _one_run(m, rng, depth, k, max_steps, seen)
        if found:
            break
    n = len(m.threads)
    bounds = {"loop_unroll": loop_unroll, "buffer": 0, "state_budget": 0,
              "pct_runs": runs, "pct_depth": depth}
    detail = {"runs_executed": done, "seed": seed,
              "per_run_probability_lower_bound": 1 / (n * k ** (depth - 1)),
              "note": "sampled: a failure that was not found is NOT ruled out"}
    if not found:
        return Verdict("none_found", "sampled", bounds, len(seen), None, None, None,
                       program_hash(program), detail)
    label, steps, final = found
    detail["found_after_runs"] = done
    if label == "race":
        var, t1, o1, t2, o2 = m.races(final)[0]
        detail.update(variable=var, threads=[t1, t2], pending={t1: f"{o1} {var}",
                                                              t2: f"{o2} {var}"})
    else:
        detail["waits"] = {t: {"kind": a, "object": b, "holder": c}
                           for t, (a, b, c) in m.waits_for(final).items()}
        if label == "lost_signal":
            detail["lost"] = [{"thread": t, "cond": c} for t, c in m.lost_waiters(final)]
    trace = build_trace(m, steps, {"search": "tierC_pct", "minimal": False})
    return Verdict(label, "sampled", bounds, len(seen), None, trace, None,
                   program_hash(program), detail)
