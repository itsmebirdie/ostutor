"""Interleaving explorer: Tier A (state graph), Tier B (symmetry), Tier C (PCT).

    analyze(program)                 auto: Tier A; on budget overflow Tier B; then Tier C
    analyze(program, tier="A"|"B"|"C")
    replay(program, witness)         independent re-execution (works for every tier)

This engine is the default for every program. The review-slice explorer is kept as an
independent oracle (`engine="slice"`): the differential tests require the two to agree on
every classic program, and the general engine adds the `findings` list (a deadlock no longer
hides a race in the same program).
"""
from __future__ import annotations

from ostutor.common.types import Trace, Verdict, program_hash
from ostutor.explorer.model import GModel, is_legacy
from ostutor.explorer.pct import analyze_pct
from ostutor.explorer.reduce import make_canon
from ostutor.explorer.replay import replay as _general_replay
from ostutor.explorer.search import DEFAULT_BUDGET, explore, label_states
from ostutor.explorer.starvation import find_starvation
from ostutor.explorer.witness import critical_point, witness_from_chain
from ostutor.slice.explorer import StateBudgetExceeded
from ostutor.slice.replay import ReplayResult

SEVERITY = ("lost_signal", "deadlock", "race", "starvation")


def analyze_full(program: dict, budget: int = DEFAULT_BUDGET, loop_unroll: int = 0,
                 tier: str = "A", fairness: str = "weak"):
    """Exhaustive search (Tier A, or Tier B with symmetry). Returns (verdict, ex, labels)."""
    m = GModel(program)
    canon = make_canon(m) if tier == "B" else None
    if tier == "B" and canon is None:
        raise ValueError("Tier B reduction does not apply: the program has no symmetry "
                         "(or has loops, which need Tier A for starvation analysis)")
    ex = explore(m, budget, canon)
    primary_bad = {"lost_signal": ex.lost_states, "deadlock": ex.deadlocks,
                   "race": ex.race_states}
    starve = find_starvation(ex, fairness) if (m.cyclic and not ex.reduced) else None
    present = [k for k in SEVERITY if (primary_bad.get(k) or (k == "starvation" and starve))]
    label = present[0] if present else "safe"
    bad = set(primary_bad.get(label, []))
    labels = label_states(ex, bad)

    witness = cp = None
    detail: dict = {"findings": present}
    params = {"search": "tierB_symmetry_bfs" if ex.reduced else "tierA_bfs"}
    if label == "starvation" and starve is not None:
        from ostutor.explorer.witness import build_trace
        pre, cyc = starve["prefix"], starve["cycle"]
        steps = [(ex.rep[c], t) for c, t in pre + cyc]
        witness = build_trace(m, steps, {**params, "cycle_start": len(pre),
                                         "starved": starve["thread"],
                                         "fairness": fairness})
        detail.update(starved=starve["thread"], fairness=fairness, cycle_start=len(pre) + 1,
                      cycle_end=len(pre) + len(cyc))
    elif bad:
        target = next(iter(primary_bad[label]))
        from ostutor.explorer.search import chain_to
        chain = chain_to(ex, target)
        witness = witness_from_chain(ex, chain, params)
        cp = critical_point(ex, chain, target, labels)
        final = ex.rep[target]
        if label == "race":
            var, t1, o1, t2, o2 = m.races(final)[0]
            detail.update(variable=var, threads=[t1, t2],
                          pending={t1: f"{o1} {var}", t2: f"{o2} {var}"})
        else:
            detail["waits"] = {t: {"kind": a, "object": b, "holder": c}
                               for t, (a, b, c) in m.waits_for(final).items()}
            if label == "lost_signal":
                detail["lost"] = [{"thread": t, "cond": c} for t, c in m.lost_waiters(final)]
    else:
        detail["terminal_states"] = len(ex.terminals)
        if m.cyclic and not ex.reduced:
            detail["fairness_checked"] = fairness

    verdict = Verdict(
        label=label,
        completeness="reduced_but_complete" if ex.reduced else "complete_for_bound",
        bounds={"loop_unroll": loop_unroll, "buffer": 0, "state_budget": budget},
        states_explored=len(ex.order),
        fairness_assumption=fairness if label == "starvation" else None,
        witness=witness, critical_point=cp, program_hash=program_hash(program), detail=detail)
    return verdict, ex, labels


def analyze(program: dict, budget: int = DEFAULT_BUDGET, loop_unroll: int = 0,
            tier: str = "auto", fairness: str = "weak", engine: str = "auto",
            pct_runs: int = 2000, pct_depth: int = 3, seed: int = 0) -> Verdict:
    if engine == "slice":
        if not is_legacy(program):
            raise ValueError("the slice explorer only handles lock/unlock/read/write programs")
        from ostutor.slice.explorer import analyze as slice_analyze
        return slice_analyze(program, budget, loop_unroll)
    if tier == "C":
        return analyze_pct(program, pct_runs, pct_depth, seed, loop_unroll)
    if tier in ("A", "B"):
        return analyze_full(program, budget, loop_unroll, tier, fairness)[0]
    try:  # auto: exact first, then reduced, then sampled
        return analyze_full(program, budget, loop_unroll, "A", fairness)[0]
    except StateBudgetExceeded:
        pass
    if make_canon(GModel(program)) is not None:
        try:
            return analyze_full(program, budget, loop_unroll, "B", fairness)[0]
        except StateBudgetExceeded:
            pass
    return analyze_pct(program, pct_runs, pct_depth, seed, loop_unroll)


def replay(program: dict, witness: Trace, expected: str | None = None) -> ReplayResult:
    if is_legacy(program) and "cycle_start" not in (witness.params or {}):
        from ostutor.slice.replay import replay as slice_replay
        return slice_replay(program, witness, expected)
    return _general_replay(program, witness, expected)
