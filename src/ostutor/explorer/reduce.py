"""Tier B reduction: symmetry (Build Plan 8.2).

A thread permutation is an *automorphism* when renaming the threads, and the mutexes,
variables, semaphores and conditions they use, maps the program onto itself (for
example Dining Philosophers rotated by one seat). Reachability of deadlocks, races
and safe completions is identical for symmetric states, so we search the quotient.
Soundness rule from the plan: every reduction is differential-tested against Tier A.
"""
from __future__ import annotations

from collections.abc import Callable
from itertools import permutations

from ostutor.explorer.model import GModel, State


def _ref_positions(op: tuple) -> list[tuple[str, str]]:
    k = op[0]
    if k in ("lock", "unlock"):
        return [("m", op[1])]
    if k in ("read", "write"):
        return [("v", op[1])]
    if k in ("down", "up"):
        return [("s", op[1])]
    if k in ("cwait", "cresume"):
        return [("c", op[1]), ("m", op[2])]
    if k in ("signal", "broadcast"):
        return [("c", op[1])]
    return []


def _try(m: GModel, pi: tuple) -> dict | None:
    maps: dict[str, dict] = {"m": {}, "v": {}, "s": {}, "c": {}}
    rev: dict[str, dict] = {"m": {}, "v": {}, "s": {}, "c": {}}
    for i, j in enumerate(pi):
        a_ops, b_ops = m.ops[i], m.ops[j]
        if len(a_ops) != len(b_ops):
            return None
        for a, b in zip(a_ops, b_ops):
            if a[0] != b[0] or (a[0] == "jump" and a[1] != b[1]):
                return None
            for (ka, oa), (_, ob) in zip(_ref_positions(a), _ref_positions(b)):
                if maps[ka].setdefault(oa, ob) != ob or rev[ka].setdefault(ob, oa) != oa:
                    return None
    sems = m.sem_names
    for s, s2 in maps["s"].items():
        if m.sem_init[sems.index(s)] != m.sem_init[sems.index(s2)]:
            return None
    # extend each partial map to a bijection; unused declared names must map to unused ones
    full = {}
    for key, names in (("m", m.mutexes), ("v", m.shared), ("s", m.sem_names), ("c", m.conds)):
        used_src = set(maps[key])
        if used_src != set(maps[key].values()):
            return None
        full[key] = {x: maps[key].get(x, x) for x in names}
    return {"t": pi, "m": [m.mutexes.index(full["m"][x]) for x in m.mutexes],
            "s": [m.sem_names.index(full["s"][x]) for x in m.sem_names],
            "c": [m.conds.index(full["c"][x]) for x in m.conds]}


def automorphisms(m: GModel, max_perm_threads: int = 7) -> list[dict]:
    n = len(m.threads)
    if n < 2 or m.cyclic:
        return []
    cands = permutations(range(n)) if n <= max_perm_threads else (
        tuple((i + r) % n for i in range(n)) for r in range(n))
    out = []
    for pi in cands:
        g = _try(m, pi)
        if g is not None:
            out.append(g)
    return out


def make_canon(m: GModel) -> Callable[[State], State] | None:
    """A canonicalising function, or None when the program has no non-trivial symmetry."""
    group = automorphisms(m)
    if len(group) < 2:
        return None

    def apply(g: dict, s: State) -> State:
        pcs, owners, sems, waits, lost = s
        t = g["t"]
        npcs = [0] * len(pcs)
        for i, pc in enumerate(pcs):
            npcs[t[i]] = pc
        nown = [-1] * len(owners)
        for i, o in enumerate(owners):
            nown[g["m"][i]] = t[o] if o >= 0 else -1
        nsem = [0] * len(sems)
        for i, v in enumerate(sems):
            nsem[g["s"][i]] = v
        nwait = [()] * len(waits)
        nlost = 0
        for i, q in enumerate(waits):
            nwait[g["c"][i]] = tuple(t[x] for x in q)
            nlost |= (lost >> i & 1) << g["c"][i]
        return (tuple(npcs), tuple(nown), tuple(nsem), tuple(nwait), nlost)

    return lambda s: min(apply(g, s) for g in group)
