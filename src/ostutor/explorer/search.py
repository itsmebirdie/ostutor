"""Tier A (and, with a canonicaliser, Tier B) breadth-first state-graph search.

Nodes are *canonical* states. Without reduction canonical == concrete. With symmetry
reduction each node keeps one concrete representative, and every edge is expanded from
that representative, so witness paths are always real schedules.
"""
from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, field

from ostutor.explorer.model import GModel, State
from ostutor.slice.explorer import StateBudgetExceeded

DEFAULT_BUDGET = 300_000


@dataclass
class Exploration:
    model: GModel
    initial: State
    order: list[State] = field(default_factory=list)
    rep: dict[State, State] = field(default_factory=dict)
    edges: dict[State, list[tuple[int, State]]] = field(default_factory=dict)
    parent: dict[State, tuple | None] = field(default_factory=dict)
    terminals: list[State] = field(default_factory=list)
    deadlocks: list[State] = field(default_factory=list)
    race_states: list[State] = field(default_factory=list)
    lost_states: list[State] = field(default_factory=list)
    reduced: bool = False


def explore(model: GModel, budget: int = DEFAULT_BUDGET,
            canon: Callable[[State], State] | None = None) -> Exploration:
    reduced = canon is not None
    canon = canon or (lambda s: s)
    init = model.initial()
    ci = canon(init)
    ex = Exploration(model, ci, reduced=reduced)
    ex.rep[ci], ex.parent[ci] = init, None
    queue = deque([ci])
    while queue:
        c = queue.popleft()
        ex.order.append(c)
        s = ex.rep[c]
        if model.is_terminal(s):
            ex.terminals.append(c)
            ex.edges[c] = []
            continue
        if model.races(s):
            ex.race_states.append(c)
        enabled = model.enabled(s)
        if not enabled:
            ex.deadlocks.append(c)
            if model.lost_waiters(s):
                ex.lost_states.append(c)
            ex.edges[c] = []
            continue
        out = []
        for t in enabled:
            s2 = model.step(s, t)
            c2 = canon(s2)
            if c2 not in ex.parent:
                if len(ex.parent) >= budget:
                    raise StateBudgetExceeded(f"more than {budget} states")
                ex.parent[c2] = (c, t)
                ex.rep[c2] = s2
                queue.append(c2)
            out.append((t, c2))
        ex.edges[c] = out
    return ex


def label_states(ex: Exploration, bad: set) -> dict:
    """ALWAYS_SAFE / MAY_FAIL / MUST_FAIL for every state.

    A state keeps a *safe completion* if some path from it either reaches a terminal
    state or runs forever, without ever visiting a bad state (greatest fixed point; on
    acyclic graphs this is the usual "can still reach a clean terminal").
    """
    preds = defaultdict(list)
    for s, outs in ex.edges.items():
        for _, s2 in outs:
            preds[s2].append(s)
    terminal = set(ex.terminals)
    safe = {s for s in ex.order if s not in bad}
    live = {s: sum(1 for _, s2 in ex.edges[s] if s2 in safe) for s in safe}
    work = deque(s for s in safe if s not in terminal and live[s] == 0)
    while work:
        s = work.popleft()
        if s not in safe:
            continue
        safe.discard(s)
        for p in preds[s]:
            if p in safe and p not in terminal:
                live[p] -= 1
                if live[p] == 0:
                    work.append(p)
    reach_bad = set(bad)
    work = deque(bad)
    while work:
        for p in preds[work.popleft()]:
            if p not in reach_bad:
                reach_bad.add(p)
                work.append(p)
    return {s: ("ALWAYS_SAFE" if s not in reach_bad else "MUST_FAIL" if s not in safe
                else "MAY_FAIL") for s in ex.order}


def chain_to(ex: Exploration, target: State) -> list[tuple[State, int]]:
    """[(canonical state before step, thread index)] from the initial state to target."""
    out = []
    cur = target
    while (link := ex.parent[cur]) is not None:
        prev, t = link
        out.append((prev, t))
        cur = prev
    out.reverse()
    return out
