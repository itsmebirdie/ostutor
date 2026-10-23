"""Starvation = a reachable *fair* cycle in which thread T never executes (Build Plan 3.4).

Reported only under a stated fairness assumption:
  weak   (justice)    a thread that is enabled in EVERY state of a cycle must run in it
  strong (compassion) a thread enabled in ANY state of a cycle that repeats must run in it
T is starved when a cycle exists, fair for every thread, in which T is unfinished but
never runs while other threads do. Works on Tier A graphs of programs with `forever` loops.
"""
from __future__ import annotations

from collections import deque

from ostutor.explorer.search import Exploration, chain_to


def _sccs(nodes: set, adj: dict) -> list[list]:
    """Iterative Tarjan over `nodes` using adjacency adj[node] = [successor, ...]."""
    index, low, on, stack, out = {}, {}, set(), [], []
    counter = 0
    for root in nodes:
        if root in index:
            continue
        work = [(root, iter(adj.get(root, ())))]
        index[root] = low[root] = counter
        counter += 1
        stack.append(root)
        on.add(root)
        while work:
            v, it = work[-1]
            advanced = False
            for w in it:
                if w not in nodes:
                    continue
                if w not in index:
                    index[w] = low[w] = counter
                    counter += 1
                    stack.append(w)
                    on.add(w)
                    work.append((w, iter(adj.get(w, ()))))
                    advanced = True
                    break
                if w in on:
                    low[v] = min(low[v], index[w])
            if advanced:
                continue
            work.pop()
            if work:
                low[work[-1][0]] = min(low[work[-1][0]], low[v])
            if low[v] == index[v]:
                comp = []
                while True:
                    w = stack.pop()
                    on.discard(w)
                    comp.append(w)
                    if w == v:
                        break
                out.append(comp)
    return out


def _nontrivial(comp: list, adj: dict) -> bool:
    return len(comp) > 1 or comp[0] in adj.get(comp[0], ())


def find_starvation(ex: Exploration, fairness: str = "weak") -> dict | None:
    m = ex.model
    n = len(m.threads)
    enabled = {s: set(m.enabled(ex.rep[s])) for s in ex.order}
    for T in range(n):
        nodes = {s for s in ex.order
                 if ex.rep[s][0][T] < len(m.ops[T])  # T has not finished
                 and (fairness == "weak" or T not in enabled[s])}
        adj = {s: [s2 for t, s2 in ex.edges[s] if t != T and s2 in nodes] for s in nodes}
        for comp in _sccs(nodes, adj):
            found = _fair_component(set(comp), adj, ex, enabled, T, fairness, n)
            if found is not None:
                return _lasso(ex, found, adj, T, fairness)
    return None


def _fair_component(comp: set, adj: dict, ex: Exploration, enabled: dict, T: int,
                    fairness: str, n: int):
    sub = {s: [s2 for s2 in adj[s] if s2 in comp] for s in comp}
    if not _nontrivial(list(comp), sub):
        return None
    runs = {u: any(t == u for s in comp for t, s2 in ex.edges[s] if s2 in comp and t != T)
            for u in range(n) if u != T}
    if fairness == "weak":
        if not any(T not in enabled[s] for s in comp):
            return None  # T continuously enabled: weak fairness forces it to run
        for u in runs:
            if not runs[u] and all(u in enabled[s] for s in comp):
                return None  # u continuously enabled yet never runs: unfair cycle
        return comp
    for u in runs:  # strong fairness: refine until stable (Streett-style)
        if not runs[u] and any(u in enabled[s] for s in comp):
            rest = {s for s in comp if u not in enabled[s]}
            radj = {s: [s2 for s2 in sub[s] if s2 in rest] for s in rest}
            for c2 in _sccs(rest, radj):
                found = _fair_component(set(c2), radj, ex, enabled, T, fairness, n)
                if found is not None:
                    return found
            return None
    return comp


def _lasso(ex: Exploration, comp: set, adj: dict, T: int, fairness: str) -> dict:
    m = ex.model
    comp_adj = {s: [(t, s2) for t, s2 in ex.edges[s] if t != T and s2 in comp] for s in comp}
    entry = next(s for s in ex.order if s in comp)  # closest to the start (BFS order)
    enabled = {s: set(m.enabled(ex.rep[s])) for s in comp}
    # waypoints the closed walk must visit so that it is fair
    way: list[tuple] = []
    if fairness == "weak":
        way.append(("state", next(s for s in comp if T not in enabled[s])))
    for u in range(len(m.threads)):
        if u == T:
            continue
        edge = next(((s, t, s2) for s in comp for t, s2 in comp_adj[s] if t == u), None)
        if edge:
            way.append(("edge", edge))
        elif fairness == "weak":
            way.append(("state", next(s for s in comp if u not in enabled[s])))
    walk: list[tuple] = []
    cur = entry

    def go(src, dst):
        prev: dict = {src: None}
        q = deque([src])
        while q:
            s = q.popleft()
            if s == dst and (s != src or not walk):
                break
            for t, s2 in comp_adj[s]:
                if s2 not in prev:
                    prev[s2] = (s, t)
                    q.append(s2)
        path = []
        s = dst
        while prev.get(s) is not None:
            p, t = prev[s]
            path.append((p, t))
            s = p
        return path[::-1]

    for kind, w in way:
        if kind == "state":
            walk += go(cur, w)
            cur = w
        else:
            s, t, s2 = w
            walk += go(cur, s)
            walk.append((s, t))
            cur = s2
    walk += go(cur, entry)
    if not walk:  # a self-loop is the whole cycle
        t, s2 = comp_adj[entry][0]
        walk = [(entry, t)]
    return {"thread": m.threads[T], "fairness": fairness, "prefix": chain_to(ex, entry),
            "cycle": walk}
