"""UI-independent helpers for the Streamlit demo (pure Python, unit-tested)."""
from __future__ import annotations

import math

from ostutor.common.types import Verdict
from ostutor.explorer import analyze_full
from ostutor.slice import rr
from ostutor.slice.explorer import StateBudgetExceeded, explore

MAX_GRAPH_NODES = 100
_FILL = {"ALWAYS_SAFE": "#c8e6c9", "MAY_FAIL": "#fff3b0", "MUST_FAIL": "#ffcdd2"}


def interleavings(lengths: list[int]) -> int:
    """Number of distinct schedules: the multinomial coefficient."""
    out = math.factorial(sum(lengths))
    for n in lengths:
        out //= math.factorial(n)
    return out


def witness_rows(verdict: Verdict) -> list[dict]:
    if not verdict.witness:
        return []
    return [{"Step": s.i, "Thread": s.thread, "Operation": f"{s.op} {s.object}",
             "Blocked after": ", ".join(s.blocked) or "-"} for s in verdict.witness.steps]


def state_graph_dot(program: dict, loop_unroll: int = 0,
                    max_nodes: int = MAX_GRAPH_NODES) -> str | None:
    """Graphviz DOT of the explored state graph, or None when it is too big to read.

    Node colour = backward-propagated label (green ALWAYS_SAFE, yellow MAY_FAIL, red
    MUST_FAIL). The shortest witness (or the starvation cycle) is drawn with thick red
    edges. Works for every program the explorer supports (semaphores, conditions, loops).
    """
    try:
        verdict, ex, labels = analyze_full(program, budget=max_nodes, loop_unroll=loop_unroll)
    except StateBudgetExceeded:
        return None
    m = ex.model
    ids = {s: f"s{i}" for i, s in enumerate(ex.order)}
    path: set = set()
    bad = None
    if verdict.witness:
        state = m.initial()
        for st in verdict.witness.steps:
            nxt = m.step(state, m.threads.index(st.thread))
            path.add((state, nxt))
            state = nxt
        if verdict.label in ("deadlock", "lost_signal", "race"):
            bad = state

    def name(s) -> str:
        pcs, owners, sems, waits, _ = s
        pos = "  ".join(f"{t}@{pc}" for t, pc in zip(m.threads, pcs))
        extra = [f"{mu}={m.threads[o]}" for mu, o in zip(m.mutexes, owners) if o >= 0]
        extra += [f"{nm}:{v}" for nm, v in zip(m.sem_names, sems)]
        extra += [f"{c} waiting: {','.join(m.threads[t] for t in q)}"
                  for c, q in zip(m.conds, waits) if q]
        return pos + (f"\\n[{'; '.join(extra)}]" if extra else "")

    lines = ["digraph G {", "  rankdir=LR;",
             '  node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=10];',
             '  edge [fontname="Helvetica", fontsize=9, color="#888888"];']
    for s in ex.order:
        extra = ', penwidth=3, color="#b71c1c"' if s == bad else ""
        if s == ex.initial:
            extra += ', peripheries=2'
        lines.append(f'  {ids[s]} [label="{name(s)}", fillcolor="{_FILL[labels[s]]}"{extra}];')
    for s in ex.order:
        for t, s2 in ex.edges[s]:
            hot = (s, s2) in path
            style = ', color="#b71c1c", penwidth=3' if hot else ""
            lines.append(f'  {ids[s]} -> {ids[s2]} [label="{m.threads[t]}"{style}];')
    lines.append("}")
    return "\n".join(lines)


def scaling_rows(max_steps: int = 10) -> list[dict]:
    """Two threads of n visible steps each: schedules vs distinct states."""
    rows = []
    for n in range(1, max_steps + 1):
        prog = {"mutexes": [], "shared": ["x"],
                "threads": {"T1": [("read", "x")] * n, "T2": [("read", "x")] * n}}
        states = len(explore(prog).order)
        sched = interleavings([n, n])
        rows.append({"steps_per_thread": n, "interleavings": sched, "states": states,
                     "ratio": round(sched / states, 1)})
    return rows


def parse_gantt(text: str) -> list[str]:
    return [tok for tok in text.replace(",", " ").split() if tok]


def rr_view(procs: list[tuple], quantum: int, student: list[str]) -> dict:
    """Correct trace, diagnosis, and the index of the first divergent time unit."""
    correct = rr.simulate(procs, quantum)
    diag = rr.diagnose(student, procs, quantum)
    div = diag.first_divergence["step"] if diag.first_divergence else None
    return {"correct": correct, "student": student, "diagnosis": diag, "divergence": div}


def example_student_answer(procs: list[tuple], quantum: int) -> list[str]:
    """The answer a student with the 'preempted queued first' misconception would give."""
    return rr.simulate(procs, quantum, "preempted_first")
