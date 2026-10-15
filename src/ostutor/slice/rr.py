"""Optional S7: Round Robin simulator, one mutant, divergence report.

Trace = per-time-unit list of running pids ("-" = idle), 1-based steps.
Correct convention (explicit parameter): when a quantum expires at the same
instant a new process arrives, the NEW arrival is queued BEFORE the preempted
process. Mutant `rr_preempted_before_new_arrival` reverses that tie.
Arrivals strictly inside a slice are always queued during the slice.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

CORRECT_POLICY = "new_first"
MUTANTS = {
    "rr_preempted_before_new_arrival": {
        "policy": "preempted_first",
        "description": "Preempted process is queued before a process arriving at the "
                       "same instant the quantum ends.",
    },
}


def simulate(procs: list[tuple], quantum: int, policy: str = CORRECT_POLICY) -> list[str]:
    """procs = [(pid, arrival, burst), ...]; returns the per-time-unit Gantt list."""
    if quantum <= 0:
        raise ValueError("quantum must be positive")
    for pid, arr, burst in procs:
        if arr < 0 or burst <= 0:
            raise ValueError(f"{pid}: arrival must be >= 0 and burst > 0")
    ps = sorted(procs, key=lambda p: (p[1], p[0]))
    rem = {p[0]: p[2] for p in ps}
    queue: deque = deque()
    gantt: list[str] = []
    i, t, done = 0, 0, 0
    while done < len(ps):
        while i < len(ps) and ps[i][1] <= t:
            queue.append(ps[i][0])
            i += 1
        if not queue:
            gantt.append("-")
            t += 1
            continue
        pid = queue.popleft()
        run = min(quantum, rem[pid])
        gantt += [pid] * run
        end = t + run
        while i < len(ps) and ps[i][1] < end:  # arrivals during the slice
            queue.append(ps[i][0])
            i += 1
        tie = []
        while i < len(ps) and ps[i][1] == end:  # arrivals exactly at slice end
            tie.append(ps[i][0])
            i += 1
        rem[pid] -= run
        if rem[pid] > 0:
            if policy == "new_first":
                queue.extend(tie)
                queue.append(pid)
            else:
                queue.append(pid)
                queue.extend(tie)
        else:
            done += 1
            queue.extend(tie)
        t = end
    return gantt


def first_divergence(correct: list[str], got: list[str]) -> dict | None:
    for k in range(max(len(correct), len(got))):
        e = correct[k] if k < len(correct) else "(end)"
        g = got[k] if k < len(got) else "(end)"
        if e != g:
            return {"step": k + 1, "expected": e, "got": g}
    return None


def edit_distance(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


@dataclass
class Diagnosis:
    method: str  # "correct" | "mutant_match" | "edit_distance"
    misconception_id: str | None
    first_divergence: dict | None
    systematic: bool | None
    confidence_note: str
    matches: list[str] = field(default_factory=list)  # ALL mutants that match


def diagnose(student: list[str], procs: list[tuple], quantum: int) -> Diagnosis:
    correct = simulate(procs, quantum, CORRECT_POLICY)
    div = first_divergence(correct, student)
    if div is None:
        return Diagnosis("correct", None, None, None, "Answer matches the correct trace.")
    matches = [mid for mid, mu in MUTANTS.items()
               if simulate(procs, quantum, mu["policy"]) == student]
    if matches:
        note = MUTANTS[matches[0]]["description"]
        if len(matches) > 1:
            note += " Several mutants match; this question cannot distinguish them."
        return Diagnosis("mutant_match", matches[0], div, True, note, matches)
    d = edit_distance(correct, student)
    dists = {mid: edit_distance(simulate(procs, quantum, mu["policy"]), student)
             for mid, mu in MUTANTS.items()}
    slip = d <= 2
    note = (f"No mutant matches. Edit distance to correct trace: {d}; to mutants: {dists}. "
            + ("Small distance suggests a random slip that cascades from the first "
               "divergence." if slip else "Large distance; LLM fall-back would be needed "
               "(not built in the slice)."))
    return Diagnosis("edit_distance", None, div, False if slip else None, note)
