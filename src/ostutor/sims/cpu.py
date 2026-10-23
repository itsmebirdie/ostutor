"""CPU scheduling: FCFS, SJF, SRTF, Priority, Round Robin, MLFQ.

A process is (pid, arrival, burst) or (pid, arrival, burst, priority; lower = more urgent).
The answer is the Gantt chart, one entry per time unit ("idle" when nothing runs).
Ties are always broken by (arrival, position in the input) unless a parameter says otherwise.
"""
from __future__ import annotations

from collections import deque

from ostutor.sims.common import SimTrace, ValidationError, is_int, require

IDLE = "idle"
ALGORITHMS = ("fcfs", "sjf", "srtf", "priority", "rr", "mlfq")


def _validate(procs: list[tuple]) -> list[dict]:
    require(len(procs) > 0, "at least one process is required")
    out, seen = [], set()
    for k, p in enumerate(procs):
        require(len(p) in (3, 4), f"process {k}: expected (pid, arrival, burst[, priority])")
        pid, arr, burst = str(p[0]), p[1], p[2]
        pri = p[3] if len(p) == 4 else 0
        require(pid not in seen, f"duplicate process id {pid}")
        require(is_int(arr) and arr >= 0, f"{pid}: arrival must be an integer >= 0")
        require(is_int(burst) and burst > 0, f"{pid}: burst must be an integer > 0")
        require(is_int(pri), f"{pid}: priority must be an integer")
        seen.add(pid)
        out.append({"pid": pid, "arrival": arr, "burst": burst, "priority": pri, "idx": k})
    return out


def _metrics(ps: list[dict], gantt: list[str]) -> dict:
    done: dict[str, int] = {}
    for t, pid in enumerate(gantt, start=1):
        if pid != IDLE:
            done[pid] = t
    per = {}
    for p in ps:
        ct = done[p["pid"]]
        ta = ct - p["arrival"]
        per[p["pid"]] = {"completion": ct, "turnaround": ta, "waiting": ta - p["burst"]}
    n = len(ps)
    return {"per_process": per,
            "avg_waiting": sum(v["waiting"] for v in per.values()) / n,
            "avg_turnaround": sum(v["turnaround"] for v in per.values()) / n}


def _pick_ready(ps, t, key):
    ready = [p for p in ps if p["arrival"] <= t and p["left"] > 0]
    return min(ready, key=key) if ready else None


def _nonpreemptive(ps, key) -> list[str]:
    gantt: list[str] = []
    t = 0
    while any(p["left"] > 0 for p in ps):
        p = _pick_ready(ps, t, key)
        if p is None:
            gantt.append(IDLE)
            t += 1
            continue
        gantt += [p["pid"]] * p["left"]
        t += p["left"]
        p["left"] = 0
    return gantt


def _srtf(ps) -> list[str]:
    gantt, t = [], 0
    while any(p["left"] > 0 for p in ps):
        p = _pick_ready(ps, t, lambda q: (q["left"], q["arrival"], q["idx"]))
        if p is None:
            gantt.append(IDLE)
        else:
            gantt.append(p["pid"])
            p["left"] -= 1
        t += 1
    return gantt


def _rr(ps, quantum: int, tie: str) -> list[str]:
    """tie = "arrival_first": a process arriving at the very instant a quantum expires is
    queued BEFORE the preempted process (the usual textbook convention);
    "preempted_first": the preempted process goes first."""
    order = sorted(ps, key=lambda p: (p["arrival"], p["idx"]))
    queue: deque = deque()
    nxt = 0
    gantt, t = [], 0

    def admit(limit: int, inclusive: bool):
        nonlocal nxt
        while nxt < len(order) and (order[nxt]["arrival"] < limit
                                    or (inclusive and order[nxt]["arrival"] == limit)):
            queue.append(order[nxt])
            nxt += 1

    while nxt < len(order) or queue:
        if not queue:
            if order[nxt]["arrival"] > t:
                gantt += [IDLE] * (order[nxt]["arrival"] - t)
                t = order[nxt]["arrival"]
            admit(t, True)
        p = queue.popleft()
        run = min(quantum, p["left"])
        gantt += [p["pid"]] * run
        t += run
        p["left"] -= run
        if tie == "arrival_first":
            admit(t, True)
            if p["left"] > 0:
                queue.append(p)
        else:
            admit(t, False)
            if p["left"] > 0:
                queue.append(p)
            admit(t, True)
    return gantt


def _mlfq(ps, quanta: list[int], demote_on_preempt: bool) -> list[str]:
    levels = len(quanta)
    queues: list[deque] = [deque() for _ in range(levels)]
    order = sorted(ps, key=lambda p: (p["arrival"], p["idx"]))
    nxt, t = 0, 0
    running = None  # (process, level, used)
    gantt: list[str] = []
    while nxt < len(order) or any(queues) or running:
        while nxt < len(order) and order[nxt]["arrival"] <= t:
            queues[0].append(order[nxt])
            nxt += 1
        if running:
            p, lv, used = running
            higher = any(queues[i] for i in range(lv))
            if higher:  # preempted by a job in a higher-priority queue
                nl = min(lv + 1, levels - 1) if demote_on_preempt else lv
                queues[nl].append(p)
                running = None
        if not running:
            top = next((i for i in range(levels) if queues[i]), None)
            if top is None:
                gantt.append(IDLE)
                t += 1
                continue
            running = (queues[top].popleft(), top, 0)
        p, lv, used = running
        gantt.append(p["pid"])
        p["left"] -= 1
        used += 1
        t += 1
        if p["left"] == 0:
            running = None
        elif used == quanta[lv]:
            queues[min(lv + 1, levels - 1)].append(p)  # used its whole quantum: demote
            running = None
        else:
            running = (p, lv, used)
    return gantt


def simulate(procs: list[tuple], algorithm: str, *, quantum: int = 2,
             rr_tie: str = "arrival_first", quanta: list[int] | None = None,
             demote_on_preempt: bool = False) -> SimTrace:
    ps = _validate(procs)
    if algorithm not in ALGORITHMS:
        raise ValidationError(f"unknown algorithm {algorithm!r}; choose from {ALGORITHMS}")
    for p in ps:
        p["left"] = p["burst"]
    params: dict = {"tie_break": "(arrival, input order)"}
    if algorithm == "fcfs":
        gantt = _nonpreemptive(ps, lambda q: (q["arrival"], q["idx"]))
    elif algorithm == "sjf":
        gantt = _nonpreemptive(ps, lambda q: (q["burst"], q["arrival"], q["idx"]))
    elif algorithm == "priority":
        gantt = _nonpreemptive(ps, lambda q: (q["priority"], q["arrival"], q["idx"]))
    elif algorithm == "srtf":
        gantt = _srtf(ps)
    elif algorithm == "rr":
        require(is_int(quantum) and quantum > 0, "quantum must be an integer > 0")
        require(rr_tie in ("arrival_first", "preempted_first"), "bad rr_tie")
        gantt = _rr(ps, quantum, rr_tie)
        params.update(quantum=quantum, rr_tie=rr_tie)
    else:
        quanta = quanta or [2, 4, 8]
        require(all(is_int(q) and q > 0 for q in quanta), "quanta must be integers > 0")
        gantt = _mlfq(ps, quanta, demote_on_preempt)
        params.update(quanta=quanta, demote_on_preempt=demote_on_preempt)
    m = _metrics(ps, gantt)
    rows = [{"pid": pid, **v} for pid, v in m["per_process"].items()]
    return SimTrace("cpu", algorithm, gantt, {k: v for k, v in m.items() if k != "per_process"}
                    | {"per_process": m["per_process"]}, rows, params)
