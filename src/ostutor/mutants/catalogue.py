"""Mutant policies (Build Plan 8.4): deliberately wrong variants of the simulators, each
tied to a named misconception and a one-line plain-language description.

A Problem is topic + algorithm + the simulator's inputs; `solve` runs the correct
simulator, `mutants_for` lists the misconceptions that apply to it.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from ostutor.sims import banker, cpu, disk, paging
from ostutor.sims.common import SimTrace, ValidationError


@dataclass
class Problem:
    topic: str       # "cpu" | "banker" | "paging" | "disk"
    algorithm: str
    params: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> Problem:
        if not isinstance(d, dict) or "topic" not in d or "algorithm" not in d:
            raise ValidationError("a problem needs 'topic' and 'algorithm'")
        return cls(d["topic"], d["algorithm"], dict(d.get("params", {})))


def solve(p: Problem) -> SimTrace:
    a = p.params
    try:
        if p.topic == "cpu":
            kw = {k: a[k] for k in ("quantum", "rr_tie", "quanta") if k in a}
            return cpu.simulate([tuple(x) for x in a["procs"]], p.algorithm, **kw)
        if p.topic == "paging":
            return paging.simulate(a["refs"], a["frames"], p.algorithm)
        if p.topic == "disk":
            kw = {k: a[k] for k in ("disk_size", "direction", "count_return") if k in a}
            return disk.simulate(a["requests"], a["head"], p.algorithm, **kw)
        if p.topic == "banker":
            if p.algorithm == "safety":
                return banker.safety(a["available"], a["maximum"], a["allocation"])
            if p.algorithm == "request":
                return banker.request(a["available"], a["maximum"], a["allocation"],
                                      a["process"], a["request"])
    except KeyError as e:
        raise ValidationError(f"missing input {e}") from e
    raise ValidationError(f"unknown topic/algorithm: {p.topic}/{p.algorithm}")


@dataclass(frozen=True)
class Mutant:
    id: str
    topic: str
    algorithm: str
    description: str
    run: Callable[[Problem], list[str]]
    accepts: Callable[[Problem, list[str]], bool] | None = None  # for non-unique answers

    def matches(self, p: Problem, student: list[str]) -> bool:
        if self.accepts is not None:
            return self.accepts(p, student)
        return self.run(p) == student


def _cpu(p: Problem, algo=None, **over) -> list[str]:
    a = p.params
    kw = {k: a[k] for k in ("quantum", "rr_tie", "quanta") if k in a}
    kw.update(over)
    return cpu.simulate([tuple(x) for x in a["procs"]], algo or p.algorithm, **kw).answer


def _invert_priority(p: Problem) -> list[str]:
    procs = [(*x[:3], -(x[3] if len(x) > 3 else 0)) for x in p.params["procs"]]
    return cpu.simulate(procs, "priority").answer


def _disk(p: Problem, algo=None, **over) -> list[str]:
    a = p.params
    kw = {k: a[k] for k in ("disk_size", "direction", "count_return") if k in a}
    kw.update(over)
    return disk.simulate(a["requests"], a["head"], algo or p.algorithm, **kw).answer


def _flip(p: Problem) -> str:
    return "down" if p.params.get("direction", "up") == "up" else "up"


def _banker_accepts(**flags):
    def accepts(p: Problem, student: list[str]) -> bool:
        a = p.params
        t = banker.safety(a["available"], a["maximum"], a["allocation"], **flags)
        if not t.metrics["safe"]:
            return student == ["UNSAFE"]
        return banker.is_valid_safe_sequence(a["available"], a["maximum"], a["allocation"],
                                             student, **flags)
    return accepts


def _banker_run(**flags):
    def run(p: Problem) -> list[str]:
        a = p.params
        return banker.safety(a["available"], a["maximum"], a["allocation"], **flags).answer
    return run


def _grant_without_safety_check(p: Problem) -> list[str]:
    a = p.params
    pid, req = a["process"], a["request"]
    need = [a["maximum"][pid][j] - a["allocation"][pid][j] for j in range(len(req))]
    if any(req[j] > need[j] for j in range(len(req))):
        return ["DENIED"]
    if any(req[j] > a["available"][j] for j in range(len(req))):
        return ["WAIT"]
    return ["GRANT"]


def _paging(p: Problem, algo=None, **over) -> list[str]:
    a = p.params
    return paging.simulate(a["refs"], a["frames"], algo or p.algorithm, **over).answer


CATALOGUE: list[Mutant] = [
    # ---- CPU scheduling
    Mutant("rr_preempted_before_new_arrival", "cpu", "rr",
           "Queues the preempted process before a process that arrives at the same instant.",
           lambda p: _cpu(p, rr_tie="preempted_first")),
    Mutant("mlfq_demote_on_preempt", "cpu", "mlfq",
           "Demotes a process to a lower queue when a higher queue preempts it, even though it "
           "had not used its whole quantum.",
           lambda p: _cpu(p, demote_on_preempt=True)),
    Mutant("srtf_not_preemptive", "cpu", "srtf",
           "Treats shortest-remaining-time-first as non-preemptive: never interrupts a running "
           "job when a shorter one arrives.",
           lambda p: _cpu(p, "sjf")),
    Mutant("sjf_preemptive", "cpu", "sjf",
           "Preempts the running job when a shorter one arrives (that is SRTF, not SJF).",
           lambda p: _cpu(p, "srtf")),
    Mutant("priority_larger_is_more_urgent", "cpu", "priority",
           "Treats a larger priority number as more urgent (the usual convention is smaller).",
           _invert_priority),
    # ---- Banker's algorithm
    Mutant("banker_max_instead_of_need", "banker", "safety",
           "Compares a process's Maximum, instead of its Need (Maximum - Allocation), with the "
           "available resources.",
           _banker_run(use_max_instead_of_need=True),
           _banker_accepts(use_max_instead_of_need=True)),
    Mutant("banker_forgets_to_reclaim", "banker", "safety",
           "Does not add a finished process's Allocation back to the available resources.",
           _banker_run(reclaim_allocation=False), _banker_accepts(reclaim_allocation=False)),
    Mutant("banker_skips_safety_check", "banker", "request",
           "Grants a request whenever the resources are available, without checking that the "
           "resulting state is safe.",
           _grant_without_safety_check),
    # ---- page replacement
    Mutant("lru_no_update_on_hit", "paging", "lru",
           "Does not refresh a page's recency when it is hit, so LRU degenerates to FIFO.",
           lambda p: _paging(p, lru_update_on_hit=False)),
    Mutant("fifo_update_on_hit", "paging", "fifo",
           "Moves a page to the back of the queue when it is hit (that is LRU, not FIFO).",
           lambda p: _paging(p, fifo_update_on_hit=True)),
    Mutant("opt_evicts_nearest", "paging", "opt",
           "Evicts the page that will be needed soonest instead of the one needed farthest "
           "in the future.",
           lambda p: _paging(p, opt_farthest=False)),
    Mutant("opt_confused_with_lru", "paging", "opt",
           "Evicts the least recently used page instead of looking at future references.",
           lambda p: _paging(p, "lru")),
    # ---- disk scheduling
    Mutant("sstf_picks_first_in_queue", "disk", "sstf",
           "Serves requests in arrival order instead of picking the nearest cylinder.",
           lambda p: _disk(p, "fcfs")),
    Mutant("scan_turns_at_last_request", "disk", "scan",
           "Reverses at the last request instead of travelling to the disk edge (that is LOOK).",
           lambda p: _disk(p, "look")),
    Mutant("cscan_serves_on_return", "disk", "cscan",
           "Serves requests on the return trip instead of jumping back to the start (that is "
           "SCAN).",
           lambda p: _disk(p, "scan")),
    Mutant("scan_wrong_initial_direction", "disk", "scan",
           "Starts the sweep in the opposite direction from the one stated.",
           lambda p: _disk(p, direction=_flip(p))),
]


def mutants_for(p: Problem) -> list[Mutant]:
    return [m for m in CATALOGUE if m.topic == p.topic and m.algorithm == p.algorithm]


def by_id(mutant_id: str) -> Mutant:
    return next(m for m in CATALOGUE if m.id == mutant_id)
