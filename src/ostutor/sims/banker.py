"""Banker's algorithm: safety check and resource-request algorithm (Dijkstra).

State: available (m), maximum (n x m), allocation (n x m); need = maximum - allocation.
Safe-sequence tie rule: in each round pick the LOWEST-index unfinished process whose need
fits in the work vector (this reproduces the textbook sequence).
"""
from __future__ import annotations

from ostutor.sims.common import SimTrace, ValidationError, is_int, require


def _validate(available, maximum, allocation) -> list[list[int]]:
    m = len(available)
    require(m > 0 and all(is_int(x) and x >= 0 for x in available),
            "available must be a non-empty list of integers >= 0")
    n = len(maximum)
    require(n > 0 and len(allocation) == n, "maximum and allocation need the same number of rows")
    for name, mat in (("maximum", maximum), ("allocation", allocation)):
        for i, row in enumerate(mat):
            require(len(row) == m, f"{name} row {i} has {len(row)} columns, expected {m}")
            require(all(is_int(x) and x >= 0 for x in row), f"{name} row {i}: values must be "
                                                           "integers >= 0")
    need = []
    for i in range(n):
        for j in range(m):
            require(allocation[i][j] <= maximum[i][j],
                    f"P{i}: allocation exceeds maximum for resource {j}")
        need.append([maximum[i][j] - allocation[i][j] for j in range(m)])
    return need


def safety(available: list[int], maximum: list[list[int]], allocation: list[list[int]],
           *, use_max_instead_of_need: bool = False, reclaim_allocation: bool = True) -> SimTrace:
    """The two flags exist only so the mutant catalogue can reuse the loop."""
    need = _validate(available, maximum, allocation)
    n, m = len(maximum), len(available)
    compare = maximum if use_max_instead_of_need else need
    work = list(available)
    finished = [False] * n
    seq: list[str] = []
    rows = []
    while True:
        pick = next((i for i in range(n) if not finished[i]
                     and all(compare[i][j] <= work[j] for j in range(m))), None)
        if pick is None:
            break
        if reclaim_allocation:
            work = [work[j] + allocation[pick][j] for j in range(m)]
        finished[pick] = True
        seq.append(f"P{pick}")
        rows.append({"process": f"P{pick}", "work_after": list(work)})
    safe = all(finished)
    answer = seq if safe else ["UNSAFE"]
    return SimTrace("banker", "safety", answer, {"safe": safe, "need": need}, rows,
                    {"tie_break": "lowest-index process that fits"})


def request(available: list[int], maximum: list[list[int]], allocation: list[list[int]],
            pid: int, req: list[int]) -> SimTrace:
    need = _validate(available, maximum, allocation)
    n, m = len(maximum), len(available)
    require(is_int(pid) and 0 <= pid < n, f"process index must be 0..{n - 1}")
    require(len(req) == m and all(is_int(x) and x >= 0 for x in req),
            f"request must be {m} integers >= 0")
    if not safety(available, maximum, allocation).metrics["safe"]:
        raise ValidationError("the initial state is unsafe; the request algorithm needs a safe "
                              "starting state")
    params = {"process": f"P{pid}", "request": list(req)}
    if any(req[j] > need[pid][j] for j in range(m)):
        return SimTrace("banker", "request", ["DENIED"], {"granted": False,
                        "reason": "request exceeds the process's declared maximum claim"},
                        [], params)
    if any(req[j] > available[j] for j in range(m)):
        return SimTrace("banker", "request", ["WAIT"], {"granted": False,
                        "reason": "resources are not available; the process must wait"},
                        [], params)
    avail2 = [available[j] - req[j] for j in range(m)]
    alloc2 = [list(r) for r in allocation]
    alloc2[pid] = [alloc2[pid][j] + req[j] for j in range(m)]
    after = safety(avail2, maximum, alloc2)
    if after.metrics["safe"]:
        return SimTrace("banker", "request", ["GRANT"],
                        {"granted": True, "reason": "the resulting state is safe",
                         "safe_sequence": after.answer}, after.rows, params)
    return SimTrace("banker", "request", ["WAIT"], {"granted": False,
                    "reason": "granting it would leave the system in an unsafe state"},
                    [], params)


def is_valid_safe_sequence(available: list[int], maximum: list[list[int]],
                           allocation: list[list[int]], sequence: list[str], *,
                           use_max_instead_of_need: bool = False,
                           reclaim_allocation: bool = True) -> bool:
    """Safe sequences are not unique: any order in which every process can finish counts."""
    need = _validate(available, maximum, allocation)
    if use_max_instead_of_need:
        need = maximum
    n, m = len(maximum), len(available)
    try:
        order = [int(str(tok).lstrip("Pp")) for tok in sequence]
    except ValueError:
        return False
    if sorted(order) != list(range(n)):
        return False
    work = list(available)
    for i in order:
        if any(need[i][j] > work[j] for j in range(m)):
            return False
        if reclaim_allocation:
            work = [work[j] + allocation[i][j] for j in range(m)]
    return True
