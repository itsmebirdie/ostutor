"""Disk scheduling: FCFS, SSTF, SCAN, C-SCAN, LOOK, C-LOOK.

Cylinders are 0..disk_size-1. `direction` is the initial head direction ("up" = toward
larger cylinder numbers). C-SCAN's return jump counts toward head movement by default
(`count_return=True`, the usual textbook convention). SSTF ties go to the lower cylinder.
The answer is the head PATH: every cylinder the head moves to, in order, including the disk
edge (SCAN/C-SCAN) and the return jump (C-SCAN). That is what a head-movement diagram shows
and it is what tells SCAN from LOOK. The plain service order is in metrics["order"].
"""
from __future__ import annotations

from ostutor.sims.common import SimTrace, ValidationError, is_int, require

ALGORITHMS = ("fcfs", "sstf", "scan", "cscan", "look", "clook")


def simulate(requests: list[int], head: int, algorithm: str, *, disk_size: int = 200,
             direction: str = "up", count_return: bool = True) -> SimTrace:
    require(is_int(disk_size) and disk_size > 1, "disk_size must be an integer > 1")
    require(is_int(head) and 0 <= head < disk_size, "head must be a cylinder on the disk")
    require(len(requests) > 0 and all(is_int(r) and 0 <= r < disk_size for r in requests),
            f"requests must be integers in 0..{disk_size - 1}")
    require(direction in ("up", "down"), "direction must be 'up' or 'down'")
    if algorithm not in ALGORITHMS:
        raise ValidationError(f"unknown algorithm {algorithm!r}; choose from {ALGORITHMS}")
    pos, total = head, 0
    order: list[int] = []
    path = [head]

    def move(to: int, counted: bool = True):
        nonlocal pos, total
        if counted:
            total += abs(to - pos)
        pos = to
        path.append(to)

    pend = list(requests)
    if algorithm == "fcfs":
        for r in pend:
            move(r)
            order.append(r)
    elif algorithm == "sstf":
        while pend:
            r = min(pend, key=lambda x: (abs(x - pos), x))
            pend.remove(r)
            move(r)
            order.append(r)
    else:
        lo = sorted(r for r in pend if r < head)
        hi = sorted(r for r in pend if r >= head)
        up = direction == "up"
        edge = disk_size - 1
        if algorithm in ("scan", "look"):
            first, second = (hi, lo[::-1]) if up else (lo[::-1], hi)
            for r in first:
                move(r)
                order.append(r)
            if second:
                if algorithm == "scan":
                    move(edge if up else 0)
                for r in second:
                    move(r)
                    order.append(r)
        else:  # c-scan / c-look: sweep one way, jump to the far end, sweep the same way
            first = hi if up else lo[::-1]
            rest = lo if up else hi[::-1]
            for r in first:
                move(r)
                order.append(r)
            if rest:
                if algorithm == "cscan":
                    move(edge if up else 0)
                    move(0 if up else edge, counted=count_return)
                else:
                    move(rest[0], counted=count_return)
                    order.append(rest[0])
                    rest = rest[1:]
                for r in rest:
                    move(r)
                    order.append(r)
    return SimTrace("disk", algorithm, [str(c) for c in path[1:]],
                    {"total_head_movement": total, "path": path,
                     "order": [str(r) for r in order]},
                    [{"to": c} for c in path[1:]],
                    {"head": head, "direction": direction, "disk_size": disk_size,
                     "count_return": count_return})
