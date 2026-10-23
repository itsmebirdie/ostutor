"""Page replacement: FIFO, LRU, OPT (Belady). Answer = "F"/"H" (fault/hit) per reference.

OPT tie rule: among pages never used again (or tied), evict the one loaded earliest.
"""
from __future__ import annotations

from ostutor.sims.common import SimTrace, ValidationError, is_int, require

ALGORITHMS = ("fifo", "lru", "opt")


def simulate(refs: list[int], frames: int, algorithm: str, *, lru_update_on_hit: bool = True,
             fifo_update_on_hit: bool = False, opt_farthest: bool = True) -> SimTrace:
    require(len(refs) > 0 and all(is_int(r) and r >= 0 for r in refs),
            "the reference string must be a non-empty list of integers >= 0")
    require(is_int(frames) and frames > 0, "the number of frames must be an integer > 0")
    if algorithm not in ALGORITHMS:
        raise ValidationError(f"unknown algorithm {algorithm!r}; choose from {ALGORITHMS}")
    mem: list[int] = []  # ordered: index 0 = oldest (FIFO) / least recently used (LRU)
    answer, rows = [], []
    for i, page in enumerate(refs):
        evicted = None
        if page in mem:
            answer.append("H")
            if (algorithm == "lru" and lru_update_on_hit) or (
                    algorithm == "fifo" and fifo_update_on_hit):
                mem.remove(page)
                mem.append(page)
        else:
            answer.append("F")
            if len(mem) == frames:
                if algorithm == "opt":
                    def next_use(p, i=i):
                        for j in range(i + 1, len(refs)):
                            if refs[j] == p:
                                return j
                        return len(refs) + 1  # never used again
                    if opt_farthest:  # ties (never used again): earliest loaded first
                        victim = max(mem, key=lambda p: (next_use(p), -mem.index(p)))
                    else:  # misconception hook: evicts the page needed SOONEST
                        victim = min(mem, key=lambda p: (next_use(p), mem.index(p)))
                else:
                    victim = mem[0]
                evicted = victim
                mem.remove(victim)
            mem.append(page)
        rows.append({"ref": page, "result": answer[-1], "frames": sorted(mem),
                     "evicted": evicted})
    faults = answer.count("F")
    return SimTrace("paging", algorithm, answer,
                    {"faults": faults, "hits": len(refs) - faults,
                     "hit_ratio": (len(refs) - faults) / len(refs)}, rows,
                    {"frames": frames, "references": list(refs)})
