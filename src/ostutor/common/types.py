"""Shared data contracts (Build Plan section 7): Trace, Step, Verdict.

Narration and Diagnosis stay plain dicts / small dataclasses elsewhere.
Two small extensions to the plan's Verdict, both optional for consumers:
  * ``detail``: facts about the failure (race variable, who blocks whom)
  * critical_point carries structured fields next to the "T2:lock b" string
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

# "none_found" is an extension for Tier C: a sampled search that found nothing. It is
# deliberately NOT "safe": sampling can never prove the absence of a failure.
LABELS = ("safe", "race", "deadlock", "starvation", "lost_signal", "none_found")


@dataclass
class Step:
    i: int  # 1-based step number within the trace
    thread: str
    op: str
    object: str  # mutex name for lock/unlock, shared variable for read/write
    delta: dict[str, Any] = field(default_factory=dict)
    blocked: list[str] = field(default_factory=list)  # threads blocked AFTER this step


@dataclass
class Trace:
    kind: str  # "concurrency" | "cpu" | ...
    steps: list[Step]
    vocab: dict[str, list[str]]  # {"threads": [...], "objects": [...], "vars": [...]}
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Verdict:
    label: str
    completeness: str  # "complete_for_bound" | "reduced_but_complete" | "sampled"
    bounds: dict[str, int]
    states_explored: int
    fairness_assumption: str | None
    witness: Trace | None
    critical_point: dict[str, Any] | None
    program_hash: str
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def program_hash(program: dict) -> str:
    """Hash of the program as given (the slice has no canonicalisation yet)."""
    blob = json.dumps(program, sort_keys=True, default=list)
    return "sha256:" + hashlib.sha256(blob.encode()).hexdigest()


def verdict_from_dict(d: dict) -> Verdict:
    """Inverse of Verdict.to_dict() (used by the SQLite cache)."""
    w = d.get("witness")
    witness = None
    if w:
        witness = Trace(w["kind"], [Step(**s) for s in w["steps"]], w["vocab"],
                        w.get("params", {}))
    return Verdict(
        label=d["label"], completeness=d["completeness"], bounds=d["bounds"],
        states_explored=d["states_explored"],
        fairness_assumption=d.get("fairness_assumption"), witness=witness,
        critical_point=d.get("critical_point"), program_hash=d["program_hash"],
        detail=d.get("detail", {}))
