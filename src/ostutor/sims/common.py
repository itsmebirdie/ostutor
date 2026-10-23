"""Shared types for the exact simulators (Build Plan 8.3): pure functions, typed traces,
explicit tie-breaking parameters, validation before any computation."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


class ValidationError(ValueError):
    """The problem instance is malformed (negative value, bad shape, unsafe start...)."""


@dataclass
class SimTrace:
    kind: str                      # "cpu" | "banker" | "paging" | "disk"
    algorithm: str
    answer: list[str]              # the canonical answer a student would write down
    metrics: dict = field(default_factory=dict)
    rows: list[dict] = field(default_factory=list)   # per-step detail for tables
    params: dict = field(default_factory=dict)       # tie-breaking rules, quantum, ...

    def to_dict(self) -> dict:
        return asdict(self)


def require(cond: bool, message: str) -> None:
    if not cond:
        raise ValidationError(message)


def is_int(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)
