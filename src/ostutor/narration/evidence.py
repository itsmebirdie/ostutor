"""Compact evidence encoder (Build Plan 8.5): the ONLY thing the LLM sees.

One line per step plus label, bounds and critical point; target <= ~1,000 tokens.
"""
from __future__ import annotations

from ostutor.common.types import Verdict


def estimate_tokens(text: str) -> int:
    """Rough token estimate (about 4 characters per token); good enough for budgets."""
    return max(1, len(text) // 4)


def encode_evidence(program: dict, verdict: Verdict) -> str:
    v = verdict
    out = [
        f"label: {v.label}",
        f"completeness: {v.completeness}",
        f"bounds: {v.bounds}",
        f"states_explored: {v.states_explored}",
        f"threads: {', '.join(sorted(program['threads']))}",
        f"mutexes: {', '.join(program['mutexes']) or '-'}",
        f"shared: {', '.join(program['shared']) or '-'}",
    ]
    if v.witness:
        out.append("witness (shortest schedule that exhibits the failure):")
        for s in v.witness.steps:
            delta = ", ".join(f"{k}={val}" for k, val in s.delta.items()) or "-"
            blocked = ", ".join(s.blocked) or "-"
            out.append(f"  step {s.i}: {s.thread} {s.op} {s.object} | {delta} "
                       f"| blocked after: {blocked}")
    else:
        out.append("witness: none (every explored schedule completes safely)")
    if v.critical_point:
        cp = v.critical_point
        out.append(f"critical_point: after step {cp['decision_after_step']}; "
                   f"bad choice {cp['bad_choice']}; safe alternative "
                   f"{cp.get('safe_alternative') or 'none'}")
    if v.detail:
        out.append(f"detail: {v.detail}")
    return "\n".join(out)
