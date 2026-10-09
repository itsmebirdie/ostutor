"""Mechanical narration checker (Build Plan 2.6 / 8.5).

Rules:
  1. every step index (claims[].step and any "step N" in text) exists
  2. every name is in the trace vocabulary
       - backticked tokens must be a thread/mutex/variable or an outcome label
       - unquoted tokens of the form letters+digits (T3, x2) must be in the vocabulary
  3. every standalone number appears in the trace or is a whitelisted count
  4. the stated label ("Verdict: `x`") equals the computed label

It validates FACTS only, not causal reasoning. Single-letter invented names that
are not backticked cannot be detected; the narration convention backticks all names.
"""
from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field

from ostutor.common.types import LABELS, Verdict
from ostutor.slice.explorer import Model

_NUM = re.compile(r"(?<![A-Za-z0-9_`.])\d+(?![A-Za-z0-9_`])")
_TICKED = re.compile(r"`([^`]+)`")
_NAMELIKE = re.compile(r"\b[A-Za-z]+\d+\b")
_STEPREF = re.compile(r"[Ss]teps?\s+(\d+)")
_VERDICT = re.compile(r"Verdict:\s*`([a-z_]+)`")


@dataclass
class CheckResult:
    ok: bool
    violations: list[dict] = field(default_factory=list)


def _texts(narration: dict) -> list[str]:
    return [c.get("text", "") for c in narration.get("claims", [])] + [
        narration.get("summary", "")]


def _allowed_numbers(program: dict, verdict: Verdict) -> set[int]:
    m = Model(program)
    allowed = {verdict.states_explored, len(m.threads), len(m.mutexes), len(m.shared)}
    if verdict.witness:
        allowed |= {s.i for s in verdict.witness.steps}
        allowed.add(len(verdict.witness.steps))  # whitelisted: step count
    if verdict.critical_point:
        allowed |= {v for v in verdict.critical_point.values() if isinstance(v, int)}
    allowed |= {v for v in verdict.bounds.values() if isinstance(v, int)}
    return allowed


def check(narration: dict, verdict: Verdict, program: dict) -> CheckResult:
    v: list[dict] = []
    m = Model(program)
    vocab = set(m.threads) | set(m.mutexes) | set(m.shared)
    n_steps = len(verdict.witness.steps) if verdict.witness else 0
    allowed_nums = _allowed_numbers(program, verdict)

    # Rule 1: step indices
    for c in narration.get("claims", []):
        s = c.get("step")
        if not isinstance(s, int) or not 1 <= s <= n_steps:
            v.append({"rule": "step_index", "detail": f"claim cites step {s!r}; "
                      f"trace has steps 1..{n_steps}"})
    for text in _texts(narration):
        for ref in _STEPREF.findall(text):
            if not 1 <= int(ref) <= n_steps:
                v.append({"rule": "step_index", "detail": f"text cites step {ref}"})

    # Rule 2: names
    for text in _texts(narration):
        for tok in _TICKED.findall(text):
            if tok not in vocab and tok not in LABELS:
                v.append({"rule": "vocabulary", "detail": f"unknown name `{tok}`"})
        for tok in _NAMELIKE.findall(text):
            if tok not in vocab:
                v.append({"rule": "vocabulary", "detail": f"unknown name {tok}"})

    # Rule 3: numbers
    for text in _texts(narration):
        for num in _NUM.findall(text):
            if int(num) not in allowed_nums:
                v.append({"rule": "number", "detail": f"number {num} is not in the trace"})

    # Rule 4: label
    stated = _VERDICT.findall(narration.get("summary", ""))
    if not stated:
        v.append({"rule": "label", "detail": "summary does not state a verdict"})
    elif any(s != verdict.label for s in stated):
        v.append({"rule": "label",
                  "detail": f"stated {stated[0]!r}, computed {verdict.label!r}"})

    # de-duplicate, keep order
    seen, uniq = set(), []
    for x in v:
        key = (x["rule"], x["detail"])
        if key not in seen:
            seen.add(key)
            uniq.append(x)
    return CheckResult(not uniq, uniq)


def tamper_examples(narration: dict, verdict: Verdict) -> dict[str, dict]:
    """The four deliberately bad narrations from the acceptance gate."""
    out = {}

    bad = copy.deepcopy(narration)
    bad["claims"].append({"step": 99, "text": "`T1` locks `a`."})
    out["invented step index"] = bad

    bad = copy.deepcopy(narration)
    victim = next(t for t in verdict.witness.vocab["threads"]) if verdict.witness \
        else "T1"
    bad["summary"] = bad["summary"].replace(f"`{victim}`", "`T9`", 1)
    if verdict.witness and bad["claims"]:
        bad["claims"][0]["text"] = bad["claims"][0]["text"].replace(f"`{victim}`", "`T9`", 1)
    if bad == narration:  # e.g. a safe verdict names no thread
        bad["summary"] += " `T9` never ran."
    out["unknown thread name"] = bad

    bad = copy.deepcopy(narration)
    n = verdict.states_explored
    bad["summary"] = bad["summary"].replace(f"{n} states", f"{n + 1000} states")
    out["wrong number"] = bad

    bad = copy.deepcopy(narration)
    other = "safe" if verdict.label != "safe" else "deadlock"
    bad["summary"] = bad["summary"].replace(f"`{verdict.label}`", f"`{other}`", 1)
    out["wrong label"] = bad
    return out
