"""Mechanical narration checker (Build Plan 2.6 / 8.5).

Rules:
  1. every step index (claims[].step and any "step N" in text) exists
  2. every name is in the trace vocabulary
       - backticked tokens must be a thread/mutex/variable or an outcome label
       - unquoted tokens of the form letters+digits (T3, x2) must be in the vocabulary
  3. every standalone number is checked AGAINST ITS CONTEXT (a number before
     "steps" must be the step count, before "states" the state count, ...);
     a number with no recognised context must be one of the recorded bounds
  4. the stated label ("Verdict: `x`") equals the computed label

Names: backticked tokens and letters+digits tokens are checked everywhere; in
addition a bare name right after "thread/mutex/lock/variable" is checked, which
catches "mutex c" in free LLM text. It validates FACTS only, not causal reasoning.
"""
from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field

from ostutor.common.types import LABELS, Verdict

_NUM = re.compile(r"(?<![A-Za-z0-9_`.])\d+(?![A-Za-z0-9_`])")
_TICKED = re.compile(r"`([^`]+)`")
_NAMELIKE = re.compile(r"\b[A-Za-z]+\d+\b")
_STEPREF = re.compile(r"[Ss]teps?\s+(\d+)")
_VERDICT = re.compile(r"Verdict:\s*`([a-z_]+)`")
_BARE_NAME = re.compile(
    r"\b(?:threads?|mutex(?:es)?|locks?|variables?|vars?)\s+([A-Za-z][A-Za-z0-9_]*)\b")
_STEP_CTX = re.compile(r"[Ss]teps?[\d\s,]*(?:and|to|or|[-\u2013])?[\d\s,]*$")
_STEP_WORDS = {"step", "steps"}
_STATE_WORDS = {"state", "states"}
_THREAD_WORDS = {"thread", "threads"}
_MUTEX_WORDS = {"mutex", "mutexes", "lock", "locks"}
_VAR_WORDS = {"variable", "variables"}
# words that can legitimately follow "thread"/"lock"/... in English prose
_PROSE = {"a", "an", "the", "is", "are", "was", "that", "which", "has", "have", "it",
          "to", "on", "in", "of", "and", "or", "for", "with", "can", "cannot", "must",
          "will", "then", "while", "when", "waits", "blocks", "holds", "held",
          "order", "ordering", "acquisition", "acquires", "released", "release"}


class _Info:
    """The vocabulary of a program (works for every operation the explorer supports)."""

    def __init__(self, program: dict):
        self.threads = sorted(program["threads"])
        self.mutexes = list(program["mutexes"])
        self.shared = list(program["shared"])
        self.sems = list(program.get("semaphores", {}))
        self.conds = list(program.get("conds", []))

    @property
    def names(self) -> set:
        return set(self.threads) | set(self.mutexes) | set(self.shared) | set(self.sems) \
            | set(self.conds)


@dataclass
class CheckResult:
    ok: bool
    violations: list[dict] = field(default_factory=list)


def _texts(narration: dict) -> list[str]:
    return [c.get("text", "") for c in narration.get("claims", [])] + [
        narration.get("summary", "")]


def _numbers(text: str):
    """Yield (value, previous word, next word) for each standalone number."""
    for m in _NUM.finditer(text):
        prev = re.search(r"([A-Za-z]+)\s*$", text[:m.start()])
        nxt = re.match(r"\s*([A-Za-z]+)", text[m.end():])
        yield (int(m.group()), prev.group(1).lower() if prev else "",
               nxt.group(1).lower() if nxt else "", text[:m.start()])


def _number_problem(num: int, prev: str, nxt: str, before: str, m: _Info,
                    verdict: Verdict, n_steps: int):
    """Return a detail string if the number is not justified by its context."""
    if prev in _STEP_WORDS or _STEP_CTX.search(before):
        return None if 1 <= num <= n_steps else f"step number {num} is not in the trace"
    if nxt in _STEP_WORDS:
        return None if num == n_steps else (
            f"number {num} is not in the trace: the witness has {n_steps} steps")
    if nxt in _STATE_WORDS:
        ok = {verdict.states_explored, verdict.bounds.get("state_budget")}
        return None if num in ok else (
            f"number {num} is not in the trace: {verdict.states_explored} states explored")
    for words, count, what in ((_THREAD_WORDS, len(m.threads), "threads"),
                               (_MUTEX_WORDS, len(m.mutexes), "mutexes"),
                               (_VAR_WORDS, len(m.shared), "variables")):
        if nxt in words:
            return None if num == count else (
                f"number {num} is not in the trace: the program has {count} {what}")
    bounds = {v for v in verdict.bounds.values() if isinstance(v, int)}
    return None if num in bounds else f"number {num} is not in the trace"


def check(narration: dict, verdict: Verdict, program: dict) -> CheckResult:
    v: list[dict] = []
    m = _Info(program)
    vocab = m.names
    n_steps = len(verdict.witness.steps) if verdict.witness else 0

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
        for tok in _BARE_NAME.findall(text):
            if tok in vocab or tok.lower() in _PROSE or len(tok) > 2 and tok.islower():
                continue  # known name, English word, or a long lowercase word
            if tok == "a":
                continue  # ambiguous with the article; the backtick rule covers it
            v.append({"rule": "vocabulary", "detail": f"unknown name {tok}"})

    # Rule 3: numbers
    for text in _texts(narration):
        for num, prev, nxt, before in _numbers(text):
            problem = _number_problem(num, prev, nxt, before, m, verdict, n_steps)
            if problem:
                v.append({"rule": "number", "detail": problem})

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
