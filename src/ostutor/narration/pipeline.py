"""Checked narration pipeline (Build Plan 8.5, design rules 1 and 5).

    cache hit?  -> return it (zero tokens)
    LLM present -> ask, parse JSON, run the mechanical checker; on failure re-prompt
                   at most twice with the violated rules quoted
    otherwise / still failing -> template narration (always passes the checker)

The LLM only ever receives computed evidence. It cannot change a number, schedule
or label: the checker rejects any narration that disagrees with the verdict.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Optional

from ostutor.cache.store import Cache
from ostutor.common.types import Verdict
from ostutor.narration.evidence import encode_evidence
from ostutor.narration.llm import LLMClient, LLMUnavailable
from ostutor.slice.check import check
from ostutor.slice.narrate import narrate as template_narrate

PROMPT_VERSION = "v1"
MAX_REPROMPTS = 2

SYSTEM_PROMPT = """You are a patient Operating Systems tutor. A verified analysis engine has \
already computed the result of a concurrency program. Your job is ONLY to explain it to a \
student in plain language.

Hard rules:
1. Use only facts from the EVIDENCE. Never invent steps, threads, mutexes, variables or numbers.
2. Every thread, mutex and shared-variable name must be written in backticks, e.g. `T1`, `a`.
3. Every number you write must appear in the evidence (step numbers, the step count, the number of states).
4. The summary MUST start with: Verdict: `<label>`. using exactly the label from the evidence.
5. Write one claim per witness step, citing that step number. Explain WHY the failure happens \
(for example a circular wait, or two threads touching a variable with no common lock) \
in the summary, and the critical decision point if one is given.
6. Reply with a single JSON object and nothing else."""

_SHAPE = '{"claims": [{"step": <int>, "text": "<one sentence>"}], "summary": "<2-4 sentences>"}'


def build_user_prompt(evidence: str, violations: Optional[list[dict]] = None,
                      previous: Optional[str] = None) -> str:
    msg = f"EVIDENCE\n{evidence}\n\nReturn JSON of exactly this shape: {_SHAPE}"
    if violations:
        rules = "\n".join(f"- {v['rule']}: {v['detail']}" for v in violations)
        msg += (f"\n\nYour previous answer was rejected by the fact checker:\n{previous}\n"
                f"Violated rules:\n{rules}\nFix every problem and return JSON only.")
    return msg


def parse_narration(raw: str) -> tuple[Optional[dict], list[dict]]:
    """Extract and structurally validate the JSON narration."""
    text = raw.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None, [{"rule": "format", "detail": "reply is not a JSON object"}]
    try:
        obj = json.loads(text[start:end + 1])
    except json.JSONDecodeError as e:
        return None, [{"rule": "format", "detail": f"invalid JSON: {e.msg}"}]
    claims = obj.get("claims")
    if (not isinstance(claims, list) or not isinstance(obj.get("summary"), str)
            or not all(isinstance(c, dict) and isinstance(c.get("step"), int)
                       and isinstance(c.get("text"), str) for c in claims)):
        return None, [{"rule": "format",
                       "detail": f"JSON must match {_SHAPE}"}]
    return {"claims": [{"step": c["step"], "text": c["text"]} for c in claims],
            "summary": obj["summary"]}, []


@dataclass
class Explanation:
    narration: dict
    source: str  # "llm" | "cache" | "template" | "template_fallback"
    checked: bool
    attempts: int = 0  # live LLM calls attempted for this explanation
    violations: list[dict] = field(default_factory=list)  # last rejection, if any
    note: str = ""

    @property
    def badge(self) -> str:
        return {"llm": "checked (LLM)", "cache": "checked (cached LLM)",
                "template": "template", "template_fallback": "template fall-back"}[self.source]


def explain(program: dict, verdict: Verdict, llm: Optional[LLMClient] = None,
            cache: Optional[Cache] = None, question: str = "why") -> Explanation:
    template = template_narrate(program, verdict)
    if llm is None:
        return Explanation(template, "template", True)

    evidence = encode_evidence(program, verdict)
    key = Cache.narration_key(evidence, question, PROMPT_VERSION, llm.config.model) \
        if cache else ""
    if cache:
        hit = cache.get_narration(key)
        if hit is not None:
            return Explanation(hit, "cache", True)

    violations: list[dict] = []
    previous = None
    attempts = 0
    for attempt in range(MAX_REPROMPTS + 1):
        user = build_user_prompt(evidence, violations or None, previous)
        try:
            raw = llm.complete(SYSTEM_PROMPT, user)
        except LLMUnavailable as e:
            return Explanation(template, "template_fallback", True, attempts,
                               violations, f"LLM unavailable: {e}")
        attempts += 1
        narration, violations = parse_narration(raw)
        if narration is not None:
            res = check(narration, verdict, program)
            violations = res.violations
            if res.ok:
                if cache:
                    cache.put_narration(key, narration)
                return Explanation(narration, "llm", True, attempts)
        previous = raw.strip()[:1500]
    return Explanation(template, "template_fallback", True, attempts, violations,
                       "LLM output failed the fact checker; showing the template")


def analyze_cached(program: dict, cache: Optional[Cache], loop_unroll: int = 0,
                   budget: int = 300_000) -> Verdict:
    from ostutor.common.types import program_hash
    from ostutor.slice.explorer import analyze
    if cache is None:
        return analyze(program, budget, loop_unroll)
    bounds = {"loop_unroll": loop_unroll, "state_budget": budget}
    key = Cache.verdict_key(program_hash(program), bounds)
    hit = cache.get_verdict(key)
    if hit is not None:
        return hit
    v = analyze(program, budget, loop_unroll)
    cache.put_verdict(key, v)
    return v
