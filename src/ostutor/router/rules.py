"""Rule-based router (Build Plan 8.7): decides which layer should handle a message.

Layers: "analyze_code" (interleaving explorer), "diagnose" (a student's wrong answer to a
scheduling/memory question), "simulate" (compute a scheduling/memory problem), "rag"
(conceptual question answered from course material).

Rules first; a classifier is only justified if these prove insufficient. Every route
carries the rules that fired, so a wrong route is easy to debug.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

CODE_LINE = re.compile(r"^\s*(thread\s+\w+\s*:|mutex\s|semaphore\s|cond\s|shared\s|forever\s*:|"
                       r"repeat\s+\d+\s*:|(lock|unlock|down|up|signal|broadcast|read|write)\s+\w+"
                       r"\s*$|wait\s+\w+\s*,\s*\w+\s*$)", re.M)
FENCE = re.compile(r"```")
SIM_TOPICS = {
    "cpu": r"\b(fcfs|sjf|srtf|round[- ]robin|time quantum|quantum|gantt|turnaround|waiting time|"
           r"burst|mlfq|multilevel feedback|priority scheduling|cpu scheduling)\b",
    "banker": r"\b(banker'?s?|safe sequence|safe state|available vector|allocation matrix|"
              r"need matrix|max(imum)? matrix|resource[- ]request)\b",
    "paging": r"\b(page replacement|reference string|page frames?|page faults?|belady|\blru\b|"
              r"\bfifo\b|optimal replacement|\bopt\b)\b",
    "disk": r"\b(disk scheduling|cylinders?|head movement|\bsstf\b|\bscan\b|c-scan|\bcscan\b|"
            r"\blook\b|c-look|seek)\b",
}
ANSWER_CUES = re.compile(r"\b(my answer|i got|i answered|i wrote|student answer|is this "
                         r"right|why is (this|my answer) wrong|what did i do wrong|"
                         r"i think the (answer|result) is|my gantt|my result)\b", re.I)
CONCURRENCY_WORDS = re.compile(r"\b(deadlock|race condition|mutex|semaphore|critical section|"
                               r"interleaving|starvation|lost signal|condition variable)\b", re.I)
ANALYZE_REQUEST = re.compile(
    r"\b(analy[sz]e|check|is (this|my|the) (code|program)|find the (bug|race|deadlock)|"
    r"(will|can|could|does|would)\s+(this|it|my|our|the)(\s+\w+){0,2}\s+"
    r"(deadlock|race|hang|starve))\b", re.I)
# process-id sequences such as "P1 P1 P2"; bare numbers are problem DATA, not an answer
SEQUENCE = re.compile(r"(\bP\d+\b[\s,]+){3,}")


@dataclass
class Route:
    layer: str
    reason: str
    topic: str | None = None
    rules: list[str] = field(default_factory=list)


def route(message: str) -> Route:
    text = message or ""
    low = text.lower()
    rules: list[str] = []

    if CODE_LINE.search(text) and len(CODE_LINE.findall(text)) >= 2:
        rules.append("code_lines")
        return Route("analyze_code", "the message contains program code", None, rules)
    if FENCE.search(text) and CONCURRENCY_WORDS.search(text):
        rules.append("fenced_code+concurrency_words")
        return Route("analyze_code", "a code block about concurrency", None, rules)

    topic = next((t for t, rx in SIM_TOPICS.items() if re.search(rx, low)), None)
    if topic:
        rules.append(f"topic:{topic}")
        if ANSWER_CUES.search(text) or SEQUENCE.search(text):
            rules.append("answer_cue")
            return Route("diagnose", "a student answer to a scheduling/memory problem", topic,
                         rules)
        concept = re.match(r"\s*(what|why|how|explain|define|difference|when)\b", low)
        has_numbers = len(re.findall(r"\d+", text)) >= 4
        if concept and not has_numbers:
            rules.append("concept_question")
            return Route("rag", "a conceptual question about " + topic, topic, rules)
        return Route("simulate", "a scheduling/memory problem to compute", topic, rules)

    if CONCURRENCY_WORDS.search(text) and ANALYZE_REQUEST.search(text):
        rules.append("analyze_request_without_code")
        return Route("rag", "asks to analyse code but none was pasted; paste the program, "
                            "or ask a conceptual question", None, rules)
    rules.append("default")
    return Route("rag", "a conceptual question", None, rules)
