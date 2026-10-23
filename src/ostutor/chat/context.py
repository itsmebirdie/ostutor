"""Detect when a question matches something the deterministic explorer can
analyse, run it, and return its output to ground the LLM."""
from __future__ import annotations

import re
import subprocess
import sys

# (regex on the student's message, demo CLI args, human topic)
ROUTES = [
    (r"\bdead\s?lock", ["deadlock_two_locks"], "deadlock with two locks"),
    (r"\brace\b|\bdata race|\bunsynchroni[sz]ed", ["race_no_lock"], "race condition without a lock"),
    (r"\bmutex|\bcritical section|\bmutual exclusion|\block(s|ing)?\b", ["safe_with_lock"],
     "correct mutual exclusion with a lock"),
    (r"round[\s-]?robin|\bquantum\b|\btime slice", ["--rr"], "Round Robin scheduling"),
    (r"state explosion|interleaving", ["--scaling"], "interleavings vs reachable states"),
]


def match_route(message: str):
    text = message.lower()
    for pattern, args, topic in ROUTES:
        if re.search(pattern, text):
            return args, topic
    return None


def run_demo(args: list[str], timeout: float = 30.0) -> str | None:
    """Run the existing demo CLI. Returns its stdout, or None on any failure
    (the chatbot then simply answers without grounding)."""
    try:
        r = subprocess.run([sys.executable, "-m", "ostutor.slice.demo", *args],
                           capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return None
    out = r.stdout.strip()
    return out[:4000] if r.returncode == 0 and out else None


def build_context(message: str, runner=run_demo):
    """Return (topic, analysis_text) or None."""
    route = match_route(message)
    if not route:
        return None
    args, topic = route
    body = runner(args)
    return (topic, body) if body else None