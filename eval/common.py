"""Shared helpers for the evaluation scripts. Run from the repo root:
    python eval/run_all.py        (writes eval/results/*.md and eval/results/summary.json)
"""
from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = HERE / "results"
for p in (str(ROOT / "src"), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)


def table(headers: list[str], rows: list[list]) -> str:
    def cell(x):
        return f"{x:.3g}" if isinstance(x, float) else str(x)
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(cell(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def save(name: str, markdown: str, data: dict) -> dict:
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{name}.md").write_text(markdown + "\n", encoding="utf-8")
    (RESULTS / f"{name}.json").write_text(json.dumps(data, indent=1, default=str), "utf-8")
    return data


def environment() -> str:
    return f"Python {platform.python_version()} on {platform.system()} {platform.machine()}"
