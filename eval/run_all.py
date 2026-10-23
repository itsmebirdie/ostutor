"""Run every evaluation and write eval/results/*.md, *.json and SUMMARY.md.

    python eval/run_all.py

Every figure for the report must come from these scripts (Build Plan definition of done 5)."""
from __future__ import annotations

import time

import run_classifier
import run_diagnosis
import run_efficiency
import run_faithfulness
import run_rag
import run_router
from common import RESULTS, environment, save

STEPS = [("classifier", run_classifier), ("efficiency", run_efficiency),
         ("faithfulness", run_faithfulness), ("diagnosis (synthetic)", run_diagnosis),
         ("router", run_router), ("rag (sample corpus)", run_rag)]


def main() -> dict:
    out, lines = {}, []
    for name, mod in STEPS:
        t0 = time.perf_counter()
        data = mod.main()
        out[name] = data["headline"]
        lines.append(f"- **{name}**: {data['headline']}  ({time.perf_counter() - t0:.1f}s, "
                     f"see `{mod.__name__.removeprefix('run_')}.md`)")
    md = f"""# Evaluation summary

*{environment()}. Produced by `python eval/run_all.py`.*

{chr(10).join(lines)}

## Read before quoting any of this

- The concurrency labels were derived by reasoning, not by two independent annotators:
  this is **agreement with construction labels**, not accuracy. Verify them by hand first.
- Diagnosis results are **synthetic and circular**; no diagnostic-accuracy claim is possible
  without real, consented student answers.
- RAG numbers use a six-section sample corpus and a lexical embedder.
- No LLM pass/regeneration rate appears until recorded answers exist (`ostutor warm`).
- Not measured anywhere: learning outcomes, latency with a real LLM, token costs at scale.
"""
    (RESULTS / "SUMMARY.md").write_text(md, encoding="utf-8")
    return save("summary", md, out)


if __name__ == "__main__":
    print(main())
