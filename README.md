# OS Tutor: an LLM-orchestrated interleaving explorer for Operating Systems tutoring

BCSE303L project by Maanas Nair (24BCT0157) and Pawan Kumar (24BCT0153).

**The engine computes. The LLM only explains. A checker verifies the explanation against
what was computed.** LLMs invent concurrent behaviour; here every verdict, schedule and
number comes from deterministic code, and any explanation (template or LLM) is checked
mechanically before a student sees it. With no LLM at all the system still gives full,
checked, template explanations.

## What is built

| Layer | What it does | Where |
|---|---|---|
| Front end | Restricted DSL: `mutex`, `semaphore`, `cond`, `shared`, `thread`, `lock/unlock`, `down/up`, `wait/signal/broadcast`, `read/write`, `repeat N`, `forever`; friendly line-numbered errors | `dsl/` |
| Interleaving explorer | **Tier A** exhaustive state graph; **Tier B** symmetry reduction; **Tier C** PCT sampling (never reports "safe"). Detects deadlock, race, lost signal, starvation (under a stated fairness assumption). Shortest witness, critical decision point, independent replay. All verdicts carry a completeness label | `explorer/` (the original review slice is kept in `slice/` as a differential-test oracle) |
| Exact simulators | CPU scheduling (FCFS, SJF, SRTF, Priority, RR, MLFQ), Banker's (safety + request), page replacement (FIFO, LRU, OPT), disk scheduling (FCFS, SSTF, SCAN, C-SCAN, LOOK, C-LOOK); validation before computation; tie-breaking as explicit parameters | `sims/` |
| Mutants and diagnosis | 16 named misconceptions; exact match (all matches reported), edit-distance fall-back, optional unverified LLM hypothesis | `mutants/`, `diagnosis/` |
| Narration | Template-first; LLM narration (Groq or any OpenAI-compatible API) validated by a mechanical checker, re-prompted twice, then template fall-back; SQLite cache; recorded fixtures | `narration/`, `cache/`, `slice/check.py` |
| Hybrid RAG | Ingest (md/txt/pdf/pptx), chunking with overlap, dedupe, BM25 + dense with reciprocal rank fusion, citations, "not found" | `rag/` |
| Router, API, UI | Rule-based router; FastAPI (`/analyze_code`, `/simulate`, `/diagnose`, `/ask`); Streamlit demo | `router/`, `api/`, `ui/` |
| Evaluation | Scripts that produce every number for the report | `eval/` |

**Not built / known limits** (see `docs/REPORT_UPDATE.md` for the full list): sleep sets and
DPOR (Tier B is symmetry reduction only), shared *values* (so no reader-writer with counters,
and races are lockset-style), real bge embeddings were not evaluated (an offline lexical
embedder ships for tests), no real student data, no LLM latency/cost measurements.

## Quick start

```bash
pip install -e ".[dev]"            # everything, including UI, API and RAG extras
python -m ostutor.cli analyze data/programs/deadlock_two_locks.ost
python -m ostutor.cli analyze data/programs/starvation_unfair_mutex.ost --fairness strong
python -m ostutor.cli analyze data/programs/dining_5_deadlock.ost --tier B
streamlit run streamlit_app.py     # the demo (see docs/DEMO.md)
uvicorn ostutor.api.app:app        # the API
python eval/run_all.py             # regenerate every evaluation result
python eval/run_rag_real.py --questions eval/my_questions.jsonl --embedder bge-small   # docs/RAG_EVAL.md
make check                         # exactly what CI runs: lint, types, tests (90% coverage floor)
```

LLM access: `cp .env.example .env`, add your key, then `python -m ostutor.cli ping`.
Model names change; `python -m ostutor.cli models` lists what your key can use
(`docs/LLM_SETUP.md`). Pre-record demo narrations with `python -m ostutor.cli warm`
(**re-run it after upgrading: recordings are keyed by the evidence text**).

## The DSL

```
mutex m
semaphore items = 0            # counting semaphore with an initial value
cond ready
shared buf
thread Producer:
    forever:                   # endless loop (top level of a thread only); `repeat 3:` unrolls
        lock m
        write buf
        signal ready
        unlock m
        up items
thread Consumer:
    down items
    lock m
    wait ready, m              # releases m, waits on `ready`, re-acquires m (FIFO wake-up)
    unlock m
```

Programs are plain data (`{"mutexes", "shared", "threads", ["semaphores"], ["conds"]}`), so
the parser, tests and API all share one structure. A bug family is built into each verdict:
`deadlock`, `lost_signal`, `race`, `starvation`, `safe`, or `none_found` (Tier C sampled, never
"safe"). `verdict.detail["findings"]` lists every kind found, ordered by severity.

## Tests, CI, releases

`pytest` (offline; `tests/conftest.py` blocks the network and removes API keys), `ruff`, `mypy`.
Highlights: differential tests of the general engine against the original explorer on 200 random
programs; Tier B against Tier A on 150 random symmetric programs and rotated dining
philosophers; golden tests from textbook examples for every simulator; mutation checks of the
fairness and replay logic; real PDF/PPTX ingestion tests; Streamlit `AppTest` and FastAPI
`TestClient` tests.

GitHub Actions: `ci.yml` (lint+types, tests on Python 3.11 to 3.13 plus Windows and macOS,
acceptance gate on the *installed* package including a live `uvicorn`, the evaluation scripts,
build), `release.yml` (tag `vX.Y.Z` to a GitHub Release), `llm-live-check.yml` (manual, uses the
`GROQ_API_KEY` secret).

## Layout

```
src/ostutor/  common/ dsl/ explorer/ slice/ sims/ mutants/ diagnosis/ narration/ cache/
              rag/ router/ api/ ui/ cli.py
data/programs/        12 example programs (.ost)       data/corpus_sample/  sample course notes
eval/                 evaluation scripts + results/    docs/                setup, demo, report update
```
