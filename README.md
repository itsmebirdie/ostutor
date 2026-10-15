# OS Tutor: review slice

A deterministic explorer computes the verdict, the shortest witness and the critical
decision point; narration (template or LLM) is then **mechanically checked** against what
was computed. The LLM is optional: with no key the tutor falls back to templates.
Still no runtime dependencies (the LLM client uses the standard library).

## Setup (Python 3.11 or newer)

```bash
cd os-tutor
python -m venv .venv

# activate the environment
source .venv/bin/activate          # Linux / macOS
.venv\Scripts\Activate.ps1         # Windows PowerShell
.venv\Scripts\activate.bat         # Windows cmd

pip install -e ".[dev]"            # installs the package and pytest
pytest -q                          # expect: 38 passed, 1 skipped
```

If you prefer not to install: `pip install pytest`, then `PYTHONPATH=src python -m ostutor.slice.demo ...`
(on Windows PowerShell: `$env:PYTHONPATH="src"`). Pytest itself finds `src/` automatically.

## Run the demo

```bash
python -m ostutor.slice.demo deadlock_two_locks            # gate item 1
python -m ostutor.slice.demo deadlock_two_locks --tamper   # gate item 4, checker rejects bad narrations
python -m ostutor.slice.demo --all                         # gate item 2, all three labels
python -m ostutor.slice.demo --scaling                     # 184,756 interleavings vs 121 states
python -m ostutor.slice.demo --rr                          # optional S7, Round Robin mutant
```

## Layout

```
src/ostutor/common/types.py   Step, Trace, Verdict (the data contracts)
src/ostutor/slice/programs.py the three programs as Python data
src/ostutor/slice/explorer.py Model, Tier A BFS, classification, witness, critical point
src/ostutor/slice/replay.py   replay(program, witness)
src/ostutor/slice/narrate.py  template narration
src/ostutor/slice/check.py    narration checker + the four tampered examples
src/ostutor/slice/rr.py       optional Round Robin + mutant + diagnosis
src/ostutor/slice/demo.py     CLI
tests/                        explorer, replay, checker, rr
```

## Hand-verified expectations (derived before running the code)

| Program | Label | States | Notes |
|---|---|---|---|
| deadlock_two_locks | deadlock | 19 | witness `T1 lock a, T2 lock b`; critical decision after step 1: `T2:lock b` is the bad choice, `T1:lock b` keeps a safe completion |
| race_no_lock | race on `x` | 9 | after `T1 read x`, T1's write and T2's read conflict; no schedule avoids it, so no critical point |
| safe_with_lock | safe | 16 | 25 pc pairs minus the 9 with both threads inside the lock; one terminal state |
| 2 threads x 10 reads | safe | 121 | 11 x 11 states versus 184,756 interleavings |
| RR quantum 2, P1(0,5) P2(1,3) P3(2,1) | | | correct `P1 P1 P2 P2 P3 P1 P1 P2 P1`; mutant diverges at time unit 5 |

## Conventions worth knowing for the review

- State = (program counters, mutex owners). Shared values are not tracked.
- Race = two threads' next operations hit the same variable, one is a write, and no common lock
  is held. This is a lockset-style check on explored states.
- `step_index` in `critical_point` is the 1-based number of the **bad step**;
  `decision_after_step` is the state before it. `detail` on Verdict and the structured
  `bad_thread` / `safe_thread` fields are small additions to the contract in the plan.
- Narration convention: every thread, mutex and variable is written in backticks, and the
  summary says ``Verdict: `label` ``. The checker depends on this.
- The checker validates facts (indices, names, numbers, label), not causal reasoning.
- Completeness is always `complete_for_bound` here; bounds are filled in to match the contract.

## Not built (by design)

DSL parser, Tier B/C, starvation and lost signal, LLM calls, RAG, API/UI, SQLite cache.


## Added in v0.2 (Phase 1 and 3 pieces)

```bash
python -m ostutor.cli analyze data/programs/dining_3_deadlock.ost          # text program in, verdict out
python -m ostutor.cli analyze prog.ost --llm --db cache.sqlite             # checked LLM narration + cache
python -m ostutor.cli ping                                                  # test your API key
```

| Path | Role |
|---|---|
| `dsl/parser.py` | text DSL -> program dict, loop unrolling, line-numbered errors, canonical hash |
| `narration/evidence.py` | compact evidence the LLM sees (about 1,000 tokens max) |
| `narration/llm.py` | OpenAI-compatible client, retry, fixtures record/replay |
| `narration/pipeline.py` | ask LLM, check, re-prompt twice, fall back to template |
| `cache/store.py` | SQLite cache for verdicts and checked narrations |
| `cli.py` | the command line above |
| `data/programs/*.ost` | six example programs (3 original, dining philosophers x2, loop) |

Setup for LLM access: see `docs/LLM_SETUP.md`.

Still not built: semaphores/condition variables, Tier B/C, starvation, lost signal,
simulators beyond the Round Robin slice, RAG, router, API, UI.

## Streamlit demo

```bash
pip install -e ".[dev]"
python -m ostutor.cli warm        # optional: pre-record LLM narrations for an offline-safe demo
streamlit run streamlit_app.py
```
Tabs: concurrency analysis (verdict, witness, replay, state graph, checked narration, live
"fool the checker"), interleavings vs states, Round Robin mutant diagnosis. See `docs/DEMO.md`
for the five-minute script, fallbacks and likely questions.

## Tests, CI and releases

```bash
make install   # editable install with dev tools
make check     # exactly what CI runs: ruff, mypy, pytest with a 90% coverage floor
make demo      # slice demo with the tamper test
```

| Workflow | When | What it does |
|---|---|---|
| `ci.yml` | every push to `main`, every pull request | lint + types; tests on Python 3.11/3.12/3.13 (Linux) and 3.12 (Windows, macOS); acceptance gate on the *installed* package; builds the wheel and checks it |
| `release.yml` | pushing a tag such as `v0.2.0` | re-runs tests, checks the tag matches `pyproject.toml`, builds, publishes a GitHub Release with the files (PyPI publishing is optional and off by default) |
| `llm-live-check.yml` | manual only | pings the real provider with the `GROQ_API_KEY` repo secret |

The test suite is fully offline: `tests/conftest.py` removes API keys from the environment and
fails any test that opens a network connection, so CI never spends tokens.

Release: bump `version` in `pyproject.toml`, commit, then `git tag v0.2.0 && git push origin v0.2.0`.
