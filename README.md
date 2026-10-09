# OS Tutor: review slice

A thin vertical cut through the design: a deterministic explorer computes the verdict,
the shortest witness and the critical decision point; template narration is then
**mechanically checked** against what was computed. No LLM, no network, no dependencies
except `pytest` for the tests.

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
