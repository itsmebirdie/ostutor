# Demo guide (Streamlit)

## One-time setup (PowerShell, from the repo root)

```powershell
pip install -e ".[dev]"                 # includes streamlit
python -m ostutor.cli ping              # confirms your key + model work
python -m ostutor.cli warm              # records one LLM narration per example (about 6 calls)
git add demo_fixtures; git commit -m "Record demo fixtures"
streamlit run streamlit_app.py          # opens http://localhost:8501
```

`warm` is the important step. It stores each real LLM answer in `demo_fixtures/`, so on demo
day the app can replay them: identical output every time, no tokens spent, and it still works
if the Wi-Fi or the API fails. The fixtures are keyed by model name, so if you change
`OSTUTOR_MODEL`, run `warm` again. `demo_cache.sqlite` is local and git-ignored.

Before presenting: browser zoom 125%, sidebar visible, run the whole script once, and keep a
terminal open with `python -m ostutor.slice.demo --all --tamper` as a backup.

## Five-minute script (follows build plan 2.9)

| Min | Do this in the app | Say this |
|---|---|---|
| 0:00 | Slide, no app yet | LLMs explain OS concepts fluently but invent schedules and misdiagnose errors. Students cannot tell a fabricated trace from a real one. |
| 0:30 | **Concurrency analysis** tab, example `deadlock_two_locks`, press **Analyze** | The engine, not the LLM, computed this: verdict, shortest witness (2 steps), 19 states. Point at the **replay** box: a separate function re-executed the witness and reproduced the deadlock. |
| 1:15 | Point at the **state graph** | Green is always safe, red is doomed. The thick red path is the witness. The **critical decision point** is where the schedule left the safe region: after T1 locks `a`, letting T2 lock `b` dooms it. |
| 2:00 | Sidebar: switch **Narrate with the LLM** on, press Analyze | The LLM receives only the computed evidence (about 1,000 tokens) and writes the explanation. The badge says "checked (LLM)": a mechanical checker accepted it. If it fails twice, the template shows instead. |
| 2:45 | Scroll to **Try to fool the checker** | An LLM might invent a step, a thread, a number or a verdict. All four are rejected. Now edit the summary: change `deadlock` to `safe` and watch the rejection appear live. |
| 3:30 | Sidebar: switch examples: `race_no_lock`, `safe_with_lock`, then `dining_3_deadlock` and `dining_3_ordered` | Three labels. Dining philosophers deadlocks; reordering one philosopher's forks gives *safe*. Parsing a text program, not hard-coded Python. |
| 4:15 | **Why state-graph search** tab | 10 steps each: 184,756 interleavings, but only 121 states. An LLM cannot enumerate that reliably; the search visits 121. |
| 4:45 | **Round Robin diagnosis** tab | A wrong Gantt chart is matched to a named misconception with the first divergence highlighted at time unit 5. Deterministic, no LLM. |
| 5:15 | Architecture slide | Built: explorer, parser, checker, cache, one mutant. Designed, not built: RAG, Tier B/C, other simulators. |

## Say these honestly (they are in the report)
- Three programs plus a few more examples show the mechanism; this is **not a statistical
  evaluation**, and there is no accuracy figure. No real student data has been collected.
- The checker validates **facts, not causal reasoning**. The UI says so.
- "complete for bound" means the search finished within the stated bounds, not a proof for
  all programs.
- Only one Round Robin mutant exists, so there is no match rate.

## If something goes wrong

| Problem | What to do |
|---|---|
| Wi-Fi or API down | Leave **Replay recorded responses** on; recorded answers still show as "checked (LLM)". Or turn the LLM toggle off: templates need no network. |
| Banner says "template fall-back" | The LLM was unavailable or failed the checker. That is the design working (rule 5). Say so and move on. |
| `model does not exist` | Groq retired it. `python -m ostutor.cli models`, set `OSTUTOR_MODEL`, re-run `warm`. |
| Streamlit will not start | `python -m ostutor.slice.demo --all --tamper` shows the same story in the terminal. |
| Edited program shows an error | The message names the line and the unsupported construct (for example `wait`); fix it or switch examples. |

## Likely questions
- **Why not just prompt an LLM?** It cannot reliably enumerate interleavings; the scaling tab shows the size of the problem.
- **What stops the LLM lying?** It never produces numbers, schedules or labels; it receives them, and the checker rejects any text that contradicts the computed trace.
- **What is the API limit risk?** Template-first, cached, and recorded fixtures; the demo degrades gracefully.
- **How will you get real student data?** Consent and institutional approval; until then the diagnosis evaluation stays synthetic and makes no accuracy claim.

## Hosting it online (optional)
Streamlit Community Cloud can host the app from your GitHub repo (main file
`streamlit_app.py`; `requirements.txt` is included). **Do not put your API key in the secrets of
a public app**: anyone could burn your daily quota. Either keep the app private, or deploy with
no key and the committed `demo_fixtures/` so visitors see recorded, checked narrations only.
A local demo on your laptop is the safest choice for the review.
