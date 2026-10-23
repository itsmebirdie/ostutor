"""Streamlit demo: the engine computes, the LLM explains, a checker verifies.

Run from the repository root:   streamlit run streamlit_app.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, cast

import streamlit as st

from ostutor.cache.store import Cache
from ostutor.diagnosis.engine import diagnose
from ostutor.dsl.parser import ParseError, canonical_hash, parse
from ostutor.explorer import replay
from ostutor.mutants.catalogue import Problem, mutants_for, solve
from ostutor.narration.llm import FixtureStore, LLMClient, LLMConfig
from ostutor.narration.pipeline import analyze_cached, explain
from ostutor.rag.answer import answer as rag_answer
from ostutor.rag.chunk import chunk_pages
from ostutor.rag.index import HashingEmbedder, Index
from ostutor.rag.ingest import load_documents
from ostutor.rag.retrieve import Retriever
from ostutor.router.rules import route
from ostutor.sims.common import ValidationError
from ostutor.slice.check import check, tamper_examples
from ostutor.slice.explorer import ProgramError, StateBudgetExceeded
from ostutor.ui import viewmodel as vm

ROOT = Path(__file__).resolve().parents[3]
EXAMPLES_DIR = ROOT / "data" / "programs"
DB_PATH = str(ROOT / "demo_cache.sqlite")
FIXTURES_DIR = ROOT / "demo_fixtures"
FALLBACK_EXAMPLE = ("mutex a, b\nthread T1:\n  lock a\n  lock b\n  unlock b\n  unlock a\n"
                    "thread T2:\n  lock b\n  lock a\n  unlock a\n  unlock b\n")
LABEL_COLOUR = {"safe": "green", "race": "orange", "deadlock": "red"}


# ----------------------------------------------------------------------------- setup
def _bridge_secrets() -> None:
    """Streamlit Cloud keeps keys in st.secrets; the LLM client reads environment variables."""
    try:
        for k in ("GROQ_API_KEY", "OSTUTOR_API_KEY", "OSTUTOR_MODEL", "OSTUTOR_BASE_URL"):
            if k in st.secrets:
                os.environ.setdefault(k, str(st.secrets[k]))
    except Exception:  # no secrets file: perfectly fine
        pass


@st.cache_resource
def _open_cache(path: str) -> Cache:
    return Cache(path)


def _cache() -> Cache:
    return _open_cache(DB_PATH)  # looked up at call time so tests can redirect it


def _examples() -> dict[str, str]:
    found = {p.stem: p.read_text() for p in sorted(EXAMPLES_DIR.glob("*.ost"))}
    return found or {"deadlock_two_locks": FALLBACK_EXAMPLE}


def _load_example() -> None:
    st.session_state["code"] = _examples()[st.session_state["example"]]
    st.session_state.pop("result", None)


# ----------------------------------------------------------------------------- analysis
def run_analysis(text: str, use_llm: bool, use_fixtures: bool) -> dict:
    parsed = parse(text)
    program = parsed.program
    cache = _cache()
    verdict = analyze_cached(program, cache, parsed.loop_unroll,
                             tier=st.session_state.get("tier", "auto"),
                             fairness=st.session_state.get("fairness", "weak"))
    llm = None
    if use_llm:
        fx = FixtureStore(FIXTURES_DIR, "auto") if use_fixtures else None
        llm = LLMClient(LLMConfig.from_env(), fixtures=fx)
    ex = explain(program, verdict, llm, cache)
    return {
        "program": program, "parsed": parsed, "verdict": verdict, "explanation": ex,
        "replay": replay(program, verdict.witness, expected=verdict.label)
        if verdict.witness else None,
        "dot": vm.state_graph_dot(program, parsed.loop_unroll),
        "schedules": None if any(op[0] == "jump" for o in program["threads"].values()
                                 for op in o)
        else vm.interleavings([len(o) for o in program["threads"].values()]),
        "calls": llm.calls if llm else 0, "tokens": llm.tokens_used if llm else 0,
    }


# ----------------------------------------------------------------------------- tabs
def _tab_analyze(use_llm: bool, use_fixtures: bool) -> None:
    st.text_area("Program (restricted DSL: mutex, semaphore, cond, shared, thread, lock, unlock, "
                 "down, up, wait, signal, read, write, repeat N, forever)", key="code",
                 height=260)
    if st.button("Analyze", type="primary"):
        try:
            with st.spinner("Searching the state graph..."):
                st.session_state["result"] = run_analysis(st.session_state["code"],
                                                          use_llm, use_fixtures)
        except ParseError as e:
            st.session_state.pop("result", None)
            st.error(f"Could not read the program, {e}")
        except (ProgramError, StateBudgetExceeded) as e:
            st.session_state.pop("result", None)
            st.error(f"Analysis stopped: {e}")

    r = st.session_state.get("result")
    if not r:
        st.info("Pick an example in the sidebar or edit the program, then press **Analyze**.")
        return
    v, ex = r["verdict"], r["explanation"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Verdict", v.label.upper())
    c2.metric("States explored", f"{v.states_explored:,}")
    if r["schedules"] is None:  # loops make the number of schedules unbounded
        c3.metric("Possible interleavings", "unbounded")
        c4.metric("Search vs brute force", "n/a (loops)")
    else:
        c3.metric("Possible interleavings", f"{r['schedules']:,}")
        c4.metric("Search vs brute force", f"{r['schedules'] / v.states_explored:,.1f}x fewer")
    st.caption(f"Completeness: **{v.completeness}**. Bounds: {v.bounds}. "
               f"'Complete for bound' means the search finished within these bounds, not a "
               f"proof for all programs. Canonical hash: `{canonical_hash(r['program'])[:23]}`")

    if v.witness:
        left, right = st.columns([3, 2])
        with left:
            st.subheader("Shortest witness")
            st.dataframe(vm.witness_rows(v), hide_index=True)
        with right:
            st.subheader("Independent replay")
            rp = r["replay"]
            if rp and rp.ok:
                st.success(f"A separate re-execution reproduced the failure: {rp.failure}")
            else:
                st.error(f"Replay failed: {rp.reason if rp else 'no replay'}")
            if v.critical_point:
                cp = v.critical_point
                st.warning(f"**Critical decision point** after step "
                           f"{cp['decision_after_step']}: choosing `{cp['bad_choice']}` leaves "
                           f"the safe region; `{cp['safe_alternative']}` keeps a safe "
                           f"completion reachable.")
            else:
                st.info("No critical point: every schedule fails.")
    else:
        st.success("Every explored schedule completes safely, so there is no witness.")

    if r["dot"]:
        st.subheader("State graph")
        st.caption("Green: always safe. Yellow: may fail. Red: must fail. Thick red path: the "
                   "shortest witness. Labels show each thread's program counter and mutex owners.")
        st.graphviz_chart(r["dot"])
    else:
        st.caption("State graph hidden: too many states to draw legibly.")

    st.subheader("Explanation")
    st.markdown(f"**Source:** {ex.badge}")
    if ex.note:
        st.warning(ex.note)
    for c in ex.narration["claims"]:
        st.markdown(f"- **Step {c['step']}.** {c['text']}")
    st.markdown(f"**Summary.** {ex.narration['summary']}")
    st.caption("The checker validates facts (step numbers, names, numbers, the verdict label), "
               "not causal reasoning.")
    if use_llm:
        st.caption(f"LLM calls this analysis: {r['calls']}, tokens: {r['tokens']}. "
                   f"Cache hits (session): {_cache().hits}")

    _fool_the_checker(r)


def _fool_the_checker(r: dict) -> None:
    v, ex, program = r["verdict"], r["explanation"], r["program"]
    st.subheader("Try to fool the checker")
    st.caption("An LLM could invent a step, a thread, a number or a verdict. The checker re-reads "
               "the explanation against the computed trace and rejects each one.")
    kinds = tamper_examples(ex.narration, v)
    cols = st.columns(len(kinds))
    for col, (name, narration) in zip(cols, kinds.items()):
        with col:
            res = check(narration, v, program)
            st.markdown(f"**{name}**")
            if res.ok:
                st.success("accepted")
            else:
                st.error("rejected")
                for viol in res.violations[:2]:
                    st.caption(f"{viol['rule']}: {viol['detail']}")
    st.markdown("**Or edit the summary yourself** (the check re-runs as you type):")
    edited = st.text_area("Summary", value=ex.narration["summary"], key=f"sum_{v.program_hash}")
    live = check({"claims": ex.narration["claims"], "summary": edited}, v, program)
    if live.ok:
        st.success("Checker: facts consistent with the computed trace.")
    else:
        st.error("Checker rejected your edit:")
        for viol in live.violations:
            st.write(f"- **{viol['rule']}**: {viol['detail']}")
    with st.expander("Show the narration as JSON"):
        st.code(json.dumps(ex.narration, indent=2), language="json")


def _tab_scaling() -> None:
    st.subheader("Why not just enumerate (or ask an LLM to enumerate) every schedule?")
    st.write("Two threads, each doing *n* visible steps. The number of interleavings explodes; "
             "the number of distinct states the search must visit grows only quadratically.")
    rows = vm.scaling_rows(10)
    st.vega_lite_chart(
        [{"n": r["steps_per_thread"], "series": "interleavings", "count": r["interleavings"]}
         for r in rows]
        + [{"n": r["steps_per_thread"], "series": "states visited", "count": r["states"]}
           for r in rows],
        {"mark": {"type": "line", "point": True},
         "encoding": {"x": {"field": "n", "type": "quantitative", "title": "steps per thread"},
                      "y": {"field": "count", "type": "quantitative", "scale": {"type": "log"},
                            "title": "count (log scale)"},
                      "color": {"field": "series", "type": "nominal"}}})
    st.dataframe(rows, hide_index=True)
    last = rows[-1]
    st.caption(f"At n=10: {last['interleavings']:,} interleavings vs {last['states']} states "
               f"(about {last['ratio']:,.0f}x). All counts come from running the explorer.")


def _tab_rr() -> None:
    st.subheader("Round Robin: which misconception explains the wrong answer?")
    st.write("The engine runs the correct scheduler and each *mutant* (a deliberately wrong "
             "policy tied to a named misconception) and compares against the student's Gantt chart.")
    procs_default = [{"pid": "P1", "arrival": 0, "burst": 5},
                     {"pid": "P2", "arrival": 1, "burst": 3},
                     {"pid": "P3", "arrival": 2, "burst": 1}]
    edited = st.data_editor(procs_default, num_rows="dynamic", key="procs")
    quantum = int(st.number_input("Time quantum", min_value=1, value=2, step=1))
    try:
        rows = cast("list[dict[str, Any]]", edited)  # list in, list of dicts out
        procs = [(str(p["pid"]), int(p["arrival"]), int(p["burst"])) for p in rows]
        default_student = " ".join(vm.example_student_answer(procs, quantum))
    except (ValueError, TypeError, KeyError) as e:
        st.error(f"Check the process table: {e}")
        return
    answer = st.text_input("Student's Gantt chart (one process id per time unit, space-separated)",
                           value=default_student, key=f"gantt_{default_student}")
    try:
        view = vm.rr_view(procs, quantum, vm.parse_gantt(answer))
    except ValueError as e:
        st.error(str(e))
        return
    d, div = view["diagnosis"], view["divergence"]
    n = max(len(view["correct"]), len(view["student"]))
    pad = lambda xs: xs + ["-"] * (n - len(xs))  # noqa: E731
    table = {"Correct": pad(view["correct"]), "Student": pad(view["student"])}
    import pandas as pd

    df = pd.DataFrame(table, index=None).T
    df.columns = [str(i) for i in range(1, n + 1)]

    def hl(_):
        styles = pd.DataFrame("", index=df.index, columns=df.columns)
        if div:
            styles.iloc[:, div - 1] = "background-color: #ffcdd2; color: black"
        return styles

    st.dataframe(df.style.apply(hl, axis=None))
    if d.method == "correct":
        st.success("The answer matches the correct trace.")
    elif d.method == "mutant_match":
        st.error(f"Matches misconception `{d.misconception_id}`. First divergence at time unit "
                 f"{div}: expected `{view['diagnosis'].first_divergence['expected']}`, got "
                 f"`{view['diagnosis'].first_divergence['got']}`.")
        st.write(d.confidence_note)
    else:
        st.warning(f"No named mutant matches. First divergence at time unit {div}.")
        st.write(d.confidence_note)
    st.caption("Only one mutant exists so far, so no match-rate claim can be made.")


PRESETS: dict[str, Problem] = {
    "CPU: Round Robin": Problem("cpu", "rr", {
        "procs": [("P1", 0, 5), ("P2", 1, 3), ("P3", 2, 1)], "quantum": 2}),
    "CPU: SRTF": Problem("cpu", "srtf", {
        "procs": [("A", 0, 7), ("B", 2, 4), ("C", 4, 1), ("D", 5, 4)]}),
    "CPU: Priority": Problem("cpu", "priority", {
        "procs": [("A", 0, 3, 3), ("B", 0, 2, 1), ("C", 0, 1, 2)]}),
    "Banker's: safety": Problem("banker", "safety", {
        "available": [3, 3, 2],
        "maximum": [[7, 5, 3], [3, 2, 2], [9, 0, 2], [2, 2, 2], [4, 3, 3]],
        "allocation": [[0, 1, 0], [2, 0, 0], [3, 0, 2], [2, 1, 1], [0, 0, 2]]}),
    "Paging: LRU (3 frames)": Problem("paging", "lru", {
        "refs": [7, 0, 1, 2, 0, 3, 0, 4, 2, 3, 0, 3, 2, 1, 2, 0, 1, 7, 0, 1], "frames": 3}),
    "Paging: FIFO (3 frames)": Problem("paging", "fifo", {
        "refs": [7, 0, 1, 2, 0, 3, 0, 4, 2, 3, 0, 3, 2, 1, 2, 0, 1, 7, 0, 1], "frames": 3}),
    "Paging: OPT (3 frames)": Problem("paging", "opt", {
        "refs": [7, 0, 1, 2, 0, 3, 0, 4, 2, 3, 0, 3, 2, 1, 2, 0, 1, 7, 0, 1], "frames": 3}),
    "Disk: SSTF": Problem("disk", "sstf", {
        "requests": [98, 183, 37, 122, 14, 124, 65, 67], "head": 53}),
    "Disk: SCAN (head moving down)": Problem("disk", "scan", {
        "requests": [98, 183, 37, 122, 14, 124, 65, 67], "head": 53, "direction": "down"}),
    "Disk: C-SCAN": Problem("disk", "cscan", {
        "requests": [98, 183, 37, 122, 14, 124, 65, 67], "head": 53}),
}


def _tab_diagnose() -> None:
    import pandas as pd
    st.subheader("Diagnose a wrong answer: any scheduling or memory topic")
    st.write("The engine solves the problem exactly, then compares the student's answer with "
             "the correct trace and with every *mutant* (a named misconception). Matches are "
             "found by exact comparison, not by an LLM.")
    name = st.selectbox("Problem", list(PRESETS), key="diag_preset")
    problem = PRESETS[name]
    try:
        trace = solve(problem)
    except ValidationError as e:
        st.error(str(e))
        return
    with st.expander("Problem data"):
        st.json(problem.params)
    mistakes = {m.id: m for m in mutants_for(problem) if m.run(problem) != trace.answer}
    choice = st.selectbox("Fill the answer with a typical mistake", ["(correct answer)"]
                          + list(mistakes), key=f"diag_mistake_{name}")
    default = trace.answer if choice == "(correct answer)" else mistakes[choice].run(problem)
    answer = st.text_input("Student's answer (space-separated; or a single number)",
                           value=" ".join(default), key=f"diag_answer_{name}_{choice}")
    if not answer.strip():
        st.info("Enter an answer.")
        return
    d = diagnose(problem, answer)
    if problem.topic == "disk":
        st.line_chart(pd.DataFrame({"cylinder": [problem.params["head"]]
                                    + [int(c) for c in trace.answer]}))
    st.markdown("**Correct answer:** " + " ".join(f"`{t}`" for t in trace.answer))
    if d.method == "correct":
        st.success(d.confidence_note)
    elif d.method == "mutant_match":
        label = d.misconception_id or "several misconceptions"
        st.error(f"Matches a known misconception: `{label}`")
        for desc in d.descriptions.values():
            st.write(f"- {desc}")
        if d.ambiguous:
            st.warning(d.confidence_note)
    else:
        st.warning(f"{d.method.replace('_', ' ')}: {d.confidence_note}")
        for desc in d.descriptions.values():
            st.write(f"- {desc}")
    if d.first_divergence:
        fd = d.first_divergence
        st.write(f"First divergence at position {fd['step']}: expected `{fd['expected']}`, "
                 f"got `{fd['got']}`.")
    if d.edit_distance is not None:
        st.caption(f"Edit distance to the correct answer: {d.edit_distance}. "
                   f"Nearest mutant: {d.nearest}.")
    st.caption("Wrong answers here are synthetic (generated from the mutants themselves), so "
               "this demonstrates the mechanism, not diagnostic accuracy.")


@st.cache_resource
def _retriever() -> Retriever:
    corpus = os.environ.get("OSTUTOR_CORPUS", str(ROOT / "data" / "corpus_sample"))
    return Retriever(Index(chunk_pages(load_documents([corpus])), HashingEmbedder()))


def _tab_ask() -> None:
    st.subheader("Ask the course material (and see how the router decides)")
    st.caption("Layer 0: hybrid retrieval (BM25 + dense, fused by reciprocal rank fusion) over "
               "the course notes; answers are extracted from retrieved chunks with citations, "
               "and off-topic questions get 'not found'. Sample notes are loaded by default; "
               "set OSTUTOR_CORPUS to point at your own material.")
    q = st.text_input("Question", value="How does chmod 755 work?", key="ask_q")
    if not q.strip():
        return
    r = route(q)
    st.write(f"**Router:** `{r.layer}`, because {r.reason}. Rules fired: {', '.join(r.rules)}")
    if r.layer == "analyze_code":
        st.info("This looks like a program: paste it in the **Concurrency analysis** tab.")
    elif r.layer in ("simulate", "diagnose"):
        st.info("This looks like a scheduling/memory problem: use the **Diagnose any topic** tab "
                "(or POST structured data to the API).")
    else:
        a = rag_answer(q, _retriever())
        if a.found:
            st.success(a.text)
            for c in a.citations:
                st.caption(f"Source: {c}")
        else:
            st.warning(a.text)


# ----------------------------------------------------------------------------- main
def main() -> None:
    st.set_page_config(page_title="OS Tutor: checked explanations", layout="wide")
    _bridge_secrets()
    cfg = LLMConfig.from_env()

    st.title("OS Tutor")
    st.markdown("**The engine computes. The LLM only explains. A checker verifies the "
                "explanation against what was computed.**")

    with st.sidebar:
        st.header("Program")
        names = list(_examples())
        st.selectbox("Example", names, key="example", on_change=_load_example,
                     index=names.index("deadlock_two_locks") if "deadlock_two_locks" in names
                     else 0)
        st.header("Search")
        st.selectbox("Tier", ["auto", "A", "B", "C"], key="tier",
                     help="auto: exact search, then symmetry reduction, then PCT sampling if "
                          "the state budget is exceeded. A exhaustive, B symmetry-reduced, "
                          "C sampled (never reported as safe).")
        st.selectbox("Fairness (starvation)", ["weak", "strong"], key="fairness",
                     help="Starvation is only reported under a stated fairness assumption.")
        st.header("Explanation")
        has_key = bool(cfg.api_key)
        use_llm = st.toggle("Narrate with the LLM", value=False, disabled=False,
                            help="Off: deterministic templates. On: the LLM writes the text and "
                                 "the checker must accept it, otherwise the template is shown.")
        use_fixtures = st.toggle("Replay recorded responses (offline-safe)", value=True,
                                 help="Uses demo_fixtures/ when present; records new answers "
                                      "there. Makes the demo repeatable and spends no tokens.")
        if use_llm and not has_key and not use_fixtures:
            st.warning("No API key found: the template will be shown. Set GROQ_API_KEY.")
        st.caption(f"Model: `{cfg.model}`  \nKey: {'set' if has_key else 'not set'}")
        st.divider()
        st.caption("Built: explorer (Tiers A/B/C, semaphores, conditions, starvation, lost "
                   "signal), DSL parser, checker, cache, CPU/Banker's/paging/disk simulators "
                   "with 16 mutants, diagnosis, hybrid RAG, router, API.")

    if "code" not in st.session_state:
        st.session_state["code"] = _examples()[st.session_state.get("example",
                                                                    list(_examples())[0])]
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Concurrency analysis", "Why state-graph search",
                                            "Round Robin diagnosis", "Diagnose any topic",
                                            "Ask the course"])
    with tab1:
        _tab_analyze(use_llm, use_fixtures)
    with tab2:
        _tab_scaling()
    with tab3:
        _tab_rr()
    with tab4:
        _tab_diagnose()
    with tab5:
        _tab_ask()
