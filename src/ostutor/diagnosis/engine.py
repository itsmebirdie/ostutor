"""Deterministic misconception diagnosis (Build Plan 8.4).

1. Run the correct simulator and every applicable mutant on the question.
2. Exact match to one or more mutants  -> name them ALL (two mutants can coincide on a
   question; we say so rather than pick one) and show the first divergence.
3. No match -> edit distance to the correct trace and to each mutant. A small distance to
   the correct trace suggests a random slip; report where it cascades.
4. Only if all of that fails, and only when a model is supplied, ask an LLM for a
   hypothesis, clearly labelled unverified.

Synthetic wrong answers are for unit tests only (matching mutants against mutant-generated
data is circular); no accuracy claim can come from this module.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from ostutor.mutants.catalogue import Mutant, Problem, mutants_for, solve
from ostutor.sims import banker
from ostutor.sims.common import ValidationError

END = "<end>"


@dataclass
class Diagnosis:
    method: str  # correct | mutant_match | numeric_match | edit_distance | llm_hypothesis |
    #              unclassified
    misconception_id: str | None
    first_divergence: dict | None
    systematic: bool | None
    confidence_note: str
    all_matches: list[str] = field(default_factory=list)
    descriptions: dict = field(default_factory=dict)
    edit_distance: int | None = None
    nearest: dict | None = None
    cascade: dict | None = None

    @property
    def ambiguous(self) -> bool:
        return len(self.all_matches) > 1

    def to_dict(self) -> dict:
        d = asdict(self)
        d["ambiguous"] = self.ambiguous
        return d


def normalise(answer) -> list[str]:
    """Accept 'P1 P1 P2', ['P1', 'P1'], or a number; return clean string tokens."""
    if isinstance(answer, str):
        answer = re.split(r"[\s,;]+", answer.strip())
    if isinstance(answer, (int, float)) and not isinstance(answer, bool):
        return [str(answer)]
    return [str(t).strip() for t in answer if str(t).strip()]


def levenshtein(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def first_divergence(correct: list[str], got: list[str]) -> dict | None:
    for i in range(max(len(correct), len(got))):
        e = correct[i] if i < len(correct) else END
        g = got[i] if i < len(got) else END
        if e != g:
            return {"step": i + 1, "expected": e, "got": g}
    return None


def _scalar(problem: Problem, trace) -> str | None:
    m = trace.metrics
    key = {"paging": "faults", "disk": "total_head_movement", "cpu": "avg_waiting"}.get(
        problem.topic)
    if key is None:
        return None
    v = m[key]
    return str(int(v)) if float(v).is_integer() else f"{v:.2f}".rstrip("0")


def _is_number(tokens: list[str]) -> bool:
    return len(tokens) == 1 and re.fullmatch(r"-?\d+(\.\d+)?", tokens[0]) is not None


def order_of(path: list[str], requests: list[int]) -> list[str]:
    """Service order recovered from a head path: entries that consume a pending request."""
    pending = [str(r) for r in requests]
    out = []
    for c in path:
        if c in pending:
            pending.remove(c)
            out.append(c)
    return out


def _forms(problem: Problem, answer: list[str]) -> list[list[str]]:
    """Equivalent ways a student may write the same answer (disk: head path or order)."""
    if problem.topic == "disk":
        return [answer, order_of(answer, problem.params["requests"])]
    return [answer]


def _correct_check(problem: Problem, student: list[str], correct: list[str]) -> bool:
    if problem.topic == "banker" and problem.algorithm == "safety":
        a = problem.params
        if correct == ["UNSAFE"]:
            return student == ["UNSAFE"]
        return banker.is_valid_safe_sequence(a["available"], a["maximum"], a["allocation"],
                                             student)
    return student in _forms(problem, correct)


def diagnose(problem: Problem, student_answer, llm=None) -> Diagnosis:
    student = normalise(student_answer)
    if not student:
        raise ValidationError("the student's answer is empty")
    trace = solve(problem)
    correct = trace.answer
    muts: list[Mutant] = mutants_for(problem)
    desc = {m.id: m.description for m in muts}

    # --- a single number (fault count, total head movement, average waiting time)
    scalar = _scalar(problem, trace)
    if _is_number(student) and scalar is not None:
        if student[0] == scalar:
            return Diagnosis("correct", None, None, None,
                             "The number matches the correct result.")
        hits = [m for m in muts if _scalar(problem, _mutant_trace(m, problem)) == student[0]]
        if hits:
            return Diagnosis(
                "numeric_match", None, None, None,
                "A single number is weak evidence: it equals what the misconception(s) below "
                "would produce, but other mistakes could give the same number. Ask for the "
                "full sequence to be sure.", [m.id for m in hits],
                {m.id: m.description for m in hits})
        return Diagnosis("unclassified", None, None, None,
                         f"The number differs from the correct result ({scalar}) and matches "
                         "no known misconception; ask the student for the full sequence.")

    if _correct_check(problem, student, correct):
        if problem.topic == "disk" and student != correct:
            lookalikes = [m for m in muts if _safe_run(m, problem) == student]
            if lookalikes:
                ids = [m.id for m in lookalikes]
                return Diagnosis(
                    "correct", None, None, None,
                    "Correct as a service order. But if this was meant as the head path, it "
                    "omits a cylinder the head must visit, which is the mistake described "
                    "below: ask the student which they meant.", ids,
                    {m.id: m.description for m in lookalikes})
        note = "The answer is correct."
        if student != correct:
            note = ("The sequence differs from the reference one but is also valid: safe "
                    "sequences are not unique.")
        return Diagnosis("correct", None, None, None, note)

    div = first_divergence(correct, student)
    matches = [m for m in muts if m.matches(problem, student)
               or (m.accepts is None and student in _forms(problem, _safe_run(m, problem)))]
    if matches:
        ids = [m.id for m in matches]
        note = ("An exact match to a known misconception.")
        if len(matches) > 1:
            note = ("Several known misconceptions give exactly this answer on this question, "
                    "so it cannot tell them apart; try a question where they differ.")
        return Diagnosis("mutant_match", ids[0] if len(ids) == 1 else None, div, True, note,
                         ids, {m.id: m.description for m in matches})

    dc = levenshtein(student, correct)
    scored = sorted((levenshtein(student, _safe_run(m, problem)), m.id) for m in muts)
    nearest: dict | None = ({"id": scored[0][1], "distance": scored[0][0]} if scored else None)
    n = max(len(correct), 1)
    after = sum(1 for i in range(div["step"] - 1, max(len(correct), len(student)))
                if (correct[i] if i < len(correct) else END)
                != (student[i] if i < len(student) else END)) if div else 0
    cascade = {"first_wrong_step": div["step"] if div else None, "wrong_positions": after,
               "total_positions": max(len(correct), len(student))}
    assert div is not None  # not correct, so the answers differ somewhere
    if dc <= max(1, round(0.15 * n)):
        return Diagnosis(
            "edit_distance", None, div, False,
            f"Close to the correct answer ({dc} edit{'s' if dc != 1 else ''}): this looks like "
            "a slip rather than a systematic misconception. The error starts at step "
            f"{div['step']} and affects {after} of {cascade['total_positions']} positions.",
            [], {}, dc, nearest, cascade)
    if nearest is not None and nearest["distance"] < dc:
        return Diagnosis(
            "edit_distance", None, div, None,
            f"No exact match. Closest known misconception: {nearest['id']} (distance "
            f"{nearest['distance']} versus {dc} from the correct answer). This is a hint, "
            "not a diagnosis.", [], {nearest["id"]: desc[nearest["id"]]}, dc, nearest, cascade)
    if llm is not None:
        hyp = _llm_hypothesis(llm, problem, correct, student, div)
        if hyp:
            return Diagnosis("llm_hypothesis", None, div, None,
                             "Unverified hypothesis from a language model: the engine found no "
                             f"matching misconception. {hyp}", [], {}, dc, nearest, cascade)
    return Diagnosis("unclassified", None, div, None,
                     "The answer matches no known misconception and is not close to the "
                     "correct one.", [], {}, dc, nearest, cascade)


def _mutant_trace(m: Mutant, problem: Problem):
    """A tiny trace-like object so scalar metrics of mutants can be compared."""
    from ostutor.mutants.catalogue import SimTrace
    ans = m.run(problem)
    base = solve(problem)
    metrics = dict(base.metrics)
    if problem.topic == "paging":
        metrics["faults"] = ans.count("F")
    elif problem.topic == "disk":
        # total movement along the mutant's head path (counts a C-SCAN return jump)
        pos = problem.params["head"]
        total = 0
        for r in ans:
            total += abs(int(r) - pos)
            pos = int(r)
        metrics["total_head_movement"] = total
    return SimTrace(base.kind, m.id, ans, metrics)


def _safe_run(m: Mutant, problem: Problem) -> list[str]:
    try:
        return m.run(problem)
    except (ValidationError, KeyError):
        return []


def _llm_hypothesis(llm, problem: Problem, correct, student, div) -> str | None:
    system = ("You are an Operating Systems tutor. A student's answer differs from the "
              "correct one. Suggest ONE plausible misconception in at most two sentences. "
              "Do not invent numbers; refer only to the tokens shown. Say it is a guess.")
    user = (f"Topic: {problem.topic}, algorithm: {problem.algorithm}.\n"
            f"Correct answer: {' '.join(correct)}\nStudent answer: {' '.join(student)}\n"
            f"First difference at step {div['step']}: expected {div['expected']}, "
            f"got {div['got']}.")
    try:
        text = llm.complete(system, user).strip()
    except Exception:  # any provider failure just means no hypothesis
        return None
    allowed = set(correct) | set(student)
    for tok in re.findall(r"`([^`]+)`", text):
        if tok not in allowed:
            return None  # refers to something that is not in the data
    return text[:400] if text else None
