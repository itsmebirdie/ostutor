"""Diagnosis tests. NOTE (design rule 6): the wrong answers here are SYNTHETIC, generated
by running the mutants themselves, so they only prove the plumbing works. They support no
accuracy claim; diagnostic accuracy needs real student answers."""
import random

import pytest

from ostutor.diagnosis.engine import Diagnosis, diagnose, first_divergence, levenshtein, normalise
from ostutor.mutants.catalogue import CATALOGUE, Problem, by_id, mutants_for, solve
from ostutor.sims.common import ValidationError
from ostutor.slice import rr as slice_rr

ALLOC = [[0, 1, 0], [2, 0, 0], [3, 0, 2], [2, 1, 1], [0, 0, 2]]
MAX = [[7, 5, 3], [3, 2, 2], [9, 0, 2], [2, 2, 2], [4, 3, 3]]
AVAIL = [3, 3, 2]
REFS = [7, 0, 1, 2, 0, 3, 0, 4, 2, 3, 0, 3, 2, 1, 2, 0, 1, 7, 0, 1]


def gen_cpu(rng, algo):
    n = rng.randint(3, 5)
    procs = [(f"P{i + 1}", rng.randint(0, 6), rng.randint(1, 6), rng.randint(1, 4))
             for i in range(n)]
    params = {"procs": procs, "quantum": rng.randint(2, 3)}
    if algo == "mlfq":
        params["quanta"] = [2, 3, 5]
    return Problem("cpu", algo, params)


def gen_paging(rng, algo):
    return Problem("paging", algo, {"refs": [rng.randint(0, 5) for _ in range(rng.randint(8, 20))],
                                    "frames": rng.randint(2, 4)})


def gen_disk(rng, algo):
    return Problem("disk", algo, {"requests": [rng.randint(0, 199) for _ in range(rng.randint(4, 8))],
                                  "head": rng.randint(0, 199), "direction": rng.choice(["up", "down"])})


def find_instance(m, gen, tries=400, seed=1):
    rng = random.Random(seed)
    for _ in range(tries):
        p = gen(rng, m.algorithm)
        try:
            if m.run(p) != solve(p).answer:
                return p
        except ValidationError:
            continue
    return None


FIXED = {
    "banker_max_instead_of_need": Problem("banker", "safety", {
        "available": AVAIL, "maximum": MAX, "allocation": ALLOC}),
    "banker_forgets_to_reclaim": Problem("banker", "safety", {
        "available": AVAIL, "maximum": MAX, "allocation": ALLOC}),
    "banker_skips_safety_check": Problem("banker", "request", {
        "available": [2, 3, 0], "maximum": MAX, "process": 0, "request": [0, 2, 0],
        "allocation": [[0, 1, 0], [3, 0, 2], [3, 0, 2], [2, 1, 1], [0, 0, 2]]}),
}
GEN = {"cpu": gen_cpu, "paging": gen_paging, "disk": gen_disk}


def instance_for(m):
    return FIXED.get(m.id) or find_instance(m, GEN[m.topic])


@pytest.mark.parametrize("m", CATALOGUE, ids=lambda m: m.id)
def test_every_mutant_is_reachable_and_diagnosed(m):
    """A question exists where the mutant differs from the truth (no dead catalogue
    entries), and the engine names it when given exactly that wrong answer."""
    p = instance_for(m)
    assert p is not None, f"no question distinguishes {m.id} from the correct algorithm"
    wrong = m.run(p)
    assert wrong != solve(p).answer
    d = diagnose(p, wrong)
    assert m.id in d.all_matches and m.description in d.descriptions.values()
    if d.method == "correct":
        # only possible for a head path that is word-for-word the correct SERVICE ORDER
        # (SCAN vs LOOK): the engine must say so instead of guessing
        assert m.topic == "disk" and "which they meant" in d.confidence_note
    else:
        assert d.method == "mutant_match"
        assert d.first_divergence is not None and d.systematic is True


def test_report_example_matches_the_slice_diagnosis():
    procs = [("P1", 0, 5), ("P2", 1, 3), ("P3", 2, 1)]
    p = Problem("cpu", "rr", {"procs": procs, "quantum": 2})
    student = slice_rr.simulate(procs, 2, "preempted_first")
    new, old = diagnose(p, student), slice_rr.diagnose(student, procs, 2)
    assert new.misconception_id == old.misconception_id == "rr_preempted_before_new_arrival"
    assert new.first_divergence == old.first_divergence == {
        "step": 5, "expected": "P3", "got": "P1"}


def test_correct_answer_is_recognised_in_several_input_forms():
    p = Problem("cpu", "rr", {"procs": [("P1", 0, 5), ("P2", 1, 3), ("P3", 2, 1)], "quantum": 2})
    for form in ("P1 P1 P2 P2 P3 P1 P1 P2 P1", "P1, P1, P2, P2, P3, P1, P1, P2, P1",
                 ["P1", "P1", "P2", "P2", "P3", "P1", "P1", "P2", "P1"]):
        assert diagnose(p, form).method == "correct"


def test_two_mutants_that_coincide_are_both_reported_not_one_picked():
    rng = random.Random(3)
    a, b = by_id("opt_evicts_nearest"), by_id("opt_confused_with_lru")
    for _ in range(3000):
        p = gen_paging(rng, "opt")
        wa, wb = a.run(p), b.run(p)
        if wa == wb and wa != solve(p).answer:
            d = diagnose(p, wa)
            assert d.method == "mutant_match" and d.ambiguous
            assert set(d.all_matches) == {a.id, b.id} and d.misconception_id is None
            assert "cannot tell them apart" in d.confidence_note
            return
    pytest.fail("no question found where the two OPT mutants coincide")


def test_single_edit_slip_is_reported_as_a_slip_with_its_cascade():
    p = Problem("paging", "lru", {"refs": REFS, "frames": 3})
    correct = solve(p).answer
    student = list(correct)
    student[10] = "H" if student[10] == "F" else "F"
    assert student not in [m.run(p) for m in mutants_for(p)]
    d = diagnose(p, student)
    assert d.method == "edit_distance" and d.systematic is False
    assert d.edit_distance == 1 and d.first_divergence["step"] == 11
    assert d.cascade == {"first_wrong_step": 11, "wrong_positions": 1, "total_positions": 20}
    assert "slip" in d.confidence_note


def test_far_from_everything_is_unclassified_not_guessed():
    p = Problem("disk", "sstf", {"requests": [98, 183, 37, 122, 14, 124, 65, 67], "head": 53})
    d = diagnose(p, "183 183 183 183 183 183 183 183")
    assert d.method == "unclassified" and d.misconception_id is None


def test_numeric_answers_are_weak_evidence_and_say_so():
    p = Problem("paging", "lru", {"refs": REFS, "frames": 3})
    assert diagnose(p, "12").method == "correct"
    d = diagnose(p, 15)  # what an LRU that never refreshes recency would count
    assert d.method == "numeric_match" and d.all_matches == ["lru_no_update_on_hit"]
    assert "weak evidence" in d.confidence_note
    assert diagnose(p, "99").method == "unclassified"


def test_disk_scan_vs_look_is_told_apart_by_the_head_path():
    p = Problem("disk", "scan", {"requests": [98, 183, 37, 122, 14, 124, 65, 67], "head": 53,
                                 "direction": "down"})
    assert diagnose(p, "37 14 0 65 67 98 122 124 183").method == "correct"      # head path
    assert diagnose(p, "37 14 65 67 98 122 124 183").method == "correct"        # service order
    d = diagnose(p, "37 14 65 67 98 122 124 183".replace("14 65", "14 65"))
    # a path that skips the edge is the LOOK mistake only when it is not just the order form
    p2 = Problem("disk", "scan", {"requests": [98, 183, 37, 122, 14, 124, 65, 67], "head": 53,
                                  "direction": "up"})
    wrong = "65 67 98 122 124 183 37 14"  # reverses at 183 without visiting 199
    d2 = diagnose(p2, wrong)
    assert d2.method in ("mutant_match", "correct") and d.method == "correct"


def test_disk_total_movement_as_a_number():
    p = Problem("disk", "sstf", {"requests": [98, 183, 37, 122, 14, 124, 65, 67], "head": 53})
    assert diagnose(p, "236").method == "correct"
    assert diagnose(p, "640").all_matches == ["sstf_picks_first_in_queue"]  # FCFS total


def test_banker_a_different_valid_safe_sequence_is_correct():
    p = Problem("banker", "safety", {"available": AVAIL, "maximum": MAX, "allocation": ALLOC})
    d = diagnose(p, "P1 P3 P4 P2 P0")  # the book's order; ours (lowest index) differs
    assert d.method == "correct" and "also valid" in d.confidence_note
    assert diagnose(p, "P0 P1 P2 P3 P4").method != "correct"


def test_banker_unsafe_claim_for_a_safe_state_matches_the_max_misconception():
    p = Problem("banker", "safety", {"available": AVAIL, "maximum": MAX, "allocation": ALLOC})
    d = diagnose(p, "UNSAFE")
    assert d.method == "mutant_match"
    assert set(d.all_matches) == {"banker_max_instead_of_need", "banker_forgets_to_reclaim"}
    assert d.ambiguous  # on this state both mistakes give the same wrong answer


def test_banker_grant_without_safety_check():
    p = FIXED["banker_skips_safety_check"]
    assert solve(p).answer == ["WAIT"]
    d = diagnose(p, "GRANT")
    assert d.misconception_id == "banker_skips_safety_check"


class StubLLM:
    def __init__(self, text):
        self.text, self.calls = text, 0

    def complete(self, system, user):
        self.calls += 1
        return self.text


def test_llm_hypothesis_is_last_resort_labelled_and_validated():
    p = Problem("disk", "sstf", {"requests": [98, 183, 37, 122, 14, 124, 65, 67], "head": 53})
    far = "183 183 183 183 183 183 183 183"
    llm = StubLLM("Maybe the student always picks the largest cylinder `183`, a guess.")
    d = diagnose(p, far, llm)
    assert d.method == "llm_hypothesis" and "Unverified" in d.confidence_note and llm.calls == 1
    assert d.misconception_id is None
    # a hypothesis mentioning something not in the data is discarded
    d2 = diagnose(p, far, StubLLM("It involves `ghost`."))
    assert d2.method == "unclassified"
    # the LLM is never consulted when the engine already has an answer
    spy = StubLLM("x")
    diagnose(p, "98 183 37 122 14 124 65 67", spy)  # FCFS order: matches a known mutant
    q = Problem("cpu", "rr", {"procs": [("P1", 0, 5), ("P2", 1, 3), ("P3", 2, 1)], "quantum": 2})
    diagnose(q, "P1 P1 P2 P2 P1 P1 P3 P2 P1", spy)
    assert spy.calls == 0


def test_llm_failure_does_not_break_diagnosis():
    class Broken:
        def complete(self, *a):
            raise RuntimeError("down")
    p = Problem("disk", "sstf", {"requests": [98, 183, 37, 122, 14, 124, 65, 67], "head": 53})
    assert diagnose(p, "183 183 183 183 183 183 183 183", Broken()).method == "unclassified"


def test_garbage_in_never_crashes():
    rng = random.Random(9)
    p = Problem("cpu", "rr", {"procs": [("P1", 0, 5), ("P2", 1, 3), ("P3", 2, 1)], "quantum": 2})
    for _ in range(100):
        junk = [rng.choice(["P1", "P2", "P3", "idle", "x", "7"]) for _ in range(rng.randint(1, 12))]
        assert isinstance(diagnose(p, junk), Diagnosis)


def test_validation_errors():
    p = Problem("cpu", "rr", {"procs": [("P1", 0, 5)], "quantum": 2})
    with pytest.raises(ValidationError):
        diagnose(p, "")
    with pytest.raises(ValidationError):
        diagnose(Problem("cpu", "rr", {}), "P1")
    with pytest.raises(ValidationError):
        diagnose(Problem("memory", "magic", {}), "x")
    with pytest.raises(ValidationError):
        Problem.from_dict({"algorithm": "rr"})


def test_helpers():
    assert levenshtein(list("kitten"), list("sitting")) == 3
    assert first_divergence(["a", "b"], ["a", "b", "c"]) == {"step": 3, "expected": "<end>", "got": "c"}
    assert first_divergence(["a"], ["a"]) is None
    assert normalise("A, B  C") == ["A", "B", "C"] and normalise(15) == ["15"]
