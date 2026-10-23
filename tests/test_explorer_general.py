"""Tests for the generalised explorer (semaphores, conditions, Tier B, Tier C, starvation).

Differential testing, as the build plan requires: the new engine must agree with the
stable slice explorer on every classic program, and Tier B must agree with Tier A.
"""
import random
from pathlib import Path

import pytest

from ostutor.dsl.parser import parse
from ostutor.explorer import analyze, analyze_full, replay
from ostutor.explorer.model import GModel, is_legacy
from ostutor.explorer.pct import analyze_pct
from ostutor.explorer.reduce import automorphisms, make_canon
from ostutor.slice.explorer import ProgramError, StateBudgetExceeded
from ostutor.slice.explorer import analyze as slice_analyze
from ostutor.slice.programs import PROGRAMS

DATA = Path(__file__).parent.parent / "data" / "programs"


def load(name):
    return parse((DATA / f"{name}.ost").read_text()).program


# ------------------------------------------------------------------ random programs
def random_legacy(rng: random.Random):
    """Random small lock/unlock/read/write program with properly paired locks."""
    mutexes = ["a", "b"][: rng.randint(1, 2)]
    shared = ["x"]

    def block(held: tuple, depth: int):
        ops = []
        for _ in range(rng.randint(1, 2)):
            r = rng.random()
            free = [m for m in mutexes if m not in held]
            if r < 0.45 and free and depth < 2:
                m = rng.choice(free)
                ops += [("lock", m)] + block(held + (m,), depth + 1) + [("unlock", m)]
            else:
                ops.append((rng.choice(["read", "write"]), "x"))
        return ops

    n = rng.randint(2, 3)
    return {"mutexes": mutexes, "shared": shared,
            "threads": {f"T{i + 1}": block((), 0) for i in range(n)}}


def random_symmetric(rng: random.Random):
    """k identical threads (so every permutation is an automorphism)."""
    base = random_legacy(rng)
    ops = base["threads"]["T1"]
    k = rng.randint(2, 4)
    return {"mutexes": base["mutexes"], "shared": base["shared"],
            "threads": {f"T{i + 1}": list(ops) for i in range(k)}}


# ------------------------------------------------------------------ differential: vs slice
@pytest.mark.parametrize("name", sorted(PROGRAMS))
def test_general_engine_matches_slice_on_original_programs(name):
    a, b = analyze(PROGRAMS[name], engine="general"), slice_analyze(PROGRAMS[name])
    assert (a.label, a.states_explored, a.critical_point) == (
        b.label, b.states_explored, b.critical_point)
    assert (len(a.witness.steps) if a.witness else None) == (
        len(b.witness.steps) if b.witness else None)


def test_general_engine_matches_slice_on_200_random_programs():
    rng = random.Random(20261016)
    labels = set()
    for _ in range(200):
        p = random_legacy(rng)
        try:
            ref = slice_analyze(p)
        except ProgramError:
            continue
        got = analyze(p, engine="general")
        labels.add(ref.label)
        assert got.label == ref.label, p
        assert got.states_explored == ref.states_explored, p
        assert (len(got.witness.steps) if got.witness else 0) == (
            len(ref.witness.steps) if ref.witness else 0), p
        assert got.critical_point == ref.critical_point, p
        if got.witness:
            assert replay(p, got.witness, got.label).ok, p
    assert {"safe", "race", "deadlock"} <= labels  # the sample exercised all three labels


# ------------------------------------------------------------------ differential: Tier B vs A
def test_tier_b_agrees_with_tier_a_on_150_random_symmetric_programs():
    rng = random.Random(7)
    reduced_somewhere = 0
    for _ in range(150):
        p = random_symmetric(rng)
        a = analyze_full(p, tier="A")[0]
        b = analyze_full(p, tier="B")[0]
        assert a.label == b.label, p
        assert b.completeness == "reduced_but_complete"
        assert b.states_explored <= a.states_explored
        reduced_somewhere += b.states_explored < a.states_explored
        assert (len(a.witness.steps) if a.witness else 0) == (
            len(b.witness.steps) if b.witness else 0), p  # shortest-ness is preserved
        if b.witness:
            assert replay(p, b.witness, b.label).ok  # reduced witnesses are real schedules
    assert reduced_somewhere > 100


@pytest.mark.parametrize("n", [3, 4, 5])
def test_rotated_dining_philosophers_reduce_but_keep_the_verdict(n):
    forks = [f"f{i + 1}" for i in range(n)]
    p = {"mutexes": forks, "shared": [], "threads": {
        f"P{i + 1}": [("lock", forks[i]), ("lock", forks[(i + 1) % n]),
                      ("unlock", forks[(i + 1) % n]), ("unlock", forks[i])]
        for i in range(n)}}
    a, b = analyze_full(p, tier="A")[0], analyze_full(p, tier="B")[0]
    assert a.label == b.label == "deadlock"
    assert len(a.witness.steps) == len(b.witness.steps) == n  # everyone takes a left fork
    assert b.states_explored < a.states_explored
    assert replay(p, b.witness, "deadlock").ok
    assert len(automorphisms(GModel(p))) == n  # the n rotations


def test_dining_5_state_counts_are_reported_honestly():
    p = load("dining_5_deadlock")
    a, b = analyze_full(p, tier="A")[0], analyze_full(p, tier="B")[0]
    assert (a.states_explored, b.states_explored) == (1743, 351)


def test_asymmetric_program_has_no_reduction():
    p = load("dining_3_ordered")  # one philosopher takes forks in the other order
    assert make_canon(GModel(p)) is None
    with pytest.raises(ValueError):
        analyze_full(p, tier="B")


# ------------------------------------------------------------------ semaphores
def test_producer_consumer_ok_is_safe_with_hand_counted_states():
    """Hand count: the consumer is blocked on `full` until the producer finishes, so the
    states are 6 producer positions with the consumer at the start, then 5 more as the
    consumer runs: 6 + 5 = 11."""
    v = analyze(load("producer_consumer_ok"))
    assert v.label == "safe" and v.states_explored == 11 and v.witness is None


def test_producer_consumer_bad_order_deadlocks_in_two_steps():
    p = load("producer_consumer_bad_order")
    v = analyze(p)
    assert v.label == "deadlock"
    assert [(s.thread, s.op, s.object) for s in v.witness.steps] == [
        ("Consumer", "lock", "m"), ("Producer", "down", "empty")]
    assert v.detail["waits"]["Consumer"]["kind"] == "semaphore"
    assert v.detail["waits"]["Producer"] == {"kind": "lock", "object": "m",
                                             "holder": "Consumer"}
    assert v.critical_point["decision_after_step"] == 0
    assert replay(p, v.witness, "deadlock").ok


def test_semaphore_counts_follow_the_trace():
    p = {"mutexes": [], "shared": [], "semaphores": {"s": 2},
         "threads": {"T1": [("down", "s"), ("down", "s")], "T2": [("down", "s")]}}
    assert analyze(p).label == "deadlock"  # three downs, two permits
    p["semaphores"]["s"] = 3
    assert analyze(p).label == "safe"


# ------------------------------------------------------------------ condition variables
def test_lost_signal_is_detected_with_five_step_witness():
    p = load("lost_signal")
    v = analyze(p)
    assert v.label == "lost_signal" and len(v.witness.steps) == 5
    assert [(s.thread, s.op) for s in v.witness.steps][:3] == [
        ("Signaller", "lock"), ("Signaller", "signal"), ("Signaller", "unlock")]
    assert v.detail["lost"] == [{"thread": "Waiter", "cond": "c"}]
    assert v.critical_point["bad_choice"] == "Signaller:lock m"
    assert v.critical_point["safe_alternative"] == "Waiter:lock m"
    assert replay(p, v.witness, "lost_signal").ok
    assert replay(p, v.witness, "deadlock").ok  # a lost signal is also a deadlock


def test_wait_without_a_predicate_is_a_lost_signal_risk_even_if_one_order_works():
    """Waiter-first works, signaller-first loses the signal: the program is still buggy."""
    p = {"mutexes": ["m"], "shared": [], "conds": ["c"], "threads": {
        "A_waiter": [("lock", "m"), ("cwait", "c", "m"), ("cresume", "c", "m"),
                     ("unlock", "m")],
        "B_signal": [("lock", "m"), ("signal", "c"), ("unlock", "m")]}}
    v = analyze_full(p)[0]
    assert v.label == "lost_signal"  # the other order still exists, so the program is buggy
    assert {s.i for s in v.witness.steps} == {1, 2, 3, 4, 5}


def test_broadcast_wakes_every_waiter_and_a_lost_broadcast_is_detected():
    p = {"mutexes": ["m"], "shared": [], "conds": ["c"], "threads": {
        "W1": [("lock", "m"), ("cwait", "c", "m"), ("cresume", "c", "m"), ("unlock", "m")],
        "W2": [("lock", "m"), ("cwait", "c", "m"), ("cresume", "c", "m"), ("unlock", "m")],
        "Z": [("lock", "m"), ("broadcast", "c"), ("unlock", "m")]}}
    v = analyze(p)
    assert v.label == "lost_signal"  # Z first loses the broadcast; but waiters-first is fine
    m = GModel(p)
    s = m.initial()
    for t in (0, 0, 1, 1):  # W1 lock, W1 cwait, W2 lock, W2 cwait
        s = m.step(s, t)
    assert s[3][0] == (0, 1)
    s = m.step(m.step(s, 2), 2)  # Z lock, Z broadcast
    assert s[3][0] == ()  # both woken at once


def test_wait_without_holding_the_mutex_is_a_program_error():
    p = {"mutexes": ["m"], "shared": [], "conds": ["c"],
         "threads": {"T": [("cwait", "c", "m"), ("cresume", "c", "m")]}}
    with pytest.raises(ProgramError):
        analyze_full(p)


# ------------------------------------------------------------------ starvation
def test_unfair_mutex_starves_a_thread_under_weak_but_not_strong_fairness():
    p = load("starvation_unfair_mutex")
    weak = analyze(p, fairness="weak")
    strong = analyze(p, fairness="strong")
    assert weak.label == "starvation" and weak.fairness_assumption == "weak"
    assert strong.label == "safe" and strong.detail["fairness_checked"] == "strong"
    assert weak.states_explored == strong.states_explored == 4  # free + 3 holders
    assert replay(p, weak.witness, "starvation").ok


def test_always_enabled_threads_are_not_starved_under_weak_fairness():
    """Two threads that can always run: a cycle where one never runs is *unfair*, because
    weak fairness forces a continuously enabled thread to run. Guards the weak-fairness rule."""
    p = {"mutexes": [], "shared": ["x"], "threads": {
        "A": [("read", "x"), ("jump", 0)], "B": [("read", "x"), ("jump", 0)]}}
    for fairness in ("weak", "strong"):
        v = analyze(p, fairness=fairness)
        assert v.label == "safe", (fairness, v.label)


def test_thread_blocked_forever_is_starved_under_any_fairness():
    p = load("starvation_blocked_forever")
    for fairness in ("weak", "strong"):
        v = analyze(p, fairness=fairness)
        assert v.label == "starvation" and v.detail["starved"] == "B"
        assert replay(p, v.witness, "starvation").ok


def test_a_cycle_that_ignores_another_enabled_thread_is_unfair():
    """U can run at once (one read) and is enabled in every state until it does, so a cycle
    that never schedules U is unfair. The reported starvation of T must therefore happen
    AFTER U has run: U's step must be in the prefix, before the cycle."""
    p = {"mutexes": ["m"], "shared": ["x"], "semaphores": {"s": 0}, "threads": {
        "A": [("lock", "m"), ("unlock", "m"), ("jump", 0)],
        "T": [("down", "s")],
        "U": [("read", "x")]}}
    v = analyze(p, fairness="weak")
    assert v.label == "starvation" and v.detail["starved"] == "T"
    k = v.witness.params["cycle_start"]
    assert any(st.thread == "U" for st in v.witness.steps[:k]), "U must run before the cycle"
    assert all(st.thread != "U" for st in v.witness.steps[k:])
    assert replay(p, v.witness, "starvation").ok


def test_blocked_thread_is_not_starved_when_only_an_unfair_schedule_blocks_it():
    """T waits for a permit that U would post. U is always enabled, so the only cycle in
    which T never runs is one where the scheduler ignores U forever: unfair, hence no
    starvation under either assumption. Guards the 'other threads' fairness rule."""
    p = {"mutexes": ["m"], "shared": [], "semaphores": {"s": 0}, "threads": {
        "A": [("lock", "m"), ("unlock", "m"), ("jump", 0)],
        "T": [("down", "s")],
        "U": [("up", "s")]}}
    for fairness in ("weak", "strong"):
        v = analyze(p, fairness=fairness)
        assert v.label == "safe", (fairness, v.label, v.detail)


def test_replay_rejects_a_lasso_where_the_starved_thread_runs():
    """Legal, closed cycle, but the thread declared starved is one that runs in it."""
    from copy import deepcopy
    p = load("starvation_unfair_mutex")
    v = analyze(p, fairness="weak")
    cycle_threads = {st.thread for st in v.witness.steps[v.witness.params["cycle_start"]:]}
    assert len(cycle_threads) >= 2
    forged = deepcopy(v.witness)
    forged.params["starved"] = sorted(cycle_threads)[0]
    res = replay(p, forged, "starvation")
    assert not res.ok and "runs inside the cycle" in res.reason


def test_replay_rejects_tampered_starvation_lassos():
    p = load("starvation_unfair_mutex")
    v = analyze(p, fairness="weak")
    from copy import deepcopy
    starved = v.witness.params["starved"]
    # 1. the starved thread secretly runs inside the cycle
    bad = deepcopy(v.witness)
    bad.steps[-2].thread = starved
    assert not replay(p, bad, "starvation").ok
    # 2. the cycle is cut short so it no longer returns to its start
    cut = deepcopy(v.witness)
    cut.steps = cut.steps[:-1]
    assert not replay(p, cut, "starvation").ok
    # 3. claiming a different outcome
    assert not replay(p, v.witness, "deadlock").ok


def test_acyclic_programs_never_report_starvation():
    for name in ("deadlock_two_locks", "safe_with_lock", "producer_consumer_ok"):
        p = load(name)
        assert "starvation" not in analyze(p, engine="general").detail["findings"]


def test_findings_list_fixes_the_masking_limitation():
    """Report limitation 3: a deadlock used to hide a race in the same program."""
    p = {"mutexes": ["a", "b"], "shared": ["x"], "threads": {
        "T1": [("lock", "a"), ("lock", "b"), ("unlock", "b"), ("unlock", "a"), ("write", "x")],
        "T2": [("lock", "b"), ("lock", "a"), ("unlock", "a"), ("unlock", "b"), ("write", "x")]}}
    v = analyze(p, engine="general")
    assert v.label == "deadlock" and v.detail["findings"] == ["deadlock", "race"]


# ------------------------------------------------------------------ Tier C (PCT)
def test_critical_point_is_the_last_state_that_can_still_finish_safely():
    """Hand-checked on the classic deadlock: after T1 locks a, T1 locking b still lets
    the program finish, but T2 locking b dooms it. After step 2 everything is doomed."""
    v = analyze(PROGRAMS["deadlock_two_locks"], engine="general")
    cp = v.critical_point
    assert (cp["decision_after_step"], cp["bad_choice"], cp["safe_alternative"]) == (
        1, "T2:lock b", "T1:lock b")
    assert cp["bad_thread"] == "T2" and cp["safe_thread"] == "T1"


def test_pct_finds_the_classic_deadlock_and_its_witness_replays():
    p = PROGRAMS["deadlock_two_locks"]
    v = analyze_pct(p, runs=500, depth=2, seed=1)
    assert v.label == "deadlock" and v.completeness == "sampled"
    assert replay(p, v.witness, "deadlock").ok
    assert v.detail["per_run_probability_lower_bound"] > 0


def test_pct_never_says_safe():
    v = analyze_pct(PROGRAMS["safe_with_lock"], runs=300, depth=3, seed=0)
    assert v.label == "none_found" and v.completeness == "sampled" and v.witness is None
    assert v.bounds["pct_runs"] == 300


def test_pct_is_reproducible_for_a_given_seed():
    p = load("dining_5_deadlock")
    a, b = analyze_pct(p, runs=400, depth=2, seed=5), analyze_pct(p, runs=400, depth=2, seed=5)
    assert a.label == b.label == "deadlock"
    assert [(s.thread, s.object) for s in a.witness.steps] == [
        (s.thread, s.object) for s in b.witness.steps]


def test_pct_finds_a_lost_signal():
    p = load("lost_signal")
    v = analyze_pct(p, runs=500, depth=2, seed=3)
    assert v.label == "lost_signal" and replay(p, v.witness, "lost_signal").ok


# ------------------------------------------------------------------ escalation
def test_auto_escalates_a_to_b_to_c_when_the_budget_is_too_small():
    sym = load("dining_5_deadlock")
    assert analyze(sym, budget=2000, engine="general").completeness == "complete_for_bound"
    assert analyze(sym, budget=400, engine="general").completeness == "reduced_but_complete"
    assert analyze(sym, budget=50, engine="general").completeness == "sampled"
    with pytest.raises(StateBudgetExceeded):
        analyze(sym, budget=50, tier="A", engine="general")


def test_general_engine_is_the_default_and_the_slice_is_kept_as_an_oracle():
    assert is_legacy(PROGRAMS["deadlock_two_locks"])
    assert not is_legacy(load("producer_consumer_ok"))
    default = analyze(PROGRAMS["deadlock_two_locks"])
    assert "findings" in default.detail  # the general engine's richer verdict
    oracle = analyze(PROGRAMS["deadlock_two_locks"], engine="slice")
    assert "findings" not in oracle.detail
    assert (default.label, default.states_explored) == (oracle.label, oracle.states_explored)
    with pytest.raises(ValueError):
        analyze(load("producer_consumer_ok"), engine="slice")


def test_findings_are_reported_on_the_default_path_for_classic_programs():
    """The report's limitation 3 must be fixed where users actually are, not only in tests
    that force the new engine."""
    p = {"mutexes": ["a", "b"], "shared": ["x"], "threads": {
        "T1": [("lock", "a"), ("lock", "b"), ("unlock", "b"), ("unlock", "a"), ("write", "x")],
        "T2": [("lock", "b"), ("lock", "a"), ("unlock", "a"), ("unlock", "b"), ("write", "x")]}}
    assert analyze(p).detail["findings"] == ["deadlock", "race"]
