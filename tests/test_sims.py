"""Golden tests for the exact simulators. Sources are the standard textbook worked
examples (Silberschatz, Galvin, Gagne, *Operating System Concepts*): CPU scheduling
chapter (FCFS, SJF, RR examples), deadlock chapter (Banker's), virtual memory chapter
(page replacement, Belady's anomaly) and mass-storage chapter (disk scheduling)."""
import pytest

from ostutor.sims import banker, cpu, disk, paging
from ostutor.sims.common import ValidationError
from ostutor.slice import rr as slice_rr

P3 = [("P1", 0, 24), ("P2", 0, 3), ("P3", 0, 3)]
ALLOC = [[0, 1, 0], [2, 0, 0], [3, 0, 2], [2, 1, 1], [0, 0, 2]]
MAX = [[7, 5, 3], [3, 2, 2], [9, 0, 2], [2, 2, 2], [4, 3, 3]]
AVAIL = [3, 3, 2]
REFS = [7, 0, 1, 2, 0, 3, 0, 4, 2, 3, 0, 3, 2, 1, 2, 0, 1, 7, 0, 1]
DISK_REQ = [98, 183, 37, 122, 14, 124, 65, 67]


# ------------------------------------------------------------------ CPU
def test_fcfs_textbook_example():
    t = cpu.simulate(P3, "fcfs")
    assert t.answer == ["P1"] * 24 + ["P2"] * 3 + ["P3"] * 3
    assert t.metrics["avg_waiting"] == 17.0  # (0 + 24 + 27) / 3


def test_sjf_textbook_example():
    t = cpu.simulate([("P1", 0, 6), ("P2", 0, 8), ("P3", 0, 7), ("P4", 0, 3)], "sjf")
    assert [p for i, p in enumerate(t.answer) if i == 0 or t.answer[i - 1] != p] == [
        "P4", "P1", "P3", "P2"]
    assert t.metrics["avg_waiting"] == 7.0


def test_round_robin_textbook_example_quantum_4():
    t = cpu.simulate(P3, "rr", quantum=4)
    assert t.metrics["per_process"]["P1"]["waiting"] == 6
    assert t.metrics["per_process"]["P2"]["waiting"] == 4
    assert t.metrics["per_process"]["P3"]["waiting"] == 7
    assert round(t.metrics["avg_waiting"], 2) == 5.67  # 17/3 (books print 5.66)


def test_round_robin_matches_the_report_trace_and_the_slice_simulator():
    procs = [("P1", 0, 5), ("P2", 1, 3), ("P3", 2, 1)]
    new = cpu.simulate(procs, "rr", quantum=2).answer
    assert new == "P1 P1 P2 P2 P3 P1 P1 P2 P1".split() == slice_rr.simulate(procs, 2)
    wrong = cpu.simulate(procs, "rr", quantum=2, rr_tie="preempted_first").answer
    assert wrong == "P1 P1 P2 P2 P1 P1 P3 P2 P1".split()


def test_srtf_preempts_and_priority_orders_by_urgency():
    t = cpu.simulate([("A", 0, 7), ("B", 2, 4), ("C", 4, 1), ("D", 5, 4)], "srtf")
    # hand trace: A 0-2; B preempts at 2 (4 < 5 left); C preempts at 4 (1 < 2 left);
    # at 5 B (2 left) beats D (4) and A (5); D 7-11; A finishes 11-16 (16 = 7+4+1+4 units)
    assert t.answer == "A A B B C B B D D D D A A A A A".split()
    p = cpu.simulate([("A", 0, 3, 3), ("B", 0, 2, 1), ("C", 0, 1, 2)], "priority")
    assert p.answer == ["B", "B", "C", "A", "A", "A"]


def test_idle_gaps_are_shown():
    assert cpu.simulate([("A", 2, 1)], "fcfs").answer == ["idle", "idle", "A"]
    assert cpu.simulate([("A", 3, 1)], "rr", quantum=2).answer == ["idle"] * 3 + ["A"]


def test_mlfq_demotes_after_a_full_quantum():
    t = cpu.simulate([("A", 0, 6), ("B", 0, 2)], "mlfq", quanta=[2, 4, 8])
    # A uses its level-0 quantum (2), B runs (2, done), then A continues at level 1
    assert t.answer == ["A", "A", "B", "B", "A", "A", "A", "A"]


@pytest.mark.parametrize("algo", ["fcfs", "sjf", "srtf", "priority", "rr", "mlfq"])
def test_cpu_invariants(algo):
    procs = [("A", 0, 5, 2), ("B", 1, 3, 1), ("C", 2, 4, 3), ("D", 9, 2, 1)]
    t = cpu.simulate(procs, algo)
    assert sum(1 for x in t.answer if x != "idle") == sum(p[2] for p in procs)
    for pid, m in t.metrics["per_process"].items():
        assert m["waiting"] >= 0 and m["turnaround"] >= m["waiting"]
        arrival = next(p[1] for p in procs if p[0] == pid)
        assert t.answer.index(pid) >= arrival  # nothing runs before it arrives


@pytest.mark.parametrize("bad,msg", [
    ([], "at least one"), ([("A", -1, 3)], "arrival"), ([("A", 0, 0)], "burst"),
    ([("A", 0, 3), ("A", 1, 2)], "duplicate"), ([("A", 0)], "expected"),
    ([("A", 0, 2.5)], "burst")])
def test_cpu_validation(bad, msg):
    with pytest.raises(ValidationError, match=msg):
        cpu.simulate(bad, "fcfs")


def test_cpu_rejects_unknown_algorithm_and_bad_quantum():
    with pytest.raises(ValidationError):
        cpu.simulate(P3, "lottery")
    with pytest.raises(ValidationError):
        cpu.simulate(P3, "rr", quantum=0)


# ------------------------------------------------------------------ Banker's
def test_banker_need_matrix_and_a_valid_safe_sequence():
    t = banker.safety(AVAIL, MAX, ALLOC)
    assert t.metrics["safe"] and t.metrics["need"] == [
        [7, 4, 3], [1, 2, 2], [6, 0, 0], [0, 1, 1], [4, 3, 1]]
    assert t.answer == ["P1", "P3", "P0", "P2", "P4"]  # lowest-index tie rule
    assert banker.is_valid_safe_sequence(AVAIL, MAX, ALLOC, t.answer)


def test_other_safe_sequences_are_also_valid_and_invalid_ones_are_rejected():
    assert banker.is_valid_safe_sequence(AVAIL, MAX, ALLOC, "P1 P3 P4 P2 P0".split())
    assert banker.is_valid_safe_sequence(AVAIL, MAX, ALLOC, "P1 P3 P4 P0 P2".split())
    assert not banker.is_valid_safe_sequence(AVAIL, MAX, ALLOC, "P0 P1 P2 P3 P4".split())
    assert not banker.is_valid_safe_sequence(AVAIL, MAX, ALLOC, "P1 P3".split())


def test_banker_request_textbook_walkthrough():
    """Grant P1 (1,0,2); in the NEW state P4 (3,3,0) must wait (not available) and
    P0 (0,2,0) must wait (hand check: available (2,1,0) then nobody's need fits)."""
    g = banker.request(AVAIL, MAX, ALLOC, 1, [1, 0, 2])
    assert g.answer == ["GRANT"] and g.metrics["granted"]
    assert g.metrics["safe_sequence"] == ["P1", "P3", "P0", "P2", "P4"]
    avail2 = [2, 3, 0]
    alloc2 = [list(r) for r in ALLOC]
    alloc2[1] = [3, 0, 2]
    assert banker.request(avail2, MAX, alloc2, 4, [3, 3, 0]).answer == ["WAIT"]
    unsafe = banker.request(avail2, MAX, alloc2, 0, [0, 2, 0])
    assert unsafe.answer == ["WAIT"] and "unsafe" in unsafe.metrics["reason"]


def test_banker_unsafe_state_and_request_over_claim():
    t = banker.safety([0, 0, 0], MAX, ALLOC)
    assert not t.metrics["safe"] and t.answer == ["UNSAFE"]
    assert banker.request(AVAIL, MAX, ALLOC, 1, [9, 0, 0]).answer == ["DENIED"]


def test_banker_validation_runs_before_any_computation():
    with pytest.raises(ValidationError, match="columns"):
        banker.safety([3, 3], MAX, ALLOC)
    with pytest.raises(ValidationError, match="exceeds maximum"):
        banker.safety(AVAIL, [[1, 1, 1]], [[2, 0, 0]])
    with pytest.raises(ValidationError, match=">= 0"):
        banker.safety([-1, 3, 2], MAX, ALLOC)
    with pytest.raises(ValidationError, match="initial state is unsafe"):
        banker.request([0, 0, 0], MAX, ALLOC, 0, [0, 0, 0])


# ------------------------------------------------------------------ paging
def test_page_replacement_textbook_counts():
    assert {a: paging.simulate(REFS, 3, a).metrics["faults"] for a in paging.ALGORITHMS} == {
        "fifo": 15, "lru": 12, "opt": 9}


def test_belady_anomaly_with_fifo_but_not_lru():
    ref = [1, 2, 3, 4, 1, 2, 5, 1, 2, 3, 4, 5]
    assert paging.simulate(ref, 3, "fifo").metrics["faults"] == 9
    assert paging.simulate(ref, 4, "fifo").metrics["faults"] == 10  # more frames, MORE faults
    assert paging.simulate(ref, 3, "lru").metrics["faults"] >= paging.simulate(
        ref, 4, "lru").metrics["faults"]


def test_opt_is_never_worse_than_fifo_or_lru():
    import random
    rng = random.Random(4)
    for _ in range(60):
        refs = [rng.randint(0, 6) for _ in range(rng.randint(5, 25))]
        n = rng.randint(1, 4)
        f = {a: paging.simulate(refs, n, a).metrics["faults"] for a in paging.ALGORITHMS}
        assert f["opt"] <= f["fifo"] and f["opt"] <= f["lru"]


def test_paging_trace_rows_and_validation():
    t = paging.simulate([1, 2, 1, 3], 2, "lru")
    assert t.answer == ["F", "F", "H", "F"]
    assert t.rows[3]["evicted"] == 2 and t.rows[3]["frames"] == [1, 3]
    with pytest.raises(ValidationError):
        paging.simulate([1, -2], 2, "fifo")
    with pytest.raises(ValidationError):
        paging.simulate([1], 0, "fifo")


# ------------------------------------------------------------------ disk
def test_disk_scheduling_textbook_totals_and_orders():
    r = {a: disk.simulate(DISK_REQ, 53, a, direction="down" if a == "scan" else "up")
         for a in ("fcfs", "sstf", "scan", "cscan")}
    assert [r[a].metrics["total_head_movement"] for a in r] == [640, 236, 236, 382]
    assert r["sstf"].metrics["order"] == "65 67 37 14 98 122 124 183".split()
    assert r["scan"].metrics["order"] == "37 14 65 67 98 122 124 183".split()
    assert r["cscan"].metrics["order"] == "65 67 98 122 124 183 14 37".split()
    # the answer is the head PATH: SCAN visits cylinder 0, C-SCAN visits 199 then jumps to 0
    assert r["scan"].answer == "37 14 0 65 67 98 122 124 183".split()
    assert r["cscan"].answer == "65 67 98 122 124 183 199 0 14 37".split()
    assert r["scan"].metrics["path"][:4] == [53, 37, 14, 0]  # SCAN really goes to cylinder 0


def test_look_and_clook_turn_at_the_last_request():
    look = disk.simulate(DISK_REQ, 53, "look", direction="down")
    assert look.metrics["total_head_movement"] == (53 - 14) + (183 - 14) == 208
    clook = disk.simulate(DISK_REQ, 53, "clook")
    assert clook.metrics["order"] == "65 67 98 122 124 183 14 37".split()
    assert look.answer == "37 14 65 67 98 122 124 183".split()  # LOOK never visits cylinder 0
    assert clook.metrics["total_head_movement"] == (183 - 53) + (183 - 14) + (37 - 14) == 322
    assert disk.simulate(DISK_REQ, 53, "clook", count_return=False).metrics[
        "total_head_movement"] == (183 - 53) + (37 - 14)


def test_disk_validation():
    with pytest.raises(ValidationError):
        disk.simulate([300], 0, "fcfs")
    with pytest.raises(ValidationError):
        disk.simulate([5], 0, "fcfs", direction="sideways")
    with pytest.raises(ValidationError):
        disk.simulate([5], 0, "elevator")
