"""Router rules. The labelled messages are written to exercise each rule; the pass rate on
them says the rules do what we meant, NOT how a real student's messages would be routed."""
import pytest

from ostutor.router.rules import route

CODE = """mutex a, b
thread T1:
    lock a
    lock b
    unlock b
    unlock a
"""

CASES = [
    (CODE, "analyze_code"),
    ("Is this deadlock-free?\n" + CODE, "analyze_code"),
    ("```\nthread T1:\n  lock m\n  unlock m\n```\nwill it deadlock with mutex?", "analyze_code"),
    ("Banker's algorithm: Available 3 3 2, is the state safe? Allocation 0 1 0 2 0 0 3 0 2", "simulate"),
    ("Compute the average waiting time for FCFS with bursts 24 3 3", "simulate"),
    ("Page replacement: reference string 7 0 1 2 0 3 0 4, 3 frames, how many faults with LRU?", "simulate"),
    ("Disk scheduling: head at 53, requests 98 183 37 122 14 124 65 67, total SSTF seek?", "simulate"),
    ("My Gantt chart for round robin with quantum 2 is P1 P1 P2 P2 P1 P1 P3 P2 P1, is this right?", "diagnose"),
    ("I got 15 page faults with LRU for the reference string 7 0 1 2 0 3 0 4 2 3 0 3 2 1 2 0 1 7 0 1", "diagnose"),
    ("what did I do wrong in my banker's safe sequence P0 P1 P2 P3 P4", "diagnose"),
    ("What is the difference between FIFO and LRU?", "rag"),
    ("Explain what a safe state means in the Banker's algorithm", "rag"),
    ("How does chmod 755 work?", "rag"),
    ("What does fork return in the child process?", "rag"),
    ("what is a deadlock", "rag"),
    ("Explain the difference between a mutex and a semaphore", "rag"),
    ("will my program deadlock", "rag"),     # asks to analyse but pastes no code
    ("hello", "rag"),
]


@pytest.mark.parametrize("msg,layer", CASES)
def test_routes(msg, layer):
    r = route(msg)
    assert r.layer == layer, (r.layer, r.rules, msg)
    assert r.rules and r.reason  # every route explains itself


def test_topic_is_reported_for_simulations():
    assert route("Compute SJF waiting time for bursts 6 8 7 3").topic == "cpu"
    assert route("Banker's algorithm available 3 3 2 allocation 0 1 0 2 0 0").topic == "banker"


def test_empty_and_none_do_not_crash():
    assert route("").layer == "rag" and route(None).layer == "rag"  # type: ignore[arg-type]


def test_asking_to_analyse_without_code_explains_what_to_do():
    r = route("can my code deadlock")
    assert r.layer == "rag" and "paste" in r.reason
