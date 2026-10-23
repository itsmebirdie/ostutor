"""The labelled concurrency set for the classifier evaluation (Build Plan 10.2).

PROVENANCE, READ THIS BEFORE QUOTING ANY NUMBER: every expected label below was derived by
construction/reasoning (written next to each program) BEFORE running the explorer. They were
NOT labelled by two independent human annotators, as the plan requires for a final claim.
Before reporting precision/recall in the report, each of you must verify every label by hand
(or label independently and report disagreements). Until then the result is "agreement with
construction labels", not accuracy.

expected is a label, or {"weak": label, "strong": label} when fairness changes the answer.
"""
from __future__ import annotations


def lock_pair(first: str, second: str, work: str = "") -> str:
    body = f"  lock {first}\n{work}  lock {second}\n  unlock {second}\n  unlock {first}\n"
    return body


def prog(*threads: tuple[str, str], decl: str = "") -> str:
    return decl + "".join(f"thread {name}:\n{body}" for name, body in threads)


def ring(n: int, ordered_fix: bool = False) -> str:
    forks = [f"f{i + 1}" for i in range(n)]
    out = f"mutex {', '.join(forks)}\n"
    for i in range(n):
        a, b = forks[i], forks[(i + 1) % n]
        if ordered_fix and i == n - 1:
            a, b = b, a  # the last philosopher takes the lower-numbered fork first
        out += f"thread P{i + 1}:\n" + lock_pair(a, b)
    return out


DATASET: list[dict] = [
    # ---- deadlocks from lock ordering ------------------------------------------------
    dict(name="dl_inversion_basic", family="lock order", expected="deadlock",
         why="T1 takes a then b, T2 takes b then a: classic circular wait",
         code=prog(("T1", lock_pair("a", "b")), ("T2", lock_pair("b", "a")), decl="mutex a, b\n")),
    dict(name="dl_inversion_with_work", family="lock order", expected="deadlock",
         why="same inversion with reads/writes inside; x is only touched while both locks "
             "are held, so no race, just the deadlock",
         code=prog(("T1", "  lock a\n  read x\n  lock b\n  write x\n  unlock b\n  unlock a\n"),
                   ("T2", "  lock b\n  read x\n  lock a\n  write x\n  unlock a\n  unlock b\n"),
                   decl="mutex a, b\nshared x\n")),
    dict(name="dl_inversion_unlock_order", family="lock order", expected="deadlock",
         why="acquisition order is inverted; the release order is irrelevant",
         code=prog(("T1", "  lock a\n  lock b\n  unlock a\n  unlock b\n"),
                   ("T2", "  lock b\n  lock a\n  unlock b\n  unlock a\n"), decl="mutex a, b\n")),
    dict(name="dl_three_cycle", family="lock order", expected="deadlock",
         why="T1 a->b, T2 b->c, T3 c->a closes a 3-cycle",
         code=prog(("T1", lock_pair("a", "b")), ("T2", lock_pair("b", "c")),
                   ("T3", lock_pair("c", "a")), decl="mutex a, b, c\n")),
    dict(name="dl_dining_3", family="lock order", expected="deadlock",
         why="3 philosophers each take the left fork first", code=ring(3)),
    dict(name="dl_dining_4", family="lock order", expected="deadlock",
         why="4 philosophers each take the left fork first", code=ring(4)),
    dict(name="dl_self_relock", family="lock order", expected="deadlock",
         why="a non-recursive mutex locked twice by one thread blocks it forever "
             "(the other thread terminates)",
         code=prog(("T1", "  lock a\n  lock a\n  unlock a\n  unlock a\n"), ("T2", "  read x\n"),
                   decl="mutex a\nshared x\n")),
    dict(name="dl_and_race_together", family="lock order", expected="deadlock",
         why="inverted locks AND an unprotected write after the critical section; deadlock "
             "outranks race, and the findings list must still mention the race",
         code=prog(("T1", "  lock a\n  lock b\n  unlock b\n  unlock a\n  write x\n"),
                   ("T2", "  lock b\n  lock a\n  unlock a\n  unlock b\n  write x\n"),
                   decl="mutex a, b\nshared x\n")),
    # ---- safe lock usage --------------------------------------------------------------
    dict(name="safe_ordered_2", family="lock order", expected="safe",
         why="both threads take a then b", code=prog(
             ("T1", lock_pair("a", "b")), ("T2", lock_pair("a", "b")), decl="mutex a, b\n")),
    dict(name="safe_ordered_3", family="lock order", expected="safe",
         why="global order a<b<c respected by T1 (a,b), T2 (b,c), T3 (a,c): no cycle",
         code=prog(("T1", lock_pair("a", "b")), ("T2", lock_pair("b", "c")),
                   ("T3", lock_pair("a", "c")), decl="mutex a, b, c\n")),
    dict(name="safe_dining_3_ordered", family="lock order", expected="safe",
         why="last philosopher takes the lower fork first: ordering breaks the cycle",
         code=ring(3, ordered_fix=True)),
    dict(name="safe_dining_4_ordered", family="lock order", expected="safe",
         why="same fix for 4 philosophers", code=ring(4, ordered_fix=True)),
    dict(name="safe_single_lock_2", family="locks", expected="safe",
         why="every access to x is inside the same mutex",
         code=prog(*[(t, "  lock m\n  read x\n  write x\n  unlock m\n") for t in ("T1", "T2")],
                   decl="mutex m\nshared x\n")),
    dict(name="safe_single_lock_3", family="locks", expected="safe",
         why="three identical threads, all accesses inside m",
         code=prog(*[(t, "  lock m\n  read x\n  write x\n  unlock m\n")
                     for t in ("T1", "T2", "T3")], decl="mutex m\nshared x\n")),
    dict(name="safe_readers_only", family="locks", expected="safe",
         why="reads never conflict with reads",
         code=prog(*[(t, "  read x\n") for t in ("T1", "T2", "T3")], decl="shared x\n")),
    dict(name="safe_private_variables", family="locks", expected="safe",
         why="the threads write different variables",
         code=prog(("T1", "  write x\n"), ("T2", "  write y\n"), decl="shared x, y\n")),
    # ---- data races -------------------------------------------------------------------
    dict(name="race_two_writers", family="race", expected="race",
         why="two unprotected writes to x",
         code=prog(("T1", "  write x\n"), ("T2", "  write x\n"), decl="shared x\n")),
    dict(name="race_read_write", family="race", expected="race",
         why="an unprotected read concurrent with a write",
         code=prog(("T1", "  read x\n"), ("T2", "  write x\n"), decl="shared x\n")),
    dict(name="race_read_modify_write", family="race", expected="race",
         why="the lost-update program: read x; write x in both threads, no lock",
         code=prog(*[(t, "  read x\n  write x\n") for t in ("T1", "T2")], decl="shared x\n")),
    dict(name="race_one_side_locked", family="race", expected="race",
         why="T1 locks, T2 does not: no common lock at the conflicting accesses",
         code=prog(("T1", "  lock m\n  write x\n  unlock m\n"), ("T2", "  write x\n"),
                   decl="mutex m\nshared x\n")),
    dict(name="race_different_locks", family="race", expected="race",
         why="x is protected by a in T1 and by b in T2: different locks protect nothing",
         code=prog(("T1", "  lock a\n  write x\n  unlock a\n"),
                   ("T2", "  lock b\n  write x\n  unlock b\n"), decl="mutex a, b\nshared x\n")),
    dict(name="race_write_after_unlock", family="race", expected="race",
         why="T1 writes x after releasing m while T2 reads x inside m",
         code=prog(("T1", "  lock m\n  read x\n  unlock m\n  write x\n"),
                   ("T2", "  lock m\n  read x\n  write x\n  unlock m\n"),
                   decl="mutex m\nshared x\n")),
    dict(name="race_loop_unprotected", family="race", expected="race",
         why="a loop of unprotected read-modify-writes",
         code=prog(*[(t, "  repeat 2:\n    read x\n    write x\n") for t in ("T1", "T2")],
                   decl="shared x\n")),
    dict(name="safe_loop_locked", family="locks", expected="safe",
         why="the same loop with every iteration inside m",
         code=prog(*[(t, "  repeat 2:\n    lock m\n    read x\n    write x\n    unlock m\n")
                     for t in ("T1", "T2")], decl="mutex m\nshared x\n")),
    # ---- semaphores -------------------------------------------------------------------
    dict(name="sem_producer_consumer_ok", family="semaphore", expected="safe",
         why="empty/full order the threads and m guards the buffer (11 states, hand-counted)",
         code=prog(("Producer", "  down empty\n  lock m\n  write buf\n  unlock m\n  up full\n"),
                   ("Consumer", "  down full\n  lock m\n  read buf\n  unlock m\n  up empty\n"),
                   decl="mutex m\nsemaphore empty = 1\nsemaphore full = 0\nshared buf\n")),
    dict(name="sem_producer_consumer_bad_order", family="semaphore", expected="deadlock",
         why="the consumer holds m while waiting for full; the producer needs m to post it",
         code=prog(("Producer", "  down empty\n  lock m\n  write buf\n  unlock m\n  up full\n"),
                   ("Consumer", "  lock m\n  down full\n  read buf\n  unlock m\n  up empty\n"),
                   decl="mutex m\nsemaphore empty = 1\nsemaphore full = 0\nshared buf\n")),
    dict(name="sem_binary_mutual_exclusion", family="semaphore", expected="safe",
         why="a semaphore initialised to 1 excludes the second thread from the access",
         code=prog(*[(t, "  down s\n  read x\n  write x\n  up s\n") for t in ("T1", "T2")],
                   decl="semaphore s = 1\nshared x\n")),
    dict(name="sem_counting_two_permits", family="semaphore", expected="race",
         why="two permits let both threads into the 'critical section' at once",
         code=prog(*[(t, "  down s\n  write x\n  up s\n") for t in ("T1", "T2")],
                   decl="semaphore s = 2\nshared x\n")),
    dict(name="sem_handoff_any_order", family="semaphore", expected="safe",
         why="a semaphore remembers a post made before the wait, so either order works",
         code=prog(("T1", "  down s\n"), ("T2", "  up s\n"), decl="semaphore s = 0\n")),
    dict(name="sem_missing_post", family="semaphore", expected="deadlock",
         why="nobody ever posts s, so T1 blocks forever after T2 finishes",
         code=prog(("T1", "  down s\n"), ("T2", "  read x\n"), decl="semaphore s = 0\nshared x\n")),
    dict(name="sem_two_semaphore_cycle", family="semaphore", expected="deadlock",
         why="T1 takes s1 then s2, T2 takes s2 then s1 (circular wait on semaphores)",
         code=prog(("T1", "  down s1\n  down s2\n  up s2\n  up s1\n"),
                   ("T2", "  down s2\n  down s1\n  up s1\n  up s2\n"),
                   decl="semaphore s1 = 1\nsemaphore s2 = 1\n")),
    # ---- condition variables ----------------------------------------------------------
    dict(name="cond_signal_may_be_lost", family="condition", expected="lost_signal",
         why="if the signaller runs first its signal is lost and the waiter sleeps forever",
         code=prog(("Waiter", "  lock m\n  wait c, m\n  unlock m\n"),
                   ("Signaller", "  lock m\n  signal c\n  unlock m\n"), decl="mutex m\ncond c\n")),
    dict(name="cond_broadcast_may_be_lost", family="condition", expected="lost_signal",
         why="same bug with broadcast",
         code=prog(("Waiter", "  lock m\n  wait c, m\n  unlock m\n"),
                   ("Signaller", "  lock m\n  broadcast c\n  unlock m\n"),
                   decl="mutex m\ncond c\n")),
    dict(name="cond_two_waiters_one_signal", family="condition", expected="lost_signal",
         why="a schedule exists where the single signal comes first and is lost; (another "
             "leaves one waiter unwoken, so findings also include deadlock)",
         code=prog(("W1", "  lock m\n  wait c, m\n  unlock m\n"),
                   ("W2", "  lock m\n  wait c, m\n  unlock m\n"),
                   ("Z", "  lock m\n  signal c\n  unlock m\n"), decl="mutex m\ncond c\n")),
    # ---- starvation (needs a fairness assumption) --------------------------------------
    dict(name="starve_unfair_mutex_3", family="starvation",
         expected={"weak": "starvation", "strong": "safe"},
         why="3 threads loop on m; weak fairness allows B,C to hand m back and forth forever "
             "while A is only intermittently enabled; strong fairness forbids that",
         code=prog(*[(t, "  forever:\n    lock m\n    unlock m\n") for t in ("A", "B", "C")],
                   decl="mutex m\n")),
    dict(name="starve_unfair_mutex_2", family="starvation",
         expected={"weak": "starvation", "strong": "safe"},
         why="the same argument with only two contenders",
         code=prog(*[(t, "  forever:\n    lock m\n    unlock m\n") for t in ("A", "B")],
                   decl="mutex m\n")),
    dict(name="starve_blocked_forever", family="starvation",
         expected={"weak": "starvation", "strong": "starvation"},
         why="B waits for a permit nobody posts while A runs forever: starved under any fairness",
         code=prog(("A", "  forever:\n    lock m\n    unlock m\n"), ("B", "  down s\n"),
                   decl="mutex m\nsemaphore s = 0\n")),
    dict(name="starve_none_free_threads", family="starvation",
         expected={"weak": "safe", "strong": "safe"},
         why="two always-enabled looping threads: fairness forces both to run",
         code=prog(*[(t, "  forever:\n    read x\n") for t in ("A", "B")], decl="shared x\n")),
    dict(name="starve_released_by_third", family="starvation",
         expected={"weak": "safe", "strong": "safe"},
         why="T is blocked only until U posts; ignoring U forever is an unfair schedule",
         code=prog(("A", "  forever:\n    lock m\n    unlock m\n"), ("T", "  down s\n"),
                   ("U", "  up s\n"), decl="mutex m\nsemaphore s = 0\n")),
]

assert len({d["name"] for d in DATASET}) == len(DATASET), "duplicate program names"
