"""Tier A explorer: hashed state-graph search (BFS), classification, witness.

State = (pcs, owners)
  pcs    : tuple, program counter of each thread (threads sorted by name)
  owners : tuple, index of the thread holding each mutex, or None
Shared variable values are NOT tracked; race detection does not need them.

Rules (Build Plan 2.6):
  * a thread is enabled unless its next op is `lock` on a mutex held by someone
  * deadlock  = non-terminal state with no enabled operation
  * race      = in one state, two different threads' NEXT ops access the same
                shared variable, at least one is a write, and the two threads
                hold no common mutex (lockset-style check on explored states)
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Optional

from ostutor.common.types import Step, Trace, Verdict, program_hash

DEFAULT_BUDGET = 300_000
VALID_OPS = {"lock", "unlock", "read", "write"}

State = tuple


class ProgramError(ValueError):
    pass


class StateBudgetExceeded(RuntimeError):
    """Tier A ran out of budget (escalation to Tier B/C is out of slice scope)."""


class Model:
    def __init__(self, program: dict):
        self.program = program
        self.threads: list[str] = sorted(program["threads"])
        self.ops = [[tuple(o) for o in program["threads"][t]] for t in self.threads]
        self.mutexes: list[str] = list(program["mutexes"])
        self.shared: list[str] = list(program["shared"])
        self._m = {m: i for i, m in enumerate(self.mutexes)}
        self._validate()

    def _validate(self) -> None:
        for t, ops in zip(self.threads, self.ops):
            for op in ops:
                if len(op) != 2 or op[0] not in VALID_OPS:
                    raise ProgramError(f"{t}: bad operation {op!r}")
                kind, obj = op
                if kind in ("lock", "unlock") and obj not in self._m:
                    raise ProgramError(f"{t}: unknown mutex {obj!r}")
                if kind in ("read", "write") and obj not in self.shared:
                    raise ProgramError(f"{t}: unknown shared variable {obj!r}")

    # -- state machine ------------------------------------------------------
    def initial(self) -> State:
        return (tuple(0 for _ in self.threads), tuple(None for _ in self.mutexes))

    def next_op(self, state: State, t: int) -> Optional[tuple]:
        pc = state[0][t]
        return self.ops[t][pc] if pc < len(self.ops[t]) else None

    def is_terminal(self, state: State) -> bool:
        return all(pc == len(o) for pc, o in zip(state[0], self.ops))

    def enabled(self, state: State) -> list[int]:
        out = []
        for t in range(len(self.threads)):
            op = self.next_op(state, t)
            if op is None:
                continue
            if op[0] == "lock" and state[1][self._m[op[1]]] is not None:
                continue
            out.append(t)
        return out

    def step(self, state: State, t: int) -> State:
        pcs, owners = state
        kind, obj = self.ops[t][pcs[t]]
        new_pcs = pcs[:t] + (pcs[t] + 1,) + pcs[t + 1:]
        if kind == "lock":
            m = self._m[obj]
            owners = owners[:m] + (t,) + owners[m + 1:]
        elif kind == "unlock":
            m = self._m[obj]
            if owners[m] != t:
                raise ProgramError(f"{self.threads[t]} unlocks {obj} it does not hold")
            owners = owners[:m] + (None,) + owners[m + 1:]
        return (new_pcs, owners)

    # -- queries ------------------------------------------------------------
    def held(self, state: State, t: int) -> frozenset:
        return frozenset(m for m, i in self._m.items() if state[1][i] == t)

    def blocked_on(self, state: State) -> dict[str, tuple[str, str]]:
        """thread -> (mutex it waits for, thread that holds it)."""
        out = {}
        for t, name in enumerate(self.threads):
            op = self.next_op(state, t)
            if op and op[0] == "lock":
                owner = state[1][self._m[op[1]]]
                if owner is not None:
                    out[name] = (op[1], self.threads[owner])
        return out

    def races(self, state: State) -> list[tuple]:
        """(var, thread1, op1, thread2, op2) for each conflicting pending pair."""
        found = []
        n = len(self.threads)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = self.next_op(state, i), self.next_op(state, j)
                if not a or not b:
                    continue
                if a[0] not in ("read", "write") or b[0] not in ("read", "write"):
                    continue
                if a[1] != b[1] or "write" not in (a[0], b[0]):
                    continue
                if self.held(state, i) & self.held(state, j):
                    continue
                found.append((a[1], self.threads[i], a[0], self.threads[j], b[0]))
        return found

    def vocab(self) -> dict[str, list[str]]:
        return {"threads": list(self.threads), "objects": list(self.mutexes),
                "vars": list(self.shared)}


# ---------------------------------------------------------------------------
@dataclass
class Exploration:
    model: Model
    initial: State
    order: list  # states in BFS discovery order
    parent: dict  # state -> (prev_state, thread_idx) | None
    edges: dict  # state -> [(thread_idx, next_state)]
    deadlocks: list
    race_states: list
    terminals: list


def explore(program: dict, budget: int = DEFAULT_BUDGET) -> Exploration:
    model = Model(program)
    init = model.initial()
    parent: dict = {init: None}
    edges: dict = {}
    order: list = []
    deadlocks, race_states, terminals = [], [], []
    queue = deque([init])
    while queue:
        s = queue.popleft()
        order.append(s)
        edges[s] = []
        if model.is_terminal(s):
            terminals.append(s)
            continue
        if model.races(s):
            race_states.append(s)
        en = model.enabled(s)
        if not en:
            deadlocks.append(s)
        for t in en:
            s2 = model.step(s, t)
            edges[s].append((t, s2))
            if s2 not in parent:
                parent[s2] = (s, t)
                queue.append(s2)
                if len(parent) > budget:
                    raise StateBudgetExceeded(f"more than {budget} states")
    return Exploration(model, init, order, parent, edges, deadlocks, race_states, terminals)


def label_states(ex: Exploration, bad: set) -> dict:
    """Backward propagation: ALWAYS_SAFE / MAY_FAIL / MUST_FAIL per state.

    safe completion = a path to a terminal state that never visits a bad state.
    """
    preds = defaultdict(list)
    for s, es in ex.edges.items():
        for _, s2 in es:
            preds[s2].append(s)

    safe = {s for s in ex.terminals if s not in bad}
    work = deque(safe)
    while work:
        for p in preds[work.popleft()]:
            if p not in safe and p not in bad:
                safe.add(p)
                work.append(p)

    reach_bad = set(bad)
    work = deque(bad)
    while work:
        for p in preds[work.popleft()]:
            if p not in reach_bad:
                reach_bad.add(p)
                work.append(p)

    labels = {}
    for s in ex.order:
        if s not in reach_bad:
            labels[s] = "ALWAYS_SAFE"
        elif s not in safe:
            labels[s] = "MUST_FAIL"
        else:
            labels[s] = "MAY_FAIL"
    return labels


def _chain(ex: Exploration, target: State) -> list:
    chain = []
    s = target
    while ex.parent[s] is not None:
        prev, t = ex.parent[s]
        chain.append((prev, t, s))
        s = prev
    chain.reverse()
    return chain


def _trace(ex: Exploration, chain: list) -> Trace:
    m = ex.model
    steps = []
    for i, (prev, t, after) in enumerate(chain, start=1):
        kind, obj = m.ops[t][prev[0][t]]
        delta = {}
        if kind == "lock":
            delta = {f"{obj}.owner": m.threads[t]}
        elif kind == "unlock":
            delta = {f"{obj}.owner": None}
        steps.append(Step(i, m.threads[t], kind, obj, delta,
                          sorted(m.blocked_on(after))))
    return Trace("concurrency", steps, m.vocab(), {"search": "tierA_bfs"})


def _critical_point(ex: Exploration, chain: list, labels_safe: set) -> Optional[dict]:
    """Last state on the witness with a safe completion, and the choice that left it."""
    states = [ex.initial] + [c[2] for c in chain]
    idx = None
    for k, s in enumerate(states):
        if s in labels_safe:
            idx = k
    if idx is None or idx >= len(chain):
        return None  # no schedule avoids the failure (e.g. a race in the code itself)
    m = ex.model
    prev, t, _ = chain[idx]
    kind, obj = m.ops[t][prev[0][t]]
    alt = None
    for t2, s2 in ex.edges[prev]:
        if s2 in labels_safe:
            alt = (t2, m.ops[t2][prev[0][t2]])
            break
    cp = {
        "step_index": idx + 1,  # 1-based number of the step that was the bad choice
        "decision_after_step": idx,  # the decision state is the one after this many steps
        "bad_choice": f"{m.threads[t]}:{kind} {obj}",
        "bad_thread": m.threads[t], "bad_op": kind, "bad_object": obj,
        "safe_alternative": None,
    }
    if alt:
        cp["safe_alternative"] = f"{m.threads[alt[0]]}:{alt[1][0]} {alt[1][1]}"
        cp.update(safe_thread=m.threads[alt[0]], safe_op=alt[1][0], safe_object=alt[1][1])
    return cp


def _safe_set(ex: Exploration, bad: set) -> set:
    labels = label_states(ex, bad)
    # safe completion <=> not MUST_FAIL (ALWAYS_SAFE or MAY_FAIL)
    return {s for s, l in labels.items() if l != "MUST_FAIL"}


def analyze_full(program: dict, budget: int = DEFAULT_BUDGET):
    """Return (Verdict, Exploration, state_labels)."""
    ex = explore(program, budget)
    m = ex.model
    if ex.deadlocks:
        label, bad, target = "deadlock", set(ex.deadlocks), ex.deadlocks[0]
    elif ex.race_states:
        label, bad, target = "race", set(ex.race_states), ex.race_states[0]
    else:
        label, bad, target = "safe", set(), None

    labels = label_states(ex, bad)
    witness = cp = None
    detail: dict = {}
    if target is not None:
        chain = _chain(ex, target)
        witness = _trace(ex, chain)
        safe = {s for s, l in labels.items() if l != "MUST_FAIL"}
        cp = _critical_point(ex, chain, safe)
        if label == "deadlock":
            detail = {
                "blocked": {t: mu for t, (mu, _) in m.blocked_on(target).items()},
                "owners": {mu: m.threads[o] for mu, o in zip(m.mutexes, target[1])
                           if o is not None},
            }
        else:
            var, t1, o1, t2, o2 = m.races(target)[0]
            detail = {"variable": var, "threads": [t1, t2],
                      "pending": {t1: f"{o1} {var}", t2: f"{o2} {var}"}}
    else:
        detail = {"terminal_states": len(ex.terminals)}

    verdict = Verdict(
        label=label,
        completeness="complete_for_bound",
        bounds={"loop_unroll": 0, "buffer": 0, "state_budget": budget},
        states_explored=len(ex.order),
        fairness_assumption=None,
        witness=witness,
        critical_point=cp,
        program_hash=program_hash(program),
        detail=detail,
    )
    return verdict, ex, labels


def analyze(program: dict, budget: int = DEFAULT_BUDGET) -> Verdict:
    return analyze_full(program, budget)[0]
