"""Generalised concurrency model (Build Plan 8.2): mutexes, semaphores, condition
variables, shared reads/writes, and `forever` loops (back-edges).

Operations (a program is plain data, as in slice/programs.py, plus two optional keys
"semaphores": {name: initial} and "conds": [names]):

    ("lock", m)  ("unlock", m)  ("read", x)  ("write", x)
    ("down", s)  ("up", s)                         semaphore P / V
    ("cwait", c, m)    release m, join c's FIFO wait queue
    ("cresume", c, m)  enabled once woken AND m is free; re-acquires m
    ("signal", c)  ("broadcast", c)
    ("jump", i)        thread-local back-edge, fused into the preceding operation

State = (pcs, owners, sems, waits, lost): program counters; mutex owner (-1 = free);
semaphore counts; per-condition FIFO queues of waiting threads; and a bitmask of
conditions on which a signal was LOST (issued while nobody waited).
"""
from __future__ import annotations

from ostutor.slice.explorer import ProgramError

State = tuple
VALID = {"lock", "unlock", "read", "write", "down", "up", "cwait", "cresume",
         "signal", "broadcast", "jump"}
ARITY = {"lock": 2, "unlock": 2, "read": 2, "write": 2, "down": 2, "up": 2, "cwait": 3,
         "cresume": 3, "signal": 2, "broadcast": 2, "jump": 2}
LEGACY_OPS = {"lock", "unlock", "read", "write"}


def is_legacy(program: dict) -> bool:
    """True when the slice explorer can handle the program (the stable demo path)."""
    return (not program.get("semaphores") and not program.get("conds")
            and all(op[0] in LEGACY_OPS for ops in program["threads"].values() for op in ops))


class GModel:
    def __init__(self, program: dict):
        self.program = program
        self.threads: list[str] = sorted(program["threads"])
        self.mutexes: list[str] = list(program["mutexes"])
        self.shared: list[str] = list(program["shared"])
        self.sem_names: list[str] = list(program.get("semaphores", {}))
        self.sem_init = tuple(int(program["semaphores"][s]) for s in self.sem_names) \
            if self.sem_names else ()
        self.conds: list[str] = list(program.get("conds", []))
        self._m = {x: i for i, x in enumerate(self.mutexes)}
        self._s = {x: i for i, x in enumerate(self.sem_names)}
        self._c = {x: i for i, x in enumerate(self.conds)}
        self.ops = [[tuple(o) for o in program["threads"][t]] for t in self.threads]
        self._validate()
        self.nxt = [self._flow(ops) for ops in self.ops]
        self.cyclic = any(o[0] == "jump" for ops in self.ops for o in ops)

    # -- validation and control flow -------------------------------------------
    def _validate(self) -> None:
        names = self.mutexes + self.shared + self.sem_names + self.conds
        if len(set(names)) != len(names):
            raise ProgramError("a name is declared twice")
        for t, ops in zip(self.threads, self.ops):
            for op in ops:
                if op[0] not in VALID or len(op) != ARITY[op[0]]:
                    raise ProgramError(f"{t}: bad operation {op!r}")
                kind = op[0]
                for obj, table, what in self._refs(op):
                    if obj not in table:
                        raise ProgramError(f"{t}: unknown {what} {obj!r}")
                if kind == "jump" and not (0 <= op[1] < len(ops)):
                    raise ProgramError(f"{t}: jump target out of range")

    def _refs(self, op):
        k = op[0]
        if k in ("lock", "unlock"):
            return [(op[1], self._m, "mutex")]
        if k in ("read", "write"):
            return [(op[1], set(self.shared), "shared variable")]
        if k in ("down", "up"):
            return [(op[1], self._s, "semaphore")]
        if k in ("cwait", "cresume"):
            return [(op[1], self._c, "condition"), (op[2], self._m, "mutex")]
        if k in ("signal", "broadcast"):
            return [(op[1], self._c, "condition")]
        return []

    @staticmethod
    def _flow(ops: list[tuple]) -> list[int]:
        """nxt[i] = index of the next *visible* operation after executing ops[i]."""
        def resolve(j: int) -> int:
            seen = set()
            while j < len(ops) and ops[j][0] == "jump":
                if j in seen:
                    raise ProgramError("a loop with no visible operation")
                seen.add(j)
                j = ops[j][1]
            return j
        return [resolve(i + 1) for i in range(len(ops))] + [len(ops)]

    # -- state machine -----------------------------------------------------------
    def initial(self) -> State:
        pcs = tuple(self._resolve0(t) for t in range(len(self.threads)))
        return (pcs, (-1,) * len(self.mutexes), self.sem_init,
                ((),) * len(self.conds), 0)

    def _resolve0(self, t: int) -> int:
        ops = self.ops[t]
        j, seen = 0, set()
        while j < len(ops) and ops[j][0] == "jump":
            if j in seen:
                raise ProgramError("a loop with no visible operation")
            seen.add(j)
            j = ops[j][1]
        return j

    def next_op(self, state: State, t: int):
        pc = state[0][t]
        return self.ops[t][pc] if pc < len(self.ops[t]) else None

    def is_terminal(self, state: State) -> bool:
        return all(pc >= len(o) for pc, o in zip(state[0], self.ops))

    def can_run(self, state: State, t: int) -> bool:
        op = self.next_op(state, t)
        if op is None:
            return False
        k = op[0]
        _, owners, sems, waits, _ = state
        if k == "lock":
            return owners[self._m[op[1]]] == -1
        if k == "down":
            return sems[self._s[op[1]]] > 0
        if k == "cresume":
            return t not in waits[self._c[op[1]]] and owners[self._m[op[2]]] == -1
        return True

    def enabled(self, state: State) -> list[int]:
        return [t for t in range(len(self.threads)) if self.can_run(state, t)]

    def step(self, state: State, t: int) -> State:
        pcs, owners, sems, waits, lost = state
        op = self.ops[t][pcs[t]]
        k = op[0]
        pcs = pcs[:t] + (self.nxt[t][pcs[t]],) + pcs[t + 1:]
        if k == "lock":
            i = self._m[op[1]]
            owners = owners[:i] + (t,) + owners[i + 1:]
        elif k in ("unlock", "cwait"):
            m = op[1] if k == "unlock" else op[2]
            i = self._m[m]
            if owners[i] != t:
                raise ProgramError(f"{self.threads[t]} releases {m} it does not hold")
            owners = owners[:i] + (-1,) + owners[i + 1:]
            if k == "cwait":
                c = self._c[op[1]]
                waits = waits[:c] + (waits[c] + (t,),) + waits[c + 1:]
        elif k == "cresume":
            i = self._m[op[2]]
            owners = owners[:i] + (t,) + owners[i + 1:]
        elif k == "down":
            i = self._s[op[1]]
            sems = sems[:i] + (sems[i] - 1,) + sems[i + 1:]
        elif k == "up":
            i = self._s[op[1]]
            sems = sems[:i] + (sems[i] + 1,) + sems[i + 1:]
        elif k in ("signal", "broadcast"):
            c = self._c[op[1]]
            if not waits[c]:
                lost |= 1 << c  # nobody was waiting: the signal evaporates
            elif k == "signal":
                waits = waits[:c] + (waits[c][1:],) + waits[c + 1:]
            else:
                waits = waits[:c] + ((),) + waits[c + 1:]
        return (pcs, owners, sems, waits, lost)

    # -- queries ---------------------------------------------------------------
    def held(self, state: State, t: int) -> frozenset:
        return frozenset(m for m, i in self._m.items() if state[1][i] == t)

    def races(self, state: State) -> list[tuple]:
        """(var, t1, op1, t2, op2): conflicting next accesses with no common mutex held."""
        out = []
        n = len(self.threads)
        for a in range(n):
            oa = self.next_op(state, a)
            if oa is None or oa[0] not in ("read", "write"):
                continue
            for b in range(a + 1, n):
                ob = self.next_op(state, b)
                if (ob is None or ob[0] not in ("read", "write") or ob[1] != oa[1]
                        or "write" not in (oa[0], ob[0])):
                    continue
                if not (self.held(state, a) & self.held(state, b)):
                    out.append((oa[1], self.threads[a], oa[0], self.threads[b], ob[0]))
        return out

    def waits_for(self, state: State) -> dict[str, tuple]:
        """Why each unfinished, non-runnable thread is stuck: (kind, object, holder|None)."""
        out = {}
        for t, name in enumerate(self.threads):
            op = self.next_op(state, t)
            if op is None or self.can_run(state, t):
                continue
            k = op[0]
            if k == "lock":
                out[name] = ("lock", op[1], self.threads[state[1][self._m[op[1]]]])
            elif k == "down":
                out[name] = ("semaphore", op[1], None)
            elif k == "cresume":
                if t in state[3][self._c[op[1]]]:
                    out[name] = ("signal", op[1], None)
                else:
                    holder = state[1][self._m[op[2]]]
                    out[name] = ("lock", op[2], self.threads[holder])
        return out

    def lost_waiters(self, state: State) -> list[tuple[str, str]]:
        """(thread, cond) pairs waiting on a condition whose signal was lost earlier."""
        out = []
        for c, cname in enumerate(self.conds):
            if state[4] >> c & 1:
                out += [(self.threads[t], cname) for t in state[3][c]]
        return out

    def delta(self, before: State, after: State, t: int) -> dict:
        d: dict = {}
        for i, m in enumerate(self.mutexes):
            if before[1][i] != after[1][i]:
                d[f"{m}.owner"] = self.threads[after[1][i]] if after[1][i] >= 0 else None
        for i, s in enumerate(self.sem_names):
            if before[2][i] != after[2][i]:
                d[f"{s}.count"] = after[2][i]
        for i, c in enumerate(self.conds):
            if before[3][i] != after[3][i]:
                d[f"{c}.waiting"] = [self.threads[x] for x in after[3][i]]
        return d
