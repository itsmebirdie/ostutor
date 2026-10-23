"""Deterministic parser for the restricted concurrency DSL (Build Plan 8.1).

    mutex a, b            # declarations (comments start with # or //)
    shared x
    thread T1:
        lock a
        repeat 2:         # literal-bound loop; also  for i in range(2):
            read x        # `while <cond>:` is unrolled to the default bound
            write x
        unlock a

    semaphore items = 0   # counting semaphore with an initial value
    cond ready            # condition variable
    thread P:
        forever:          # an endless loop (top level of a thread only)
            down items    # semaphore P ; `up s` is V
            wait ready, m # release m, wait on `ready`, re-acquire m (needs m held)
            signal ready  # or `broadcast ready`

Output is the dict shape of slice/programs.py, plus optional "semaphores"/"conds" keys.
Anything outside the supported subset raises ParseError naming the construct and
the line (student-friendly, never a crash).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from ostutor.common.types import program_hash

DEFAULT_UNROLL = 2
MAX_UNROLL = 20
_IDENT = r"[A-Za-z_][A-Za-z0-9_]*"
_UNSUPPORTED = {
    "acquire": "semaphore calls: use 'down s'", "release": "semaphore calls: use 'up s'",
    "sem": "semaphores: declare with 'semaphore s = 1'", "if": "conditionals",
    "else": "conditionals", "elif": "conditionals", "def": "functions",
    "return": "functions", "print": "I/O", "sleep": "timing",
}


class ParseError(ValueError):
    def __init__(self, line: int, message: str):
        self.line, self.message = line, message
        super().__init__(f"line {line}: {message}" if line else message)


@dataclass
class Parsed:
    program: dict
    loop_unroll: int  # largest loop bound used (0 = no loops); report it to the student


def _lines(text: str):
    for no, raw in enumerate(text.splitlines(), start=1):
        code = re.split(r"#|//", raw, maxsplit=1)[0].rstrip()
        if not code.strip():
            continue
        expanded = code.expandtabs(4)
        yield no, len(expanded) - len(expanded.lstrip()), expanded.strip()


def _names(rest: str, no: int, what: str) -> list[str]:
    names = [n.strip() for n in rest.split(",")]
    if not rest.strip() or any(not re.fullmatch(_IDENT, n) for n in names):
        raise ParseError(no, f"expected a comma-separated list of names after '{what}'")
    return names


def parse(text: str, default_unroll: int = DEFAULT_UNROLL) -> Parsed:
    mutexes: list[str] = []
    shared: list[str] = []
    sems: dict[str, int] = {}
    conds: list[str] = []
    threads: dict[str, list[tuple]] = {}
    max_loop = 0

    items = list(_lines(text))
    i = 0
    while i < len(items):  # ---- top level: declarations and threads
        no, indent, line = items[i]
        word = line.split()[0]
        if indent != 0:
            raise ParseError(no, "unexpected indentation outside a thread")
        taken = set(mutexes) | set(shared) | set(sems) | set(conds)
        if word == "mutex":
            for n in _names(line[len(word):], no, "mutex"):
                if n in taken:
                    raise ParseError(no, f"'{n}' is declared twice")
                mutexes.append(n)
            i += 1
        elif word == "semaphore":
            m = re.fullmatch(rf"semaphore\s+({_IDENT})\s*(?:=\s*(\d+))?", line)
            if not m:
                raise ParseError(no, "write a semaphore as 'semaphore s = 1'")
            if m.group(1) in taken:
                raise ParseError(no, f"'{m.group(1)}' is declared twice")
            sems[m.group(1)] = int(m.group(2) or 0)
            i += 1
        elif word == "cond":
            for n in _names(line[len(word):], no, "cond"):
                if n in taken:
                    raise ParseError(no, f"'{n}' is declared twice")
                conds.append(n)
            i += 1
        elif word == "shared":
            rest = re.sub(r"=\s*[-\d]+", "", line[len(word):])  # "shared x = 0" is fine
            for n in _names(rest, no, "shared"):
                if n in taken:
                    raise ParseError(no, f"'{n}' is declared twice")
                shared.append(n)
            i += 1
        elif word == "thread":
            m = re.fullmatch(rf"thread\s+({_IDENT})\s*:", line)
            if not m:
                raise ParseError(no, "write a thread as 'thread T1:' (with the colon)")
            name = m.group(1)
            if name in threads:
                raise ParseError(no, f"thread '{name}' is defined twice")
            body_start = i + 1
            j = body_start
            while j < len(items) and items[j][1] > 0:
                j += 1
            if j == body_start:
                raise ParseError(no, f"thread '{name}' has no statements (indent its body)")
            ops, pos, loops = _block(items, body_start, j, items[body_start][1],
                                     (mutexes, shared, list(sems), conds), default_unroll,
                                     top=True)
            max_loop = max(max_loop, loops)
            threads[name] = ops
            i = j
        elif word in _UNSUPPORTED:
            raise ParseError(no, f"'{word}' ({_UNSUPPORTED[word]}) is not supported yet; "
                                 "this version handles mutex, semaphore, cond, shared, "
                                 "lock, unlock, down, up, wait, signal, read, write and loops")
        else:
            raise ParseError(no, f"unexpected '{word}': expected 'mutex', 'semaphore', "
                                 "'cond', 'shared' or 'thread NAME:'")
    if not threads:
        raise ParseError(0, "the program has no threads")
    program: dict = {"mutexes": mutexes, "shared": shared, "threads": threads}
    if sems:
        program["semaphores"] = sems
    if conds:
        program["conds"] = conds
    return Parsed(program, max_loop)


def _block(items, start, end, indent, decls, default_unroll, top=False):
    """Parse items[start:end], all at `indent` or deeper. Returns (ops, end, max_loop)."""
    mutexes, shared, sems, conds = decls
    ops: list[tuple] = []
    max_loop = 0
    i = start
    while i < end:
        no, ind, line = items[i]
        if ind != indent:
            raise ParseError(no, "inconsistent indentation")
        word = line.split()[0]

        if re.fullmatch(r"forever\s*:", line):
            if not top:
                raise ParseError(no, "'forever' must be at the top level of a thread")
            j = i + 1
            while j < end and items[j][1] > indent:
                j += 1
            if j == i + 1:
                raise ParseError(no, "'forever' needs an indented body")
            if j < end:
                raise ParseError(items[j][0], "statements after 'forever' are never reached")
            body, _, inner = _block(items, i + 1, j, items[i + 1][1], decls, default_unroll)
            if not body:
                raise ParseError(no, "'forever' body has no operations")
            ops.extend([*body, ("jump", len(ops))])
            max_loop = max(max_loop, inner)
            i = j
            continue

        loop = _loop_header(line, no, default_unroll)
        if loop is not None:
            j = i + 1
            while j < end and items[j][1] > indent:
                j += 1
            if j == i + 1:
                raise ParseError(no, "a loop needs an indented body")
            body, _, inner = _block(items, i + 1, j, items[i + 1][1], decls, default_unroll)
            ops.extend(body * loop)
            max_loop = max(max_loop, loop, inner)
            i = j
            continue

        m = re.fullmatch(rf"(lock|unlock|read|write)\s+({_IDENT})", line)
        if m:
            kind, obj = m.groups()
            if kind in ("lock", "unlock") and obj not in mutexes:
                hint = f" (declared: {', '.join(mutexes)})" if mutexes else " (none declared)"
                raise ParseError(no, f"mutex '{obj}' is not declared{hint}")
            if kind in ("read", "write") and obj not in shared:
                hint = f" (declared: {', '.join(shared)})" if shared else " (none declared)"
                raise ParseError(no, f"shared variable '{obj}' is not declared{hint}")
            ops.append((kind, obj))
            i += 1
            continue
        m = re.fullmatch(rf"(down|up)\s+({_IDENT})", line)
        if m:
            if m.group(2) not in sems:
                hint = f" (declared: {', '.join(sems)})" if sems else " (none declared)"
                raise ParseError(no, f"semaphore '{m.group(2)}' is not declared{hint}")
            ops.append((m.group(1), m.group(2)))
            i += 1
            continue
        m = re.fullmatch(rf"(signal|broadcast)\s+({_IDENT})", line)
        if m:
            if m.group(2) not in conds:
                hint = f" (declared: {', '.join(conds)})" if conds else " (none declared)"
                raise ParseError(no, f"condition '{m.group(2)}' is not declared{hint}")
            ops.append((m.group(1), m.group(2)))
            i += 1
            continue
        m = re.fullmatch(rf"wait\s+({_IDENT})\s*,\s*({_IDENT})", line)
        if m:
            c, mu = m.groups()
            if c not in conds:
                raise ParseError(no, f"condition '{c}' is not declared")
            if mu not in mutexes:
                raise ParseError(no, f"mutex '{mu}' is not declared")
            ops.extend([("cwait", c, mu), ("cresume", c, mu)])
            i += 1
            continue
        if word in ("wait", "down", "up", "signal", "broadcast"):
            hint = {"wait": "'wait cond, mutex'", "down": "'down s'", "up": "'up s'",
                    "signal": "'signal c'", "broadcast": "'broadcast c'"}[word]
            raise ParseError(no, f"write this as {hint}")
        if word in _UNSUPPORTED:
            raise ParseError(no, f"'{word}' ({_UNSUPPORTED[word]}) is not supported yet; "
                                 "use lock, unlock, read, write, down, up, wait, signal or a loop")
        if word in ("lock", "unlock", "read", "write"):
            raise ParseError(no, f"'{word}' needs exactly one name, e.g. '{word} m'")
        raise ParseError(no, f"cannot understand '{line}'")
    return ops, end, max_loop


def _loop_header(line: str, no: int, default_unroll: int):
    """Return the iteration count if `line` opens a loop, else None."""
    m = re.fullmatch(r"repeat\s+(\d+)\s*:", line) or \
        re.fullmatch(rf"for\s+{_IDENT}\s+in\s+range\(\s*(\d+)\s*\)\s*:", line)
    if m:
        n = int(m.group(1))
        if not 1 <= n <= MAX_UNROLL:
            raise ParseError(no, f"loop count {n} must be between 1 and {MAX_UNROLL} "
                                 "(the state space grows quickly)")
        return n
    if re.fullmatch(r"while\b.*:", line):
        return default_unroll
    if re.match(r"(repeat|for|while)\b", line):
        raise ParseError(no, "loops must look like 'repeat 3:' or 'for i in range(3):'")
    return None


# ---------------------------------------------------------------------------
def canonicalise(program: dict) -> dict:
    """Rename threads and every declared object in order of first appearance."""
    tmap = {t: f"T{k}" for k, t in enumerate(program["threads"], 1)}
    kind_of = {"lock": "m", "unlock": "m", "read": "v", "write": "v", "down": "s", "up": "s",
               "signal": "c", "broadcast": "c", "cwait": "c", "cresume": "c"}
    maps: dict[str, dict] = {"m": {}, "v": {}, "s": {}, "c": {}}

    def ren(table: str, name: str) -> str:
        return maps[table].setdefault(name, f"{table}{len(maps[table]) + 1}")

    threads = {}
    for t, ops in program["threads"].items():
        out = []
        for op in ops:
            if op[0] == "jump":
                out.append(tuple(op))
                continue
            new = [op[0], ren(kind_of[op[0]], op[1])]
            if op[0] in ("cwait", "cresume"):
                new.append(ren("m", op[2]))
            out.append(tuple(new))
        threads[tmap[t]] = out
    for obj in program["mutexes"]:
        ren("m", obj)
    for obj in program["shared"]:
        ren("v", obj)
    sems = {ren("s", s): n for s, n in program.get("semaphores", {}).items()}
    for c in program.get("conds", []):
        ren("c", c)
    out_prog: dict = {"mutexes": sorted(maps["m"].values()),
                      "shared": sorted(maps["v"].values()), "threads": threads}
    if sems:
        out_prog["semaphores"] = dict(sorted(sems.items()))
    if maps["c"]:
        out_prog["conds"] = sorted(maps["c"].values())
    return out_prog


def canonical_hash(program: dict) -> str:
    """Invariant under renaming and spacing. Used to detect equivalent programs."""
    return program_hash(canonicalise(program))
