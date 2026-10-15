"""Deterministic parser for the restricted concurrency DSL (Build Plan 8.1).

    mutex a, b            # declarations (comments start with # or //)
    shared x
    thread T1:
        lock a
        repeat 2:         # literal-bound loop; also  for i in range(2):
            read x        # `while <cond>:` is unrolled to the default bound
            write x
        unlock a

Output is exactly the dict shape of slice/programs.py, so the explorer is unchanged.
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
    "wait": "condition variables", "signal": "condition variables",
    "broadcast": "condition variables", "cond": "condition variables",
    "semaphore": "semaphores", "sem": "semaphores", "acquire": "semaphores",
    "release": "semaphores", "if": "conditionals", "else": "conditionals",
    "elif": "conditionals", "def": "functions", "return": "functions",
    "print": "I/O", "sleep": "timing",
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
    threads: dict[str, list[tuple]] = {}
    max_loop = 0

    items = list(_lines(text))
    i = 0
    while i < len(items):  # ---- top level: declarations and threads
        no, indent, line = items[i]
        word = line.split()[0]
        if indent != 0:
            raise ParseError(no, "unexpected indentation outside a thread")
        if word == "mutex":
            for n in _names(line[len(word):], no, "mutex"):
                if n in mutexes or n in shared:
                    raise ParseError(no, f"'{n}' is declared twice")
                mutexes.append(n)
            i += 1
        elif word == "shared":
            rest = re.sub(r"=\s*[-\d]+", "", line[len(word):])  # "shared x = 0" is fine
            for n in _names(rest, no, "shared"):
                if n in mutexes or n in shared:
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
                                     mutexes, shared, default_unroll)
            max_loop = max(max_loop, loops)
            threads[name] = ops
            i = j
        elif word in _UNSUPPORTED:
            raise ParseError(no, f"'{word}' ({_UNSUPPORTED[word]}) is not supported yet; "
                                 "this version handles mutex, shared, lock, unlock, "
                                 "read, write and loops")
        else:
            raise ParseError(no, f"unexpected '{word}': expected 'mutex', 'shared' "
                                 "or 'thread NAME:'")
    if not threads:
        raise ParseError(0, "the program has no threads")
    return Parsed({"mutexes": mutexes, "shared": shared, "threads": threads}, max_loop)


def _block(items, start, end, indent, mutexes, shared, default_unroll):
    """Parse items[start:end], all at `indent` or deeper. Returns (ops, end, max_loop)."""
    ops: list[tuple] = []
    max_loop = 0
    i = start
    while i < end:
        no, ind, line = items[i]
        if ind != indent:
            raise ParseError(no, "inconsistent indentation")
        word = line.split()[0]

        loop = _loop_header(line, no, default_unroll)
        if loop is not None:
            j = i + 1
            while j < end and items[j][1] > indent:
                j += 1
            if j == i + 1:
                raise ParseError(no, "a loop needs an indented body")
            body, _, inner = _block(items, i + 1, j, items[i + 1][1], mutexes, shared,
                                    default_unroll)
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
        if word in _UNSUPPORTED:
            raise ParseError(no, f"'{word}' ({_UNSUPPORTED[word]}) is not supported yet; "
                                 "use lock, unlock, read, write or a loop")
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
    """Rename threads, mutexes and variables in order of first appearance."""
    tmap = {t: f"T{k}" for k, t in enumerate(program["threads"], 1)}
    mmap: dict[str, str] = {}
    vmap: dict[str, str] = {}
    threads = {}
    for t, ops in program["threads"].items():
        out = []
        for kind, obj in ops:
            table, prefix = (mmap, "m") if kind in ("lock", "unlock") else (vmap, "v")
            table.setdefault(obj, f"{prefix}{len(table) + 1}")
            out.append((kind, table[obj]))
        threads[tmap[t]] = out
    for obj in program["mutexes"]:
        mmap.setdefault(obj, f"m{len(mmap) + 1}")
    for obj in program["shared"]:
        vmap.setdefault(obj, f"v{len(vmap) + 1}")
    return {"mutexes": sorted(mmap.values()), "shared": sorted(vmap.values()),
            "threads": threads}


def canonical_hash(program: dict) -> str:
    """Invariant under renaming and spacing. Used to detect equivalent programs."""
    return program_hash(canonicalise(program))
