"""The three hard-coded programs for the review slice (plain Python data).

The future DSL parser must produce exactly this shape.
"""

from typing import Any

PROGRAMS: dict[str, dict[str, Any]] = {
    "deadlock_two_locks": {
        "mutexes": ["a", "b"],
        "shared": [],
        "threads": {
            "T1": [("lock", "a"), ("lock", "b"), ("unlock", "b"), ("unlock", "a")],
            "T2": [("lock", "b"), ("lock", "a"), ("unlock", "a"), ("unlock", "b")],
        },
    },
    "race_no_lock": {
        "mutexes": [],
        "shared": ["x"],
        "threads": {
            "T1": [("read", "x"), ("write", "x")],
            "T2": [("read", "x"), ("write", "x")],
        },
    },
    "safe_with_lock": {
        "mutexes": ["m"],
        "shared": ["x"],
        "threads": {
            "T1": [("lock", "m"), ("read", "x"), ("write", "x"), ("unlock", "m")],
            "T2": [("lock", "m"), ("read", "x"), ("write", "x"), ("unlock", "m")],
        },
    },
}
