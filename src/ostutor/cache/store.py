"""SQLite cache (Build Plan 8.7): verdicts and checked narrations.

Verdict key  : program hash + bounds. NOTE: the hash is over the program with its
               own names (not the canonical hash) because thread/mutex names appear
               in witnesses and narration; returning another program's names would
               be wrong. canonical_hash() is for detecting equivalent programs.
Narration key: hash of evidence + question type + prompt version + model.
Only narrations that PASSED the checker are stored; template fall-backs never are.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading

from ostutor.common.types import Verdict, verdict_from_dict

_SCHEMA = """
CREATE TABLE IF NOT EXISTS verdicts (key TEXT PRIMARY KEY, body TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS narrations (key TEXT PRIMARY KEY, body TEXT NOT NULL);
"""


def _h(*parts) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()


class Cache:
    def __init__(self, path: str = ":memory:"):
        # Streamlit runs scripts on worker threads, so allow cross-thread use behind a lock
        self.db = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.RLock()
        self.db.executescript(_SCHEMA)
        self.hits = {"verdict": 0, "narration": 0}
        self.misses = {"verdict": 0, "narration": 0}

    # -- verdicts ---------------------------------------------------------
    @staticmethod
    def verdict_key(program_hash: str, bounds: dict) -> str:
        return _h(program_hash, bounds)

    def get_verdict(self, key: str) -> Verdict | None:
        return verdict_from_dict(json.loads(self._get("verdicts", key, "verdict")))\
            if self._peek("verdicts", key) else self._miss("verdict")

    def put_verdict(self, key: str, verdict: Verdict) -> None:
        self._put("verdicts", key, json.dumps(verdict.to_dict()))

    # -- narrations -------------------------------------------------------
    @staticmethod
    def narration_key(evidence: str, question: str, prompt_version: str, model: str) -> str:
        return _h(evidence, question, prompt_version, model)

    def get_narration(self, key: str) -> dict | None:
        return json.loads(self._get("narrations", key, "narration"))\
            if self._peek("narrations", key) else self._miss("narration")

    def put_narration(self, key: str, narration: dict) -> None:
        self._put("narrations", key, json.dumps(narration))

    # -- internals --------------------------------------------------------
    def _peek(self, table: str, key: str):
        with self._lock:
            return self.db.execute(f"SELECT 1 FROM {table} WHERE key=?", (key,)).fetchone()

    def _get(self, table: str, key: str, kind: str) -> str:
        self.hits[kind] += 1
        with self._lock:
            return self.db.execute(f"SELECT body FROM {table} WHERE key=?", (key,)).fetchone()[0]

    def _miss(self, kind: str):
        self.misses[kind] += 1
        return None

    def _put(self, table: str, key: str, body: str) -> None:
        with self._lock:
            self.db.execute(f"INSERT OR REPLACE INTO {table}(key, body) VALUES (?, ?)",
                            (key, body))
            self.db.commit()

    def close(self) -> None:
        self.db.close()
