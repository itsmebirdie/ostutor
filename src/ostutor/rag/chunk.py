"""Layer 0, steps 2-3: chunk to 250-400 tokens with 10-15% overlap on paragraph/slide
boundaries, tag each chunk, and deduplicate by normalised-text hash (Build Plan 8.6)."""
from __future__ import annotations

import hashlib
import re
from collections.abc import Callable
from dataclasses import dataclass

from ostutor.rag.ingest import Page

MAX_TOKENS = 512  # the embedder's limit; a test asserts no chunk exceeds it
TARGET_MIN, TARGET_MAX = 250, 400
OVERLAP = 0.12


def count_tokens(text: str) -> int:
    """Approximation (about 1.3 tokens per word). Pass the real tokenizer's counter to
    `chunk_pages` when you embed with bge so the 512 limit is checked exactly."""
    return int(len(re.findall(r"\S+", text)) * 1.3) + 1


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    chapter: str
    page: int
    topic: str

    def cite(self) -> str:
        where = f"{self.source}, p. {self.page}" if self.page else self.source
        return f"{where}" + (f" ({self.chapter})" if self.chapter else "")


def _units(text: str) -> list[str]:
    """Paragraphs; overlong paragraphs are split on sentence boundaries."""
    out = []
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if para:
            out.append(para)
    return out


def _split_long(unit: str, counter: Callable[[str], int], limit: int) -> list[str]:
    if counter(unit) <= limit:
        return [unit]
    sents = re.split(r"(?<=[.!?])\s+", unit)
    parts, cur = [], ""
    for s in sents:
        if cur and counter(cur + " " + s) > limit:
            parts.append(cur)
            cur = s
        else:
            cur = f"{cur} {s}".strip()
    if cur:
        parts.append(cur)
    out = []
    for p in parts:  # a single sentence longer than the limit: hard split on words
        if counter(p) <= limit:
            out.append(p)
        else:
            words = p.split()
            step = max(1, int(limit / 1.4))
            out += [" ".join(words[i:i + step]) for i in range(0, len(words), step)]
    return out


def _overlap_tail(units: list[str], counter: Callable[[str], int]) -> list[str]:
    """The last ~12% of the chunk as a text fragment, so neighbours share a little context
    without ever carrying a whole large unit into the next chunk."""
    budget = int(OVERLAP * TARGET_MAX)  # about 48 tokens
    words = " ".join(units).split()
    take = max(1, int(budget / 1.3))
    tail = " ".join(words[-take:])
    while counter(tail) > budget and take > 1:
        take -= 1
        tail = " ".join(words[-take:])
    return [tail]


def normalise(text: str) -> str:
    return re.sub(r"\W+", " ", text.lower()).strip()


def chunk_pages(pages: list[Page], counter: Callable[[str], int] = count_tokens,
                topic_of: Callable[[Page], str] | None = None) -> list[Chunk]:
    chunks: list[Chunk] = []
    seen: set[str] = set()
    for pg in pages:
        topic = topic_of(pg) if topic_of else (pg.chapter or pg.source)
        units: list[str] = []
        for u in _units(pg.text):
            units += _split_long(u, counter, TARGET_MAX)
        cur: list[str] = []
        n = 0
        for u in units:
            cost = counter(u)
            if cur and n + cost > TARGET_MAX:
                chunks.append(_make(cur, pg, topic, len(chunks), seen))
                keep = _overlap_tail(cur, counter)
                cur, n = keep, sum(counter(x) for x in keep)
            cur.append(u)
            n += cost
        if cur:
            chunks.append(_make(cur, pg, topic, len(chunks), seen))
    return [c for c in chunks if c is not None]


def _make(units: list[str], pg: Page, topic: str, k: int, seen: set):
    text = "\n\n".join(units)
    h = hashlib.sha256(normalise(text).encode()).hexdigest()
    if h in seen:
        return None  # near-identical slides must not fill the top-k slots
    seen.add(h)
    return Chunk(f"c{k:04d}", text, pg.source, pg.chapter, pg.page, topic)
