"""Layer 0, steps 5-6: hybrid retrieval fused with reciprocal rank fusion, and the
"not found in course material" rule (Build Plan 8.6)."""
from __future__ import annotations

from dataclasses import dataclass

from ostutor.rag.chunk import Chunk
from ostutor.rag.index import Index, tokens

RRF_K = 60


@dataclass
class Hit:
    chunk: Chunk
    score: float          # fused reciprocal-rank score
    bm25: float
    dense: float
    ranks: dict


def _ranks(scores: list[float]) -> dict[int, int]:
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    return {i: r for r, i in enumerate(order, start=1)}


class Retriever:
    """`dense_min` and `bm25_min` are the not-found thresholds: a query needs lexical
    evidence (BM25 at least `bm25_min`) or semantic evidence (cosine at least `dense_min`).
    Tune them on a validation query set (eval/rag_eval.py), not by eye."""

    def __init__(self, index: Index, bm25_min: float = 1.0, dense_min: float = 0.25):
        self.index, self.bm25_min, self.dense_min = index, bm25_min, dense_min

    def retrieve(self, query: str, k: int = 4) -> list[Hit]:
        idx = self.index
        if not idx.chunks:
            return []
        q = tokens(query)
        bm = idx.bm25.scores(q) if q else [0.0] * len(idx.chunks)
        de = idx.dense_scores(query)
        rb, rd = _ranks(bm), _ranks(de)
        fused = {i: (1 / (RRF_K + rb[i])) + (1 / (RRF_K + rd[i])) for i in range(len(bm))}
        best = sorted(fused, key=lambda i: -fused[i])[:k]
        hits = [Hit(idx.chunks[i], fused[i], bm[i], de[i], {"bm25": rb[i], "dense": rd[i]})
                for i in best]
        # evidence gate: keep only candidates that clear a lexical or semantic threshold
        # (so an off-topic query yields no hits and the caller says "not found")
        return [h for h in hits if h.bm25 >= self.bm25_min or h.dense >= self.dense_min]
