"""Layer 0, steps 4-5: dense + BM25 indexes over the same chunks (Build Plan 8.6).

Embedders are pluggable:
  HashingEmbedder           deterministic, offline, no downloads (tests, demos). It matches
                            on shared words, NOT meaning; do not use it for evaluation claims.
  SentenceTransformerEmbedder  real dense embeddings, for example BAAI/bge-large-en-v1.5
                            (or bge-small as the fall-back). Needs `pip install
                            sentence-transformers`; embed the corpus once (for example on
                            Colab), then load the saved index locally.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Protocol

from ostutor.rag.chunk import Chunk

_TOKEN = re.compile(r"-{0,2}[a-z0-9][a-z0-9_\-+.]*[a-z0-9]|-{0,2}[a-z0-9]", re.I)
STOP = set("a an the of to in on for and or is are was were be by with as at it its this that "
           "from into than then so but not no do does did how what why when which who can will "
           "if you your i we they them he she his her".split())


def tokens(text: str, drop_stop: bool = True) -> list[str]:
    """Lower-cased words; keeps command names and flags such as chmod, ls, -l intact."""
    out = [t.lower().strip(".") for t in _TOKEN.findall(text)]
    return [t for t in out if t and not (drop_stop and t in STOP)]


class Embedder(Protocol):
    name: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class HashingEmbedder:
    name = "hashing-512"

    def __init__(self, dim: int = 512):
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            vec = [0.0] * self.dim
            toks = tokens(text)
            feats = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
            for f in feats:
                h = int(hashlib.md5(f.encode()).hexdigest(), 16)
                vec[h % self.dim] += 1.0 if (h >> 64) % 2 else -1.0
                vec[(h >> 8) % self.dim] += 0.5
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out


class SentenceTransformerEmbedder:
    def __init__(self, model: str = "BAAI/bge-small-en-v1.5"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise RuntimeError("dense embeddings need `pip install sentence-transformers`") from e
        self.name = model
        self._m = SentenceTransformer(model)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [list(map(float, v)) for v in self._m.encode(texts, normalize_embeddings=True)]


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [Counter(d) for d in docs]
        self.len = [len(d) for d in docs]
        self.avg = (sum(self.len) / len(docs)) if docs else 0.0
        df: Counter = Counter()
        for d in self.docs:
            df.update(d.keys())
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: list[str]) -> list[float]:
        out = []
        for d, ln in zip(self.docs, self.len):
            s = 0.0
            for t in query:
                f = d.get(t, 0)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (
                        f + self.k1 * (1 - self.b + self.b * ln / (self.avg or 1)))
            out.append(s)
        return out


class Index:
    def __init__(self, chunks: list[Chunk], embedder: Embedder, vectors=None):
        self.chunks = chunks
        self.embedder = embedder
        self.vectors = vectors if vectors is not None else embedder.embed(
            [c.text for c in chunks])
        self.bm25 = BM25([tokens(c.text) for c in chunks])

    def dense_scores(self, query: str) -> list[float]:
        q = self.embedder.embed([query])[0]
        return [sum(a * b for a, b in zip(q, v)) for v in self.vectors]

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({
            "embedder": self.embedder.name,
            "chunks": [c.__dict__ for c in self.chunks], "vectors": self.vectors}))

    @classmethod
    def load(cls, path: str | Path, embedder: Embedder) -> Index:
        d = json.loads(Path(path).read_text())
        if d["embedder"] != embedder.name:
            raise ValueError(f"index was built with {d['embedder']!r}, not {embedder.name!r}")
        return cls([Chunk(**c) for c in d["chunks"]], embedder, d["vectors"])
