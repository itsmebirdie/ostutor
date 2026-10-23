"""Layer 0, step 6: answer ONLY from retrieved chunks, with citations (Build Plan 8.6).

Without an LLM the answer is extractive (the best-matching sentences, cited). With an LLM
the reply must be JSON {"answer", "citations"}; citations must be retrieved chunk numbers
and every number in the answer must occur in the cited text, otherwise we fall back to the
extractive answer. Below the evidence threshold the reply is "not found in course material".
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from ostutor.rag.index import tokens
from ostutor.rag.retrieve import Hit, Retriever

NOT_FOUND = "Not found in course material."

SYSTEM = ("You answer Operating Systems questions using ONLY the numbered course excerpts "
          "provided. If they do not contain the answer, reply with an empty answer. Cite the "
          "excerpts you used by number. Never add facts or numbers that are not in them. "
          'Reply with JSON only: {"answer": "<2-4 sentences>", "citations": [<numbers>]}')


@dataclass
class Answer:
    text: str
    found: bool
    source: str  # "extractive" | "llm" | "not_found"
    citations: list[str] = field(default_factory=list)
    hits: list[Hit] = field(default_factory=list)
    note: str = ""


def _sentences(text: str) -> list[str]:
    body = "\n".join(ln for ln in text.splitlines() if not ln.lstrip().startswith("#"))
    flat = re.sub(r"\s*\n\s*", " ", body)
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z`])", flat) if len(s.strip()) > 20]


def _extractive(query: str, hits: list[Hit], idf: dict, max_sentences: int = 3) -> Answer:
    q = set(tokens(query))
    scored = []
    for rank, h in enumerate(hits[:3], start=1):
        for s in _sentences(h.chunk.text):
            overlap = sum(idf.get(t, 1.0) for t in q & set(tokens(s)))  # rare terms count more
            if overlap:
                scored.append((overlap, -rank, s, rank))
    if not scored:
        first = _sentences(hits[0].chunk.text)[:1]
        scored = [(0, -1, first[0] if first else hits[0].chunk.text[:200], 1)]
    scored.sort(key=lambda t: (-t[0], -t[1]))
    top = scored[:max_sentences]
    used = sorted({r for *_, r in top})
    body = " ".join(f"{s} [{r}]" for _, _, s, r in top)
    return Answer(body, True, "extractive", [hits[r - 1].chunk.cite() for r in used], hits)


def answer(query: str, retriever: Retriever, llm=None, k: int = 4) -> Answer:
    hits = retriever.retrieve(query, k)
    if not hits:
        return Answer(NOT_FOUND, False, "not_found", [], [])
    base = _extractive(query, hits, retriever.index.bm25.idf)
    if llm is None:
        return base
    excerpts = "\n\n".join(f"[{i}] ({h.chunk.cite()})\n{h.chunk.text}"
                           for i, h in enumerate(hits, start=1))
    try:
        raw = llm.complete(SYSTEM, f"EXCERPTS\n{excerpts}\n\nQUESTION\n{query}")
        data = json.loads(raw[raw.find("{"):raw.rfind("}") + 1])
        text, cites = str(data.get("answer", "")).strip(), data.get("citations", [])
    except Exception as e:  # network, JSON, anything: degrade to the extractive answer
        base.note = f"LLM unavailable or invalid ({type(e).__name__}); showing extracted text"
        return base
    if not text:
        base.note = "the model found no answer in the excerpts; showing the closest text"
        return base
    if (not isinstance(cites, list) or not cites
            or not all(isinstance(c, int) and 1 <= c <= len(hits) for c in cites)):
        base.note = "the model's citations were invalid; showing extracted text"
        return base
    cited_text = " ".join(hits[c - 1].chunk.text for c in cites)
    cited_numbers = set(re.findall(r"\d+", cited_text))
    if any(n not in cited_numbers for n in re.findall(r"\d+", text)):
        base.note = "the answer contained a number not in the cited text; showing extracted text"
        return base
    return Answer(text, True, "llm", [hits[c - 1].chunk.cite() for c in sorted(set(cites))], hits)
