"""Evaluate RAG retrieval with a REAL embedder (bge) on YOUR corpus and YOUR questions.

    python eval/run_rag_real.py --corpus data/corpus_raw --questions eval/my_questions.jsonl \
        --embedder bge-small --out eval/results/rag_real

Protocol (so the numbers can be defended):
  1. Questions are split deterministically into dev and test by a hash of the question text.
  2. Ranking quality (hit@k, MRR) is measured WITHOUT any threshold, for BM25 only, dense only
     and hybrid (reciprocal rank fusion), so you can say what each part contributes.
  3. The "not found" thresholds are tuned on DEV ONLY (grid over observed score quantiles,
     because bge and the offline hashing embedder have very different score scales).
  4. TEST is evaluated once with the dev-chosen thresholds; 95% bootstrap intervals are given.
Do not iterate on the test split: if you change anything after looking at it, re-split.

Questions file (JSON lines):
  {"question": "...", "answerable": true,  "source": "os_notes.pdf", "page": 12}
  {"question": "...", "answerable": true,  "source": "week3.pptx", "page": [4, 5]}
  {"question": "who won the world cup", "answerable": false}
Optional "chapter" narrows what counts as relevant. Write them from real student questions if
you can; questions invented by the person who built the system are easier than real ones.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

from common import ROOT, environment, table

from ostutor.rag.chunk import MAX_TOKENS, chunk_pages, count_tokens
from ostutor.rag.index import HashingEmbedder, Index, SentenceTransformerEmbedder
from ostutor.rag.ingest import load_documents
from ostutor.rag.retrieve import Retriever

MODELS = {"bge-small": "BAAI/bge-small-en-v1.5", "bge-large": "BAAI/bge-large-en-v1.5"}
MODES = ("bm25", "dense", "hybrid")


# ------------------------------------------------------------------ questions and splits
def load_questions(path: Path) -> list[dict]:
    qs = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            q = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"{path.name} line {n}: invalid JSON ({e.msg})") from e
        if not isinstance(q.get("question"), str) or not q["question"].strip():
            raise ValueError(f"{path.name} line {n}: 'question' is required")
        if not isinstance(q.get("answerable"), bool):
            raise ValueError(f"{path.name} line {n}: 'answerable' must be true or false")
        if q["answerable"] and not q.get("source"):
            raise ValueError(f"{path.name} line {n}: an answerable question needs a 'source'")
        qs.append(q)
    if not qs:
        raise ValueError("no questions found")
    return qs


def split(questions: list[dict], dev_fraction: float, salt: str = "ostutor-rag") -> tuple:
    """Deterministic: the same question always lands in the same split."""
    dev, test = [], []
    for q in questions:
        h = int(hashlib.sha256((salt + q["question"].strip().lower()).encode()).hexdigest(), 16)
        (dev if (h % 10_000) < dev_fraction * 10_000 else test).append(q)
    return dev, test


def relevant(chunk, q: dict) -> bool:
    if not q["answerable"] or chunk.source != q["source"]:
        return False
    page = q.get("page")
    if page is not None and chunk.page not in (page if isinstance(page, list) else [page]):
        return False
    return q.get("chapter") is None or chunk.chapter == q["chapter"]


# ------------------------------------------------------------------ metrics
def ranking_outcomes(retriever: Retriever, questions: list[dict], k: int) -> list[dict]:
    out = []
    for q in questions:
        if not q["answerable"]:
            continue
        hits = retriever.retrieve(q["question"], k)
        rank = next((i for i, h in enumerate(hits, 1) if relevant(h.chunk, q)), None)
        out.append({"q": q, "rank": rank,
                    "got": [f"{h.chunk.source} p.{h.chunk.page}" for h in hits[:3]]})
    return out


def summarise(outcomes: list[dict], k: int) -> dict:
    n = len(outcomes)
    h1 = [o["rank"] == 1 for o in outcomes]
    h3 = [o["rank"] is not None and o["rank"] <= 3 for o in outcomes]
    hk = [o["rank"] is not None for o in outcomes]
    rr = [1 / o["rank"] if o["rank"] else 0.0 for o in outcomes]
    return {"n": n, "hit1": h1, "hit3": h3, "hitk": hk, "mrr": rr, "k": k}


def bootstrap_ci(values: list, resamples: int = 2000, seed: int = 0) -> tuple[float, float]:
    if not values:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples))
    return means[int(0.025 * resamples)], means[int(0.975 * resamples) - 1]


def fmt(values: list) -> str:
    if not values:
        return "n/a"
    lo, hi = bootstrap_ci([float(v) for v in values])
    return f"{sum(values) / len(values):.3f} [{lo:.3f}, {hi:.3f}]"


def mcnemar_p(a: list[bool], b: list[bool]) -> float:
    """Exact two-sided McNemar test on paired per-question outcomes."""
    only_a = sum(x and not y for x, y in zip(a, b))
    only_b = sum(y and not x for x, y in zip(a, b))
    n = only_a + only_b
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(only_a, only_b) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def quantiles(values: list[float], points: int = 12) -> list[float]:
    vs = sorted(values)
    if not vs:
        return [0.0]
    grid = {vs[min(len(vs) - 1, int(i * (len(vs) - 1) / (points - 1)))] for i in range(points)}
    return sorted(grid | {0.0, vs[-1] + 1e-9})  # include "accept everything" and "reject all"


def gate_stats(index: Index, questions: list[dict], bm25_min: float, dense_min: float,
               cache: dict) -> tuple[float, float]:
    """(accept rate on answerable questions, rejection rate on unanswerable ones)."""
    acc = rej = na = nu = 0
    for q in questions:
        top = cache[q["question"]]
        accepted = top[0] >= bm25_min or top[1] >= dense_min
        if q["answerable"]:
            na += 1
            acc += accepted
        else:
            nu += 1
            rej += not accepted
    return (acc / na if na else float("nan"), rej / nu if nu else float("nan"))


def top_scores(retriever: Retriever, questions: list[dict]) -> dict:
    """Best BM25 and best cosine over the whole corpus for each question (gate inputs)."""
    idx, cache = retriever.index, {}
    from ostutor.rag.index import tokens
    for q in questions:
        bm = idx.bm25.scores(tokens(q["question"]))
        de = idx.dense_scores(q["question"])
        cache[q["question"]] = (max(bm, default=0.0), max(de, default=0.0))
    return cache


# ------------------------------------------------------------------ index building
def make_embedder(name: str, model: str | None):
    if name == "hashing":
        return HashingEmbedder()
    return SentenceTransformerEmbedder(model or MODELS.get(name, name))


def build_index(chunks, embedder, cache_path: Path) -> tuple[Index, bool]:
    fingerprint = hashlib.sha256(
        (embedder.name + "|" + "|".join(c.id + c.text for c in chunks)).encode()).hexdigest()
    meta = cache_path.with_suffix(".fingerprint")
    if cache_path.exists() and meta.exists() and meta.read_text() == fingerprint:
        return Index.load(cache_path, embedder), True
    index = Index(chunks, embedder)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    index.save(cache_path)
    meta.write_text(fingerprint)
    return index, False


# ------------------------------------------------------------------ main
def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default=str(ROOT / "data" / "corpus_raw"))
    ap.add_argument("--questions", required=True)
    ap.add_argument("--embedder", default="bge-small",
                    help="bge-small | bge-large | hashing | or any sentence-transformers id")
    ap.add_argument("--model", help="explicit model id (overrides --embedder)")
    ap.add_argument("--out", default=str(ROOT / "eval" / "results" / "rag_real"))
    ap.add_argument("--dev-fraction", type=float, default=0.5)
    ap.add_argument("-k", type=int, default=5)
    a = ap.parse_args(argv)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    questions = load_questions(Path(a.questions))
    dev, test = split(questions, a.dev_fraction)

    embedder = make_embedder(a.embedder, a.model)
    counter = getattr(embedder, "count_tokens", count_tokens)
    pages = load_documents([a.corpus])
    if not pages:
        raise SystemExit(f"no documents found under {a.corpus}")
    chunks = chunk_pages(pages, counter)
    sizes = [counter(c.text) for c in chunks]
    over = [s for s in sizes if s > MAX_TOKENS]
    exact = "the embedder's own tokenizer" if hasattr(embedder, "count_tokens") else \
        "an approximate counter (words x 1.3), NOT the real tokenizer"
    index, reused = build_index(chunks, embedder, out / f"{embedder.name.replace('/', '_')}.index.json")

    # 2. ranking quality without thresholds, all three systems, both splits
    res: dict = {}
    for mode in MODES:
        r = Retriever(index, bm25_min=-1e9, dense_min=-1e9, mode=mode)
        res[mode] = {"dev": summarise(ranking_outcomes(r, dev, a.k), a.k),
                     "test": summarise(ranking_outcomes(r, test, a.k), a.k)}
        res[mode]["test_outcomes"] = ranking_outcomes(r, test, a.k)

    # 3. tune the not-found gate on DEV only
    hyb = Retriever(index, mode="hybrid")
    dev_top = top_scores(hyb, dev)
    best = None
    for bmin in quantiles([t[0] for t in dev_top.values()]):
        for dmin in quantiles([t[1] for t in dev_top.values()]):
            acc, rej = gate_stats(index, dev, bmin, dmin, dev_top)
            if math.isnan(acc) or math.isnan(rej):
                continue
            score = (acc + rej) / 2  # balanced accuracy of the accept/reject decision
            if best is None or score > best[0] + 1e-12:
                best = (score, bmin, dmin, acc, rej)
    if best is None:
        best = (float("nan"), 0.0, 0.0, float("nan"), float("nan"))
    _, bmin, dmin, dev_acc, dev_rej = best

    # 4. TEST, once
    test_top = top_scores(hyb, test)
    t_acc, t_rej = gate_stats(index, test, bmin, dmin, test_top)
    n_ans = sum(q["answerable"] for q in test)
    n_unans = len(test) - n_ans

    warnings = []
    if n_ans < 30:
        warnings.append(f"only {n_ans} answerable TEST questions: intervals are wide; aim for 100+")
    if n_unans < 10:
        warnings.append(f"only {n_unans} unanswerable TEST questions: the not-found rate is "
                        "barely measured; aim for 30+")
    if not hasattr(embedder, "count_tokens") and a.embedder != "hashing":
        warnings.append("the 512-token limit was checked with an approximate counter")
    if a.embedder == "hashing":
        warnings.append("embedder is the offline HASHING embedder: lexical, not semantic. "
                        "This run only checks the plumbing; use bge for a real result")
    if over:
        warnings.append(f"{len(over)} chunk(s) exceed {MAX_TOKENS} tokens")

    rows = []
    for mode in MODES:
        t = res[mode]["test"]
        rows.append([mode, t["n"], fmt(t["hit1"]), fmt(t["hit3"]), fmt(t["hitk"]), fmt(t["mrr"])])
    h, b, d = (res[m]["test"] for m in ("hybrid", "bm25", "dense"))
    p_hb = mcnemar_p(h["hit1"], b["hit1"]) if h["n"] else float("nan")
    p_hd = mcnemar_p(h["hit1"], d["hit1"]) if h["n"] else float("nan")
    fails = [o for o in res["hybrid"]["test_outcomes"] if o["rank"] is None or o["rank"] > 3]
    dev_rows = [[m, res[m]["dev"]["n"], fmt(res[m]["dev"]["hit1"]), fmt(res[m]["dev"]["hit3"]),
                 fmt(res[m]["dev"]["mrr"])] for m in MODES]

    md = f"""# RAG evaluation with a real embedder

*{environment()}. Generated by `eval/run_rag_real.py`.*

- Embedder: **{embedder.name}** ({"index loaded from cache" if reused else "index built now"})
- Corpus: {len(pages)} pages/slides -> {len(chunks)} chunks; token sizes {min(sizes)}-{max(sizes)}
  (limit {MAX_TOKENS}, counted with {exact})
- Questions: {len(questions)} total -> dev {len(dev)} / test {len(test)}
  (answerable/unanswerable on test: {n_ans}/{n_unans})

{"**Warnings**" if warnings else ""}
{chr(10).join("- " + w for w in warnings)}

## Ranking quality on TEST (no threshold; mean [95% bootstrap CI])

{table(["system", "answerable q", "hit@1", "hit@3", f"hit@{a.k}", "MRR"], rows)}

Paired exact McNemar on hit@1: hybrid vs BM25 p={p_hb:.3f}; hybrid vs dense p={p_hd:.3f}
(small p means the two systems really differ on these questions).

## Not-found gate

Thresholds were tuned on **dev only** (balanced accuracy): bm25_min={bmin:.3f},
dense_min={dmin:.3f}. On dev: {dev_acc:.2f} of answerable questions accepted, {dev_rej:.2f} of
unanswerable ones rejected. On **test**: answerable accepted **{t_acc:.3f}**
(false "not found" rate {1 - t_acc:.3f}); unanswerable rejected **{t_rej:.3f}**
(leak rate {1 - t_rej:.3f}).

Use these thresholds in `Retriever(index, bm25_min=..., dense_min=...)`. The shipped defaults
(1.0, 0.25) were chosen for the hashing embedder and are wrong for bge.

## Dev results (for reference only; do not report these as the result)

{table(["system", "answerable q", "hit@1", "hit@3", "MRR"], dev_rows)}

## Test failures for error analysis (hybrid, answer not in the top 3)

{chr(10).join(f"- {o['q']['question']!r}: wanted {o['q'].get('source')} p.{o['q'].get('page')}; got {o['got']}" for o in fails) or "None."}
"""
    summary = {"embedder": embedder.name, "chunks": len(chunks), "dev": len(dev),
               "test": len(test), "thresholds": {"bm25_min": bmin, "dense_min": dmin},
               "test_gate": {"answerable_accepted": t_acc, "unanswerable_rejected": t_rej},
               "systems": {m: {"hit1": sum(res[m]["test"]["hit1"]), "hit3": sum(res[m]["test"]["hit3"]),
                               "n": res[m]["test"]["n"],
                               "mrr": (sum(res[m]["test"]["mrr"]) / res[m]["test"]["n"]
                                       if res[m]["test"]["n"] else None)} for m in MODES},
               "mcnemar_hybrid_vs_bm25": p_hb, "warnings": warnings,
               "headline": f"{embedder.name}: hybrid hit@3 "
                           f"{sum(h['hit3'])}/{h['n']} on test; gate leak {1 - t_rej:.2f}"}
    (out / "rag_real.md").write_text(md + "\n", encoding="utf-8")
    (out / "rag_real.json").write_text(json.dumps(summary, indent=1, default=str), "utf-8")
    return summary


if __name__ == "__main__":
    print(main()["headline"])
