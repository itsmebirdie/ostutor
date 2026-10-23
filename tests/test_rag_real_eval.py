"""Tests for the real-embedder RAG evaluation harness and the bge embedder adapter.

The bge model itself cannot be downloaded in CI, so the adapter is tested against a fake
`sentence_transformers` module that records how it is called."""
import json
import sys
import types
from pathlib import Path

import pytest

EVAL = Path(__file__).parent.parent / "eval"
sys.path.insert(0, str(EVAL))
import run_rag_real as rr  # noqa: E402

from ostutor.rag.chunk import Chunk  # noqa: E402
from ostutor.rag.index import BGE_QUERY_PREFIX, HashingEmbedder, Index  # noqa: E402
from ostutor.rag.retrieve import Retriever  # noqa: E402

ROOT = EVAL.parent
CORPUS = ROOT / "data" / "corpus_sample"
QUESTIONS = EVAL / "rag_questions.example.jsonl"


# ------------------------------------------------------------------ bge adapter (faked library)
@pytest.fixture()
def fake_st(monkeypatch):
    calls = {"encoded": [], "loaded": []}
    inner = HashingEmbedder(dim=16)

    class Tok:
        def encode(self, text, add_special_tokens=True):
            return list(range(len(text.split()) + (2 if add_special_tokens else 0)))

    class Model:
        def __init__(self, name):
            calls["loaded"].append(name)
            self.tokenizer = Tok()

        def encode(self, texts, normalize_embeddings=False, batch_size=None):
            calls["encoded"].append((list(texts), normalize_embeddings, batch_size))
            return inner.embed(list(texts))

    mod = types.ModuleType("sentence_transformers")
    mod.SentenceTransformer = Model  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", mod)
    return calls


def test_bge_queries_get_the_instruction_prefix_but_passages_do_not(fake_st):
    from ostutor.rag.index import SentenceTransformerEmbedder
    emb = SentenceTransformerEmbedder("BAAI/bge-small-en-v1.5")
    assert emb.query_prefix == BGE_QUERY_PREFIX and fake_st["loaded"] == ["BAAI/bge-small-en-v1.5"]
    emb.embed(["a passage about chmod"])
    emb.embed_query("how does chmod work")
    passage_call, query_call = fake_st["encoded"]
    assert passage_call[0] == ["a passage about chmod"] and passage_call[1] is True
    assert query_call[0] == [BGE_QUERY_PREFIX + "how does chmod work"]  # prefix on the query only


def test_non_bge_models_get_no_prefix_and_a_custom_prefix_is_respected(fake_st):
    from ostutor.rag.index import SentenceTransformerEmbedder
    assert SentenceTransformerEmbedder("some/other-model").query_prefix == ""
    assert SentenceTransformerEmbedder("some/other-model", query_prefix="query: ").query_prefix == "query: "


def test_index_uses_embed_query_when_the_embedder_has_one(fake_st):
    from ostutor.rag.index import SentenceTransformerEmbedder
    emb = SentenceTransformerEmbedder("BAAI/bge-small-en-v1.5")
    idx = Index([Chunk("c0", "chmod changes permission bits", "n.md", "C", 1, "t")], emb)
    fake_st["encoded"].clear()
    idx.dense_scores("what is chmod")
    assert fake_st["encoded"][0][0][0].startswith(BGE_QUERY_PREFIX)


def test_token_limit_uses_the_models_own_tokenizer(fake_st):
    from ostutor.rag.index import SentenceTransformerEmbedder
    emb = SentenceTransformerEmbedder("BAAI/bge-small-en-v1.5")
    assert emb.count_tokens("one two three") == 5  # 3 words + the two special tokens


def test_missing_sentence_transformers_gives_an_actionable_error(monkeypatch):
    from ostutor.rag.index import SentenceTransformerEmbedder
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(RuntimeError, match="pip install sentence-transformers"):
        SentenceTransformerEmbedder()


# ------------------------------------------------------------------ retrieval ablations
def test_ablation_modes_rank_with_only_one_retriever():
    chunks = [Chunk("c0", "chmod changes file permission bits", "n.md", "A", 1, "t"),
              Chunk("c1", "pipes connect commands together", "n.md", "B", 2, "t")]
    idx = Index(chunks, HashingEmbedder())
    for mode in ("bm25", "dense", "hybrid"):
        hits = Retriever(idx, bm25_min=-1, dense_min=-1, mode=mode).retrieve("chmod permission", 2)
        assert hits[0].chunk.id == "c0"
    only_bm25 = Retriever(idx, bm25_min=-1, dense_min=-1, mode="bm25").retrieve("chmod", 2)
    assert only_bm25[0].score == pytest.approx(1 / (60 + 1))  # no dense contribution
    with pytest.raises(ValueError):
        Retriever(idx, mode="magic")


# ------------------------------------------------------------------ harness pieces
def test_split_is_deterministic_disjoint_and_roughly_proportional():
    qs = rr.load_questions(QUESTIONS)
    dev1, test1 = rr.split(qs, 0.5)
    dev2, test2 = rr.split(qs, 0.5)
    assert dev1 == dev2 and test1 == test2
    assert len(dev1) + len(test1) == len(qs)
    assert not {q["question"] for q in dev1} & {q["question"] for q in test1}
    assert 0.25 < len(dev1) / len(qs) < 0.75
    assert rr.split(qs, 0.5, salt="other")[0] != dev1  # a different salt re-splits


@pytest.mark.parametrize("line,msg", [
    ("not json", "invalid JSON"), ('{"answerable": true, "source": "a"}', "'question' is required"),
    ('{"question": "q"}', "'answerable' must be"),
    ('{"question": "q", "answerable": true}', "needs a 'source'")])
def test_question_file_validation(tmp_path, line, msg):
    f = tmp_path / "q.jsonl"
    f.write_text(line + "\n")
    with pytest.raises(ValueError, match=msg):
        rr.load_questions(f)


def test_relevance_rules_for_source_page_lists_and_chapter():
    c = Chunk("c", "t", "a.pdf", "Intro", 4, "x")
    assert rr.relevant(c, {"answerable": True, "source": "a.pdf"})
    assert rr.relevant(c, {"answerable": True, "source": "a.pdf", "page": [3, 4]})
    assert not rr.relevant(c, {"answerable": True, "source": "a.pdf", "page": 5})
    assert not rr.relevant(c, {"answerable": True, "source": "b.pdf"})
    assert not rr.relevant(c, {"answerable": True, "source": "a.pdf", "chapter": "Other"})
    assert not rr.relevant(c, {"answerable": False})


def test_bootstrap_ci_and_mcnemar():
    lo, hi = rr.bootstrap_ci([1.0] * 30 + [0.0] * 10)
    assert lo < 0.75 < hi and 0 <= lo <= hi <= 1
    assert rr.bootstrap_ci([1.0] * 20) == (1.0, 1.0)
    assert all(map(lambda x: x != x, rr.bootstrap_ci([])))  # NaN for no data
    assert rr.mcnemar_p([True] * 10, [True] * 10) == 1.0
    assert rr.mcnemar_p([True] * 12 + [False] * 2, [False] * 12 + [True] * 2) < 0.05
    assert rr.mcnemar_p([True, False, True, False], [False, True, False, True]) == 1.0


def test_quantile_grid_includes_accept_everything_and_reject_everything():
    grid = rr.quantiles([0.2, 0.5, 0.9])
    assert grid[0] == 0.0 and grid[-1] > 0.9 and grid == sorted(grid)


# ------------------------------------------------------------------ the whole script
def run(tmp_path, *extra):
    return rr.main(["--corpus", str(CORPUS), "--questions", str(QUESTIONS), "--embedder",
                    "hashing", "--out", str(tmp_path), *extra])


def test_end_to_end_dry_run_writes_report_and_labels_the_hashing_embedder(tmp_path):
    s = run(tmp_path)
    md = (tmp_path / "rag_real.md").read_text()
    assert s["dev"] + s["test"] == 32 and set(s["systems"]) == {"bm25", "dense", "hybrid"}
    assert "HASHING embedder" in md and "only checks the plumbing" in md
    assert "aim for 100+" in md  # the small-sample warning is raised
    assert "Thresholds were tuned on **dev only**" in md
    assert json.loads((tmp_path / "rag_real.json").read_text())["embedder"] == "hashing-512"


def test_thresholds_are_chosen_on_dev_and_test_is_scored_with_them(tmp_path):
    s = run(tmp_path)
    t = s["thresholds"]
    qs = rr.load_questions(QUESTIONS)
    dev, test = rr.split(qs, 0.5)
    from ostutor.rag.chunk import chunk_pages
    from ostutor.rag.ingest import load_documents
    idx = Index(chunk_pages(load_documents([CORPUS])), HashingEmbedder())
    cache = rr.top_scores(Retriever(idx), test)
    acc, rej = rr.gate_stats(idx, test, t["bm25_min"], t["dense_min"], cache)
    assert s["test_gate"] == {"answerable_accepted": acc, "unanswerable_rejected": rej}


def test_second_run_reuses_the_cached_index_and_a_changed_corpus_invalidates_it(tmp_path):
    run(tmp_path)
    assert "index built now" in (tmp_path / "rag_real.md").read_text()
    run(tmp_path)
    assert "index loaded from cache" in (tmp_path / "rag_real.md").read_text()
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "n.md").write_text("# New\n\nSomething entirely different about schedulers.\n")
    rr.main(["--corpus", str(corpus), "--questions", str(QUESTIONS), "--embedder", "hashing",
             "--out", str(tmp_path)])
    assert "index built now" in (tmp_path / "rag_real.md").read_text()


def test_empty_corpus_is_a_clear_error(tmp_path):
    with pytest.raises(SystemExit, match="no documents found"):
        rr.main(["--corpus", str(tmp_path / "nothing"), "--questions", str(QUESTIONS),
                 "--embedder", "hashing", "--out", str(tmp_path / "o")])


def test_bge_path_runs_through_the_harness_with_the_fake_library(tmp_path, fake_st):
    s = rr.main(["--corpus", str(CORPUS), "--questions", str(QUESTIONS), "--embedder", "bge-small",
                 "--out", str(tmp_path)])
    assert s["embedder"] == "BAAI/bge-small-en-v1.5" and fake_st["loaded"] == ["BAAI/bge-small-en-v1.5"]
    md = (tmp_path / "rag_real.md").read_text()
    assert "the embedder's own tokenizer" in md and "HASHING embedder" not in md
