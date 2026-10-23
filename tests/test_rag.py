import json
import re
import struct
from pathlib import Path

import pytest

from ostutor.rag.answer import NOT_FOUND, answer
from ostutor.rag.chunk import MAX_TOKENS, Chunk, chunk_pages, count_tokens, normalise
from ostutor.rag.index import BM25, HashingEmbedder, Index, tokens
from ostutor.rag.ingest import Page, load_documents, strip_boilerplate
from ostutor.rag.retrieve import Retriever

CORPUS = Path(__file__).parent.parent / "data" / "corpus_sample"


@pytest.fixture(scope="module")
def retriever():
    chunks = chunk_pages(load_documents([CORPUS]))
    return Retriever(Index(chunks, HashingEmbedder()))


# ------------------------------------------------------------------ ingestion
def test_markdown_sections_become_pages_with_chapters():
    pages = load_documents([CORPUS])
    assert len(pages) == 6 and pages[0].chapter == "File permissions and chmod"
    assert all(p.source == "unix_basics.md" and p.page >= 1 for p in pages)


def test_page_numbers_headers_and_footers_are_stripped():
    pages = [Page("b.pdf", i, f"Operating Systems Notes\nBody text number {i}.\nPage {i}\n{i}")
             for i in range(1, 6)]
    out = strip_boilerplate(pages)
    assert all("Operating Systems Notes" not in p.text for p in out)
    assert all(not re.search(r"^\s*(Page )?\d+\s*$", p.text, re.M) for p in out)
    assert out[2].text == "Body text number 3."


def make_pdf(path, lines):
    """A minimal but valid one-page PDF containing the given text lines."""
    stream = "BT /F1 12 Tf 50 750 Td 14 TL " + " ".join(
        f"({ln}) Tj T*" for ln in lines) + " ET"
    objs = ["<< /Type /Catalog /Pages 2 0 R >>",
            "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
            "/Resources << /Font << /F1 5 0 R >> >> >>",
            f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream",
            "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    out, offsets = b"%PDF-1.4\n", []
    for i, o in enumerate(objs, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n{o}\nendobj\n".encode()
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF".encode()
    Path(path).write_bytes(out)
    assert struct  # (kept to show no binary packing is needed)


def test_pdf_ingestion_extracts_text_with_page_metadata(tmp_path):
    pytest.importorskip("pypdf")
    make_pdf(tmp_path / "notes.pdf", ["The scheduler picks the next process.", "Quantum is a time slice."])
    pages = load_documents([tmp_path])
    assert len(pages) == 1 and pages[0].page == 1 and pages[0].source == "notes.pdf"
    assert "scheduler picks the next process" in pages[0].text


def test_pptx_ingestion_keeps_slide_numbers_and_titles(tmp_path):
    pptx = pytest.importorskip("pptx")
    prs = pptx.Presentation()
    for title, body in [("Deadlock", "Circular wait"), ("Paging", "Page tables map pages")]:
        slide = prs.slides.add_slide(prs.slide_layouts[1])
        slide.shapes.title.text = title
        slide.placeholders[1].text = body
    prs.save(str(tmp_path / "week3.pptx"))
    pages = load_documents([tmp_path])
    assert [(p.page, p.chapter) for p in pages] == [(1, "Deadlock"), (2, "Paging")]
    assert "Page tables map pages" in pages[1].text


# ------------------------------------------------------------------ chunking
def test_no_chunk_exceeds_the_embedder_limit_and_sizes_hit_the_target():
    long = "\n\n".join(" ".join(f"word{i}_{j}" for j in range(60)) + "." for i in range(40))
    chunks = chunk_pages([Page("big.md", 1, long, "Big")])
    sizes = [count_tokens(c.text) for c in chunks]
    assert max(sizes) <= MAX_TOKENS and len(chunks) > 3
    assert all(250 <= s <= 450 for s in sizes[:-1]), sizes  # the last chunk may be short


def test_overlap_repeats_a_little_text_between_neighbours():
    paras = [f"Paragraph {i} " + "filler " * 80 for i in range(12)]
    chunks = chunk_pages([Page("o.md", 1, "\n\n".join(paras), "O")])
    assert len(chunks) >= 2
    tail = " ".join(chunks[0].text.split()[-8:])
    assert tail in " ".join(chunks[1].text.split())  # the end of one chunk reappears in the next
    assert count_tokens(chunks[1].text) <= MAX_TOKENS


def test_one_giant_sentence_is_hard_split_below_the_limit():
    chunks = chunk_pages([Page("g.md", 1, " ".join(f"w{i}" for i in range(2000)), "G")])
    assert max(count_tokens(c.text) for c in chunks) <= MAX_TOKENS


def test_duplicate_slides_are_removed_by_normalised_hash():
    same = "Deadlock needs four conditions: mutual exclusion, hold and wait."
    pages = [Page("s.pptx", 1, same, "A"), Page("s.pptx", 2, same.upper() + "!!", "A"),
             Page("s.pptx", 3, "Something else entirely about paging.", "B")]
    chunks = chunk_pages(pages)
    assert len(chunks) == 2 and normalise(same) == normalise(same.upper() + "!!")


def test_chunks_are_tagged_and_citable(retriever):
    c = retriever.index.chunks[0]
    assert isinstance(c, Chunk) and c.source and c.chapter and c.page == 1 and c.topic
    assert c.cite() == "unix_basics.md, p. 1 (File permissions and chmod)"


# ------------------------------------------------------------------ retrieval
def test_tokenizer_keeps_command_names_and_flags():
    assert tokens("Use `chmod 755` and ls -l, not the") == ["use", "chmod", "755", "ls", "-l"]


def test_bm25_ranks_the_exact_term_first_even_when_dense_vectors_are_vague():
    docs = [tokens("the grep command searches text"), tokens("chmod changes permission bits"),
            tokens("pipes connect commands")]
    s = BM25(docs).scores(tokens("chmod"))
    assert s.index(max(s)) == 1 and s[0] == 0 and s[2] == 0


@pytest.mark.parametrize("query,chapter", [
    ("how do I use chmod 755", "File permissions and chmod"),
    ("what does the pipe symbol do", "Pipes and redirection"),
    ("what does fork return in the child", "Processes: fork, exec and wait"),
    ("difference between a hard link and symbolic link", "Files, inodes and paths"),
    ("what is a system call and kernel mode", "System calls"),
    ("what happens on a page fault", "Virtual memory and paging"),
    ("what is a zombie process", "Processes: fork, exec and wait"),
    ("redirect standard error to a file", "Pipes and redirection"),
])
def test_retrieval_finds_the_right_section(retriever, query, chapter):
    hits = retriever.retrieve(query, k=3)
    assert hits and hits[0].chunk.chapter == chapter, [h.chunk.chapter for h in hits]
    assert set(hits[0].ranks) == {"bm25", "dense"}  # fused from both retrievers


@pytest.mark.parametrize("query", ["who won the football world cup",
                                   "best pizza recipe in naples", "xyzzy plugh"])
def test_off_topic_questions_get_not_found(retriever, query):
    a = answer(query, retriever)
    assert not a.found and a.text == NOT_FOUND and a.source == "not_found" and not a.citations


def test_answers_are_extractive_with_valid_citations(retriever):
    a = answer("How do I give the owner execute permission with chmod 755?", retriever)
    assert a.found and a.source == "extractive"
    assert "chmod 755" in a.text and a.citations == [
        "unix_basics.md, p. 1 (File permissions and chmod)"]
    assert set(map(int, re.findall(r"\[(\d+)\]", a.text))) <= set(range(1, len(a.hits) + 1))


# ------------------------------------------------------------------ LLM answers
class Stub:
    def __init__(self, payload):
        self.payload, self.calls = payload, 0

    def complete(self, system, user):
        self.calls += 1
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload if isinstance(self.payload, str) else json.dumps(self.payload)


Q = "What does fork return in the child process?"


def test_valid_llm_answer_is_used_and_cited(retriever):
    llm = Stub({"answer": "It returns zero in the child.", "citations": [1]})
    a = answer(Q, retriever, llm)
    assert a.source == "llm" and a.text == "It returns zero in the child." and a.citations
    assert llm.calls == 1


@pytest.mark.parametrize("payload,why", [
    ({"answer": "Zero.", "citations": [9]}, "invalid"),            # citation out of range
    ({"answer": "Zero.", "citations": []}, "invalid"),             # no citation at all
    ({"answer": "It returns 42 in the child.", "citations": [1]}, "number"),  # invented number
    ({"answer": "", "citations": [1]}, "no answer"),
    ("not json at all", "invalid"),
    (RuntimeError("provider down"), "unavailable"),
])
def test_bad_llm_output_falls_back_to_extracted_text(retriever, payload, why):
    a = answer(Q, retriever, Stub(payload))
    assert a.source == "extractive" and a.found and a.citations
    assert why in a.note


def test_llm_is_never_called_when_nothing_relevant_was_retrieved(retriever):
    llm = Stub({"answer": "x", "citations": [1]})
    assert answer("who won the football world cup", retriever, llm).source == "not_found"
    assert llm.calls == 0


# ------------------------------------------------------------------ persistence
def test_index_round_trips_and_rejects_a_different_embedder(tmp_path, retriever):
    path = tmp_path / "idx.json"
    retriever.index.save(path)
    again = Index.load(path, HashingEmbedder())
    r2 = Retriever(again)
    assert [h.chunk.id for h in r2.retrieve("chmod 755")] == [
        h.chunk.id for h in retriever.retrieve("chmod 755")]

    class Other(HashingEmbedder):
        name = "something-else"

    with pytest.raises(ValueError):
        Index.load(path, Other())


def test_empty_index_is_safe():
    assert Retriever(Index([], HashingEmbedder())).retrieve("anything") == []
