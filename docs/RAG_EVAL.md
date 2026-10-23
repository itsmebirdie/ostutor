# Evaluating RAG with real bge embeddings

The shipped evaluation (`eval/run_rag.py`) uses six sample sections and an offline *lexical*
embedder, so its numbers say almost nothing. A real evaluation needs three things: your real
course material, real questions, and a real embedder. Use `eval/run_rag_real.py`.

## 1. Install

```bash
pip install -e ".[dev,embed]"        # embed = sentence-transformers (pulls in PyTorch)
```

On a laptop without a GPU, PyTorch's CPU build is enough. First use downloads the model
(bge-small about 130 MB, bge-large about 1.3 GB) from Hugging Face.

## 2. Put your material in `data/corpus_raw/`

PDF, PPTX, Markdown or text. The folder is git-ignored (course material is copyrighted; ingest
only what you may use, for example OSTEP, which is openly available). Slides and pages become
chunks of 250-400 tokens with a small overlap, tagged with source and page.

## 3. Write the questions (this is the part that decides whether the result means anything)

Create `eval/my_questions.jsonl`, one JSON object per line (see `rag_questions.example.jsonl`):

```json
{"question": "What does the TLB cache?", "answerable": true, "source": "os_notes.pdf", "page": 41}
{"question": "Explain the Banker's algorithm", "answerable": true, "source": "week5.pptx", "page": [3, 4]}
{"question": "Who won the world cup?", "answerable": false}
```

- **Source and page** are where the answer actually is; a retrieved chunk counts as correct
  only if it is on that page (a list is allowed for answers that span pages).
- Aim for **100+ answerable** questions and **30+ unanswerable** ones. With 16 questions a
  hit rate of 0.94 has a confidence interval of roughly 0.81 to 1.00, which proves nothing.
- Use **real student questions** if you can. Questions written by the person who built the
  system are systematically easier (they reuse the document's own words).
- Include **hard unanswerable** questions: OS-adjacent but not in your material ("how does
  Windows ReFS work?"), not only "who won the world cup".

## 4. Run

```bash
python eval/run_rag_real.py --corpus data/corpus_raw --questions eval/my_questions.jsonl \
    --embedder bge-small --out eval/results/rag_real_small
python eval/run_rag_real.py ... --embedder bge-large --out eval/results/rag_real_large
python eval/run_rag_real.py ... --embedder hashing  --out eval/results/rag_real_hash   # baseline
```

The index is embedded once and cached (`*.index.json` plus a fingerprint of the chunks and
the model); later runs load it and only embed the queries. If you change the corpus or chunking
the fingerprint changes and it re-embeds.

### Embedding the corpus once on Colab (the plan's route for bge-large)

1. In a Colab GPU runtime: clone the repo, `pip install -e ".[embed]"`, upload `data/corpus_raw`
   and your questions file, run the command above with `--embedder bge-large`.
2. Download `eval/results/<out>/BAAI_bge-large-en-v1.5.index.json` and the `.fingerprint` file.
3. Put both in the same `--out` folder locally and run the same command. It loads the vectors
   and only needs the model to embed queries.

The fingerprint must match exactly, so use the same corpus files and the same `pypdf` version on
both machines (PDF text extraction can differ slightly between versions). If it does not match,
the script silently re-embeds locally, which is correct but slow on CPU.

Queries need the model locally: bge-large costs a second or two per query on a laptop CPU (measure
yours). If that is too slow for the demo, switch to bge-small (the plan's fall-back).

## 5. What the report gives you, and how to read it

- **Dev/test split** by hash of the question text, so it is deterministic. Tune on dev, report
  test, **once**. If you change anything after reading the test numbers, you need new test
  questions (re-split with a different salt only if you have not looked).
- **Ranking quality** for BM25 only, dense only, and hybrid (reciprocal rank fusion), with
  hit@1, hit@3, hit@5 and MRR with 95% bootstrap intervals, and a paired exact McNemar test
  between systems. This answers "does bge help, and does fusing help?". If hybrid is not
  significantly better than BM25 alone on your questions, **say so**: that is a result.
- **Not-found gate**: the thresholds are tuned on dev only (balanced accuracy of "answer"
  versus "not found") over the observed score quantiles. Copy them into
  `Retriever(index, bm25_min=..., dense_min=...)`. **The shipped defaults (1.0 / 0.25) are for the
  offline embedder and are wrong for bge**, whose cosine scores for unrelated text are often
  0.3 to 0.5.
- **Failure list**: every test question whose answer was not in the top 3. Read it; it is where
  you will find chunking problems (answers split across chunks), duplicate slides, or questions
  whose "source" label is wrong.

## 6. Sentences you can defend in the report

- "On N answerable test questions, hybrid retrieval with bge-large put a relevant chunk in the
  top 3 for X% (95% CI a to b); BM25 alone: Y%; paired McNemar p=..."
- "Thresholds tuned on a dev split of M questions; on the held-out test split, K of U
  unanswerable questions were wrongly answered (leak rate L)."

## Things this does not measure

Whether the *answer* is faithful to the cited chunk (retrieval is only half of RAG; the LLM
answer path validates citations and numbers mechanically but is not audited by this script),
answer quality as students perceive it, and latency under load.
