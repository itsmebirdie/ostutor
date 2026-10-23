# Evaluation summary

*Python 3.12.3 on Linux x86_64. Produced by `python eval/run_all.py`.*

- **classifier**: Tier A agrees on 44/44; replay 26/26  (28.3s, see `classifier.md`)
- **efficiency**: 94,864 states/s, 1,372 bytes/state at 7,759 states  (2.7s, see `efficiency.md`)
- **faithfulness**: templates 44/44; tamper rejection 26/26, 44/44, 44/44, 44/44  (0.0s, see `faithfulness.md`)
- **diagnosis (synthetic)**: 16/16 mutants recognised (synthetic, circular)  (0.1s, see `diagnosis.md`)
- **router**: 31/32 agree  (0.0s, see `router.md`)
- **rag (sample corpus)**: hit@1 22/24, hit@3 24/24, off-topic leaked 1/8  (0.2s, see `rag.md`)

## Read before quoting any of this

- The concurrency labels were derived by reasoning, not by two independent annotators:
  this is **agreement with construction labels**, not accuracy. Verify them by hand first.
- Diagnosis results are **synthetic and circular**; no diagnostic-accuracy claim is possible
  without real, consented student answers.
- RAG numbers use a six-section sample corpus and a lexical embedder.
- No LLM pass/regeneration rate appears until recorded answers exist (`ostutor warm`).
- Not measured anywhere: learning outcomes, latency with a real LLM, token costs at scale.

