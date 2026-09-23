# Code Retrieval — Samsung PRISM Theme 1 (Agentic Code Intelligence)

Given a natural-language query and a corpus of code snippets, rank the snippets by
relevance to the query. Evaluated on the CoIR `AppsRetrieval` benchmark (English
problem statements → Python solutions), scored on NDCG@10 and MRR.

## Approach

**Retrieval model: dense-dense fusion of two off-the-shelf embedding models**
(`BAAI/bge-m3` + `intfloat/e5-base-v2`), combined via min-max normalized score
fusion. No fine-tuning — this stays within the problem statement's explicit
scope ("free to use the models and methods available and combine them with
your own methods"); fine-tuning the embedding model was separately confirmed
**out of scope** by the organizers.

| Metric | Value |
|---|---|
| NDCG@10 | 0.1761 |
| MRR@10 | 0.1527 |
| Recall@100 | 0.4744 |

This is a **+19% relative improvement** over the strongest single model we
tested (BGE-M3 alone, NDCG@10 = 0.1475), and roughly **2.7x** our initial
baseline (`all-MiniLM-L6-v2`, NDCG@10 = 0.0660).

See [`experiments/README.md`](experiments/README.md) for the full experiment
log — including three approaches we tried and rejected with evidence
(query preprocessing, cross-encoder reranking, narrative-signal fusion).

**P1 (retrieval across versions):** implemented as a content-hash-based
incremental index — see [`src/incremental_index.py`](src/incremental_index.py)
and the demo below. Re-indexing a small commit (23 changed docs out of 8,765)
is **~200x faster** than a full rebuild.

## Repository structure

```
src/                    Final submission pipeline
  official_fusion_submission.py   Generates the official MTEB submission JSON (P0)
  incremental_index.py            Reusable incremental embedding index (P1)
  p1_demo.py                      P1 speedup demonstration

results/
  OFFICIAL_fusion_submission_results.json   The submitted evaluation JSON

experiments/            Full experiment log, in chronological order
  README.md             Narrative summary with all numbers
  01-09_*.py            Individual experiment scripts (rejected ones clearly labeled)
  out_of_scope_finetuning/   Fine-tuning attempt, kept for transparency
                              (confirmed out of scope by organizers)
```

## Setup

```bash
python -m venv venv
# Windows: venv\Scripts\Activate.ps1
# macOS/Linux: source venv/bin/activate

pip install -r requirements.txt
```

Requires a CUDA GPU for reasonable indexing speed (development was done on an
RTX 3050, 4GB VRAM). **Inference-time retrieval is CPU-friendly** — both
models are small enough (568M and 110M parameters) to query on CPU alone, per
the problem statement's requirement; GPU is only used here to make the
one-time corpus indexing step faster.

## Reproducing the official submission

```bash
cd src
python official_fusion_submission.py
```

This embeds the full corpus + all test queries with both models, fuses the
scores, and runs the **official MTEB evaluation pipeline**
(`mteb.evaluate()`), producing `OFFICIAL_fusion_submission_results.json` —
identical in format to the file in `results/`. Takes ~9 minutes on an RTX
3050.

## Reproducing the P1 (re-indexing speed) demo

```bash
cd src
python p1_demo.py
```

Builds a full index (version 1, ~4 minutes), simulates a small commit
(15 docs changed, 5 removed, 8 added), then re-syncs the index — demonstrating
that only the changed documents are re-embedded, not the full corpus.

## Why not just ask an LLM to rank the snippets?

The corpus contains thousands of snippets, some individually quite long —
too much to fit in an LLM's context window, and retrieval is expected to be
the fast first stage of a RAG pipeline, ahead of any (slower) generation step.
