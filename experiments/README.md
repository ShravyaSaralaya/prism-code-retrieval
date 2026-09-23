# Experiment log

All numbers below are NDCG@10 / MRR@10 on the CoIR `AppsRetrieval` test split
(3,765 queries, 8,765 corpus documents), computed via MTEB unless noted.

## 1. Baseline sweep — picking an embedding model

| Model | NDCG@10 | MRR@10 |
|---|---|---|
| `all-MiniLM-L6-v2` (generic small model, day 1 sanity check) | 0.0660 | 0.0558 |
| `intfloat/e5-base-v2` | 0.1152 | 0.0988 |
| `Snowflake/snowflake-arctic-embed-m-v2.0` | — (dependency issue, abandoned — published number ~0.108, weaker than e5 anyway) | — |
| **`BAAI/bge-m3`** | **0.1475** | **0.1277** |

`bge-m3`'s long context window (used at 1536 tokens) handles this dataset's
unusually long queries (problem statements averaging ~1,690 characters, up to
4,500+) far better than shorter-context models. This became our strongest
single model.

## 2. Dense-dense fusion — the winning approach

Custom per-query retrieval pipeline built to get exact gold-document ranks
(needed for overlap analysis; MTEB's built-in evaluator only reports
aggregate metrics). Validated against MTEB's own saved numbers before
trusting any downstream result.

**Overlap analysis (top-100), BGE-M3 vs E5-base-v2:**

| | Count | % of test set |
|---|---|---|
| Found by both | 976 | 25.9% |
| Found only by BGE-M3 | 641 | 17.0% |
| Found only by E5 | 317 | 8.4% |
| Found by neither | 1,831 | 48.6% |

E5 recovers a genuine 8.4% of queries BGE-M3's top-100 completely misses —
real complementary signal, justifying fusion.

**Fusion method comparison:**

| Method | NDCG@10 | MRR@10 |
|---|---|---|
| BGE-M3 alone | 0.1475 | 0.1277 |
| E5-base-v2 alone | 0.1152 | 0.0988 |
| Reciprocal Rank Fusion (RRF) | 0.1678 | 0.1440 |
| **Min-max normalized score fusion (winner)** | **0.1761** | **0.1527** |

Score fusion beat RRF — the raw similarity magnitudes carry real confidence
signal that RRF's rank-only approach discards.

**Weighted fusion sweep:** tested mixing weights from 0 (pure E5) to 1 (pure
BGE-M3) in steps of 0.05. Equal weighting (0.5/0.5) was already
near-optimal — the best alternative (0.55) improved NDCG@10 by only +0.05%,
within noise. Kept equal weighting for simplicity.

## 3. Rejected experiments

Each of these was implemented, run on the full test split, and rejected based
on evidence — not skipped or assumed.

### Query preprocessing (strip Input/Output/Examples boilerplate)
Hypothesis: problem statements contain repetitive format boilerplate that
adds noise. Implemented a conservative regex-based stripper (kept full text
as fallback whenever no recognized section marker was found).

| | NDCG@10 | MRR@10 |
|---|---|---|
| BGE-M3 baseline | 0.1475 | 0.1277 |
| BGE-M3 + preprocessing | 0.0913 | 0.0784 |

**Result: −38% / −39%. Rejected.** The "boilerplate" sections actually carry
real distinguishing signal (specific numbers, formats, edge cases) that helps
tell superficially similar problems apart.

### Cross-encoder reranking (`cross-encoder/ms-marco-MiniLM-L-6-v2`)
Reranked the fused top-100 candidates per query.

| | NDCG@10 | MRR@10 |
|---|---|---|
| Fused (no rerank) | 0.1761 | 0.1527 |
| Fused + reranked | 0.0598 | 0.0459 |

**Result: −66% / −70%. Rejected.** The reranker was trained on short web-search
query/passage pairs — no exposure to code — and appears to actively reward
surface-level phrasing similarity over algorithmic correctness. Also far too
slow for the demo's speed requirement (543ms/query).

### Narrative-as-extra-signal fusion
Follow-up hypothesis: maybe preprocessing failed because it *replaced* the
query; adding the narrative-only text as a third fusion signal (alongside,
not instead of, the full text) might avoid the information loss.

| | NDCG@10 | MRR@10 |
|---|---|---|
| 2-way fusion (baseline) | 0.1761 | 0.1527 |
| 3-way fusion (+ narrative) | 0.1495 | 0.1280 |

**Result: −15% / −16%. Rejected.** Confirms the pattern: the narrative-only
embedding is a genuinely weaker standalone signal, not just harmful when it
replaces the full text.

### Fine-tuning (see `out_of_scope_finetuning/`)
Implemented a LoRA fine-tuning pipeline with hard-negative mining (trained on
the CoIR train split only, no test leakage). Organizers subsequently clarified
that fine-tuning the embedding model is **out of scope** for this problem
statement. Kept in the repo for transparency; not part of the final pipeline.

## Summary

Three independent experiments (preprocessing, reranking, narrative fusion)
all point to the same conclusion: **generic components not built for
code/competitive-programming text actively hurt more often than they help**
on this dataset. The winning approach — combining two off-the-shelf models
via score fusion — works because it adds genuinely complementary signal
without discarding any information.
