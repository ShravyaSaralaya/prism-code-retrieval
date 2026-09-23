"""
Narrative-as-extra-signal experiment.

Unlike the earlier (rejected) preprocessing attempt, we do NOT replace the
query text -- we ADD a third embedding (BGE-M3 on narrative-only text) as an
extra fusion signal alongside our two trusted full-text embeddings
(BGE-M3 full, E5 full). If narrative text carries real distinguishing signal
without the destructive information loss of full replacement, 3-way fusion
should help or be neutral -- not actively hurt like full replacement did.

Reuses cached corpus embeddings (corpus text is unchanged) -- only need to
embed the narrative-only QUERY text fresh. Much faster than a full re-embed.

Run:  python narrative_signal_fusion.py
"""

import os
import re
import json
import numpy as np
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CACHE_DIR = "embed_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
K_VALUES = [1, 3, 5, 10, 20, 50, 100]

def cache_path(name):
    return os.path.join(CACHE_DIR, name)

# ---- Same conservative preprocessing validated in preview_preprocessing.py ----
DASHED_HEADER_RE = re.compile(
    r'\n-{3,}\s*(Input|Output|Examples?|Note|Constraints?)\s*-{3,}',
    re.IGNORECASE
)
BARE_HEADER_RE = re.compile(
    r'^[ \t]*(Input|Output|Constraints?|Examples?|Note)[ \t]*\d*[ \t]*:?[ \t]*$',
    re.IGNORECASE | re.MULTILINE
)

def strip_io_boilerplate(text: str) -> str:
    candidates = []
    m = DASHED_HEADER_RE.search(text)
    if m:
        candidates.append(m.start())
    m = BARE_HEADER_RE.search(text)
    if m:
        candidates.append(m.start())
    if candidates:
        cut_at = min(candidates)
        narrative = text[:cut_at].strip()
        if len(narrative) >= 40:
            return narrative
    return text.strip()

def minmax_normalize(sims):
    mn = sims.min(axis=1, keepdims=True)
    mx = sims.max(axis=1, keepdims=True)
    return (sims - mn) / (mx - mn + 1e-8)

def gold_rank_from_sims(sims, gold_idx):
    gold_sims = sims[np.arange(len(gold_idx)), gold_idx]
    return np.sum(sims > gold_sims[:, None], axis=1) + 1

def compute_metrics(ranks, k_values=K_VALUES):
    metrics = {}
    for k in k_values:
        metrics[f"recall_at_{k}"] = float(np.mean(ranks <= k))
    ndcg10 = np.where(ranks <= 10, 1.0 / np.log2(ranks + 1), 0.0)
    mrr10 = np.where(ranks <= 10, 1.0 / ranks, 0.0)
    metrics["ndcg_at_10"] = float(np.mean(ndcg10))
    metrics["mrr_at_10"] = float(np.mean(mrr10))
    return metrics

# ---------------- Load data ----------------
print("Loading dataset...")
corpus = load_dataset("CoIR-Retrieval/apps", "corpus", split="corpus")
queries = load_dataset("CoIR-Retrieval/apps", "queries", split="queries")
qrels = load_dataset("CoIR-Retrieval/apps", split="test")

corpus_ids = corpus["_id"]
id_to_corpus_idx = {cid: i for i, cid in enumerate(corpus_ids)}
query_id_to_text = {qid: text for qid, text in zip(queries["_id"], queries["text"])}
test_query_ids = qrels["query-id"]
test_query_texts = [query_id_to_text[qid] for qid in test_query_ids]
gold_idx = np.array([id_to_corpus_idx[cid] for cid in qrels["corpus-id"]])
n_queries = len(test_query_ids)

# ---------------- Load cached full-text embeddings (required) ----------------
required = ["bge_corpus.npy", "bge_query.npy", "e5_corpus.npy", "e5_query.npy"]
missing = [f for f in required if not os.path.exists(cache_path(f))]
if missing:
    raise FileNotFoundError(f"Missing cached embeddings: {missing}. Run experiment2_reranker.py first.")

print("Loading cached full-text embeddings...")
bge_corpus_emb = np.load(cache_path("bge_corpus.npy"))
bge_query_emb_full = np.load(cache_path("bge_query.npy"))
e5_corpus_emb = np.load(cache_path("e5_corpus.npy"))
e5_query_emb_full = np.load(cache_path("e5_query.npy"))

# ---------------- Narrative-only query embeddings (NEW, query-side only) ----------------
narrative_cache = cache_path("bge_narrative_query.npy")
if os.path.exists(narrative_cache):
    print("Loading cached narrative-query embeddings...")
    bge_query_emb_narrative = np.load(narrative_cache)
else:
    print("Building narrative-only query text...")
    narrative_texts = [strip_io_boilerplate(t) for t in test_query_texts]
    unchanged = sum(1 for a, b in zip(narrative_texts, test_query_texts) if a == b)
    print(f"({unchanged}/{n_queries} queries had no recognized marker, kept full text as fallback)")

    print("Embedding narrative-only queries with BGE-M3 (corpus reused from cache, no re-embed needed)...")
    bge_model = SentenceTransformer(
        "BAAI/bge-m3",
        model_kwargs={"torch_dtype": torch.float16} if DEVICE == "cuda" else {},
        device=DEVICE,
    )
    bge_model.max_seq_length = 1536
    bge_query_emb_narrative = bge_model.encode(
        narrative_texts, batch_size=8, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True,
    ).astype(np.float32)
    del bge_model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    np.save(narrative_cache, bge_query_emb_narrative)

# ---------------- Compute all three similarity signals ----------------
print("\nComputing similarities...")
bge_sims_full = minmax_normalize(bge_query_emb_full @ bge_corpus_emb.T)
e5_sims_full = minmax_normalize(e5_query_emb_full @ e5_corpus_emb.T)
bge_sims_narrative = minmax_normalize(bge_query_emb_narrative @ bge_corpus_emb.T)

# ---------------- Baseline: 2-way fusion (our current banked best) ----------------
fused_2way = bge_sims_full + e5_sims_full
ranks_2way = gold_rank_from_sims(fused_2way, gold_idx)
metrics_2way = compute_metrics(ranks_2way)
print("\n=== 2-way fusion (BGE-M3 full + E5 full) -- current banked baseline ===")
print(json.dumps(metrics_2way, indent=2))

# ---------------- New: 3-way fusion (adding narrative as extra signal) ----------------
fused_3way = bge_sims_full + e5_sims_full + bge_sims_narrative
ranks_3way = gold_rank_from_sims(fused_3way, gold_idx)
metrics_3way = compute_metrics(ranks_3way)
print("\n=== 3-way fusion (BGE-M3 full + E5 full + BGE-M3 narrative) ===")
print(json.dumps(metrics_3way, indent=2))

# ---------------- Comparison ----------------
print("\n" + "=" * 80)
print("COMPARISON: 2-way fusion vs 3-way fusion (with narrative signal added)")
print("=" * 80)
print(f"{'Metric':<15}{'2-way fusion':>15}{'3-way fusion':>15}{'Change':>12}")
for key in ["recall_at_1","recall_at_3","recall_at_5","recall_at_10","recall_at_20",
            "recall_at_50","recall_at_100","ndcg_at_10","mrr_at_10"]:
    a, b = metrics_2way[key], metrics_3way[key]
    change = f"{(b-a)/a*100:+.2f}%" if a > 0 else "n/a"
    print(f"{key:<15}{a:>15.4f}{b:>15.4f}{change:>12}")

with open("narrative_signal_fusion_results.json", "w") as f:
    json.dump({"2way_fusion": metrics_2way, "3way_fusion": metrics_3way}, f, indent=2)
print("\nSaved to: narrative_signal_fusion_results.json")