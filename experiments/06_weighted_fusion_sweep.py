"""
Weighted fusion sweep: is equal-weight (1x BGE-M3 + 1x E5) actually optimal,
or does weighting toward the stronger model (BGE-M3) do better?

Reuses the embeddings already cached in embed_cache/ from experiment2 --
no model loading, no GPU, pure numpy. Should run in seconds.

Run:  python weighted_fusion_sweep.py
"""

import os
import json
import numpy as np
from datasets import load_dataset

CACHE_DIR = "embed_cache"
K_VALUES = [1, 3, 5, 10, 20, 50, 100]

def cache_path(name):
    return os.path.join(CACHE_DIR, name)

required = ["bge_corpus.npy", "bge_query.npy", "e5_corpus.npy", "e5_query.npy"]
missing = [f for f in required if not os.path.exists(cache_path(f))]
if missing:
    raise FileNotFoundError(
        f"Missing cached embeddings: {missing}. "
        f"Run experiment2_reranker.py at least once first to populate embed_cache/."
    )

print("Loading cached embeddings (no GPU needed)...")
bge_corpus_emb = np.load(cache_path("bge_corpus.npy"))
bge_query_emb = np.load(cache_path("bge_query.npy"))
e5_corpus_emb = np.load(cache_path("e5_corpus.npy"))
e5_query_emb = np.load(cache_path("e5_query.npy"))

print("Loading dataset (for gold labels only)...")
corpus = load_dataset("CoIR-Retrieval/apps", "corpus", split="corpus")
qrels = load_dataset("CoIR-Retrieval/apps", split="test")
corpus_ids = corpus["_id"]
id_to_corpus_idx = {cid: i for i, cid in enumerate(corpus_ids)}
gold_idx = np.array([id_to_corpus_idx[cid] for cid in qrels["corpus-id"]])

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

print("Computing similarities...")
bge_sims = bge_query_emb @ bge_corpus_emb.T
e5_sims = e5_query_emb @ e5_corpus_emb.T
bge_norm = minmax_normalize(bge_sims)
e5_norm = minmax_normalize(e5_sims)

print("\nSweeping fusion weight alpha (fused = alpha*BGE-M3 + (1-alpha)*E5)...\n")
print(f"{'alpha (BGE-M3 weight)':<25}{'ndcg_at_10':>12}{'mrr_at_10':>12}{'recall_at_10':>14}{'recall_at_100':>15}")

results = []
for alpha in np.arange(0.0, 1.01, 0.05):
    fused = alpha * bge_norm + (1 - alpha) * e5_norm
    ranks = gold_rank_from_sims(fused, gold_idx)
    m = compute_metrics(ranks)
    results.append((alpha, m))
    print(f"{alpha:<25.2f}{m['ndcg_at_10']:>12.4f}{m['mrr_at_10']:>12.4f}{m['recall_at_10']:>14.4f}{m['recall_at_100']:>15.4f}")

best_alpha, best_metrics = max(results, key=lambda r: r[1]["ndcg_at_10"])
print(f"\nBest alpha by NDCG@10: {best_alpha:.2f}")
print(json.dumps(best_metrics, indent=2))

# Reference: equal-weight (alpha=0.5) result, our current banked baseline
equal_weight_metrics = next(m for a, m in results if abs(a - 0.5) < 1e-6)
print(f"\nFor comparison, equal-weight (alpha=0.5, our current banked baseline):")
print(json.dumps(equal_weight_metrics, indent=2))

improvement = (best_metrics["ndcg_at_10"] - equal_weight_metrics["ndcg_at_10"]) / equal_weight_metrics["ndcg_at_10"] * 100
print(f"\nBest alpha improvement over equal-weight: {improvement:+.2f}% NDCG@10")

with open("weighted_fusion_sweep_results.json", "w") as f:
    json.dump({
        "sweep": [{"alpha": float(a), **m} for a, m in results],
        "best_alpha": float(best_alpha),
        "best_metrics": best_metrics,
        "equal_weight_metrics": equal_weight_metrics,
    }, f, indent=2)
print("\nSaved to: weighted_fusion_sweep_results.json")