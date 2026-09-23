"""
Experiment: cross-encoder reranking of the top-100 fused (BGE-M3 + E5) candidates.

Embeddings are cached to embed_cache/*.npy so a buggy rerank step doesn't force
re-embedding everything from scratch on retry.

Run:  python experiment2_reranker.py
"""

import os
import time
import json
import numpy as np
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, CrossEncoder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CACHE_DIR = "embed_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
TOP_K_RERANK = 100
K_VALUES = [1, 3, 5, 10, 20, 50, 100]

def cache_path(name):
    return os.path.join(CACHE_DIR, name)

def load_or_compute(tag, model_name, corpus_texts, query_texts,
                     query_prefix="", passage_prefix="", batch_size=8, seq_len=1536, model_kwargs=None):
    cpath, qpath = cache_path(f"{tag}_corpus.npy"), cache_path(f"{tag}_query.npy")
    if os.path.exists(cpath) and os.path.exists(qpath):
        print(f"Loading cached embeddings for {tag}...")
        return np.load(cpath), np.load(qpath)
    print(f"Computing embeddings for {tag} (no cache found -- this is the slow part)...")
    model = SentenceTransformer(model_name, model_kwargs=model_kwargs or {}, device=DEVICE)
    model.max_seq_length = seq_len
    corpus_emb = model.encode(
        [passage_prefix + t for t in corpus_texts], batch_size=batch_size,
        show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True,
    ).astype(np.float32)
    query_emb = model.encode(
        [query_prefix + t for t in query_texts], batch_size=batch_size,
        show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True,
    ).astype(np.float32)
    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    np.save(cpath, corpus_emb)
    np.save(qpath, query_emb)
    return corpus_emb, query_emb

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
corpus_texts = corpus["text"]
id_to_corpus_idx = {cid: i for i, cid in enumerate(corpus_ids)}
query_id_to_text = {qid: text for qid, text in zip(queries["_id"], queries["text"])}

test_query_ids = qrels["query-id"]
test_gold_ids = qrels["corpus-id"]
test_query_texts = [query_id_to_text[qid] for qid in test_query_ids]
gold_idx = np.array([id_to_corpus_idx[cid] for cid in test_gold_ids])

n_queries = len(test_query_ids)
n_corpus = len(corpus_ids)
print(f"{n_queries} test queries, {n_corpus} corpus docs\n")

# ---------------- Embeddings (cached) ----------------
bge_corpus_emb, bge_query_emb = load_or_compute(
    "bge", "BAAI/bge-m3", corpus_texts, test_query_texts,
    batch_size=8, seq_len=1536,
    model_kwargs={"torch_dtype": torch.float16} if DEVICE == "cuda" else {},
)
e5_corpus_emb, e5_query_emb = load_or_compute(
    "e5", "intfloat/e5-base-v2", corpus_texts, test_query_texts,
    query_prefix="query: ", passage_prefix="passage: ", batch_size=32, seq_len=512,
)

# ---------------- Fusion (same as Experiment 1's winner: simple score fusion) ----------------
print("\nComputing fused similarity scores...")
bge_sims = bge_query_emb @ bge_corpus_emb.T
e5_sims = e5_query_emb @ e5_corpus_emb.T
fused_sims = minmax_normalize(bge_sims) + minmax_normalize(e5_sims)

fused_ranks = gold_rank_from_sims(fused_sims, gold_idx)
fused_metrics = compute_metrics(fused_ranks)
print("Fused (BGE-M3 + E5, no reranking) -- should match Experiment 1's score-fusion numbers:")
print(json.dumps(fused_metrics, indent=2))

top100_idx = np.argsort(-fused_sims, axis=1)[:, :TOP_K_RERANK]  # [n_queries, 100] candidate corpus indices

# ---------------- Cross-encoder reranking ----------------
print(f"\n=== Cross-encoder reranking top-{TOP_K_RERANK} candidates ===")
CE_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
print(f"Loading {CE_MODEL} (max_length=512 -- expect truncation on our long texts)...")
ce = CrossEncoder(CE_MODEL, max_length=512, device=DEVICE)

all_pairs = []
for qi in range(n_queries):
    q_text = test_query_texts[qi]
    for ci in top100_idx[qi]:
        all_pairs.append((q_text, corpus_texts[ci]))

print(f"Scoring {len(all_pairs)} (query, candidate) pairs...")
t0 = time.time()
ce_scores_flat = ce.predict(all_pairs, batch_size=128, show_progress_bar=True)
ce_time = time.time() - t0
print(f"Cross-encoder scoring took {ce_time:.1f}s total ({ce_time/n_queries*1000:.1f}ms/query avg, "
      f"{len(all_pairs)/ce_time:.1f} pairs/sec)")

ce_scores = np.array(ce_scores_flat).reshape(n_queries, TOP_K_RERANK)
rerank_order = np.argsort(-ce_scores, axis=1)
reranked_top100_idx = np.take_along_axis(top100_idx, rerank_order, axis=1)

# Final rank: if gold was in top-100, use its NEW position after reranking.
# If not, it was never reachable -- keep its original (already >100) rank.
final_ranks = fused_ranks.copy()
for qi in range(n_queries):
    match = np.where(reranked_top100_idx[qi] == gold_idx[qi])[0]
    if len(match) > 0:
        final_ranks[qi] = match[0] + 1

reranked_metrics = compute_metrics(final_ranks)

# ---------------- Comparison ----------------
print("\n" + "=" * 90)
print("FINAL COMPARISON: fused retrieval vs fused + cross-encoder reranking")
print("=" * 90)
print(f"{'Metric':<15}{'Fused (no rerank)':>20}{'Fused + reranked':>20}{'Change':>12}")
for key in ["recall_at_1","recall_at_3","recall_at_5","recall_at_10","recall_at_20",
            "recall_at_50","recall_at_100","ndcg_at_10","mrr_at_10"]:
    a, b = fused_metrics[key], reranked_metrics[key]
    change = f"{(b-a)/a*100:+.1f}%" if a > 0 else "n/a"
    print(f"{key:<15}{a:>20.4f}{b:>20.4f}{change:>12}")

results_all = {
    "fused_no_rerank": fused_metrics,
    "fused_plus_reranked": reranked_metrics,
    "cross_encoder_model": CE_MODEL,
    "timing_seconds": {
        "cross_encoder_scoring_total": ce_time,
        "cross_encoder_ms_per_query": ce_time / n_queries * 1000,
        "pairs_scored": len(all_pairs),
    },
}
with open("experiment2_reranker_results.json", "w") as f:
    json.dump(results_all, f, indent=2)

print("\nSaved full results to: experiment2_reranker_results.json")