"""
Experiment 1: BGE-M3 vs E5-base-v2 dense-dense fusion.

Custom per-query retrieval pipeline (MTEB's evaluate() only gives aggregate
metrics -- we need per-query gold-doc ranks to answer "found by both /
only one / neither", and to implement RRF/score fusion ourselves).

One script, one run: embeds both models once, computes everything.

Run:  python experiment1_fusion.py
"""

import time
import json
import numpy as np
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
K_VALUES = [1, 3, 5, 10, 20, 50, 100]

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

# ---------------- Embedding helper ----------------
def embed_all(model, corpus_texts, query_texts, query_prefix="", passage_prefix="",
              batch_size=8, seq_len=1536):
    model.max_seq_length = seq_len
    t0 = time.time()
    corpus_emb = model.encode(
        [passage_prefix + t for t in corpus_texts],
        batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True,
        normalize_embeddings=True,
    )
    t1 = time.time()
    query_emb = model.encode(
        [query_prefix + t for t in query_texts],
        batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True,
        normalize_embeddings=True,
    )
    t2 = time.time()
    return corpus_emb.astype(np.float32), query_emb.astype(np.float32), (t1 - t0), (t2 - t1)

def get_sims_and_ranking(query_emb, corpus_emb):
    sims = query_emb @ corpus_emb.T  # cosine similarity (embeddings are normalized)
    ranking = np.argsort(-sims, axis=1)  # descending, per query
    return sims, ranking

def gold_rank_from_sims(sims, gold_idx):
    """Exact 1-based rank of the gold doc per query (fast, no full-array loop)."""
    gold_sims = sims[np.arange(len(gold_idx)), gold_idx]
    ranks = np.sum(sims > gold_sims[:, None], axis=1) + 1
    return ranks

def compute_metrics(ranks, k_values=K_VALUES):
    metrics = {}
    for k in k_values:
        metrics[f"recall_at_{k}"] = float(np.mean(ranks <= k))
    ndcg10 = np.where(ranks <= 10, 1.0 / np.log2(ranks + 1), 0.0)
    mrr10 = np.where(ranks <= 10, 1.0 / ranks, 0.0)
    metrics["ndcg_at_10"] = float(np.mean(ndcg10))
    metrics["mrr_at_10"] = float(np.mean(mrr10))
    return metrics

# ---------------- BGE-M3 ----------------
print("=== Embedding with BGE-M3 ===")
bge_model = SentenceTransformer(
    "BAAI/bge-m3",
    model_kwargs={"torch_dtype": torch.float16} if DEVICE == "cuda" else {},
    device=DEVICE,
)
bge_corpus_emb, bge_query_emb, bge_corpus_time, bge_query_time = embed_all(
    bge_model, corpus_texts, test_query_texts, batch_size=8, seq_len=1536
)
del bge_model
if DEVICE == "cuda":
    torch.cuda.empty_cache()

bge_sims, bge_ranking = get_sims_and_ranking(bge_query_emb, bge_corpus_emb)
bge_ranks = gold_rank_from_sims(bge_sims, gold_idx)
bge_metrics = compute_metrics(bge_ranks)
print(f"BGE-M3: {json.dumps(bge_metrics, indent=2)}")
print(f"(sanity check -- should be close to saved baseline: ndcg_at_10=0.1475, mrr_at_10=0.1277)")
print(f"Timing -- corpus embed: {bge_corpus_time:.1f}s, query embed: {bge_query_time:.1f}s\n")

# ---------------- E5-base-v2 ----------------
print("=== Embedding with E5-base-v2 ===")
e5_model = SentenceTransformer("intfloat/e5-base-v2", device=DEVICE)
e5_corpus_emb, e5_query_emb, e5_corpus_time, e5_query_time = embed_all(
    e5_model, corpus_texts, test_query_texts,
    query_prefix="query: ", passage_prefix="passage: ",
    batch_size=32, seq_len=512,
)
del e5_model
if DEVICE == "cuda":
    torch.cuda.empty_cache()

e5_sims, e5_ranking = get_sims_and_ranking(e5_query_emb, e5_corpus_emb)
e5_ranks = gold_rank_from_sims(e5_sims, gold_idx)
e5_metrics = compute_metrics(e5_ranks)
print(f"E5-base-v2: {json.dumps(e5_metrics, indent=2)}")
print(f"Timing -- corpus embed: {e5_corpus_time:.1f}s, query embed: {e5_query_time:.1f}s\n")

# ---------------- Overlap analysis @ top-100 ----------------
bge_found = bge_ranks <= 100
e5_found = e5_ranks <= 100
both = int(np.sum(bge_found & e5_found))
only_bge = int(np.sum(bge_found & ~e5_found))
only_e5 = int(np.sum(~bge_found & e5_found))
neither = int(np.sum(~bge_found & ~e5_found))
union_recall_100 = float(np.mean(bge_found | e5_found))

print("=== Overlap @ top-100 ===")
print(f"Found by BOTH       : {both} ({both/n_queries*100:.2f}%)")
print(f"Found ONLY by BGE-M3: {only_bge} ({only_bge/n_queries*100:.2f}%)")
print(f"Found ONLY by E5    : {only_e5} ({only_e5/n_queries*100:.2f}%)")
print(f"Found by NEITHER    : {neither} ({neither/n_queries*100:.2f}%)")
print(f"Union Recall@100    : {union_recall_100:.4f}  (vs BGE-M3 alone: {bge_metrics['recall_at_100']:.4f})\n")

# ---------------- RRF fusion ----------------
def rrf_fuse(ranking_a, ranking_b, k=60):
    n_q, n_c = ranking_a.shape
    fused_scores = np.zeros((n_q, n_c), dtype=np.float64)
    for rank_arr in (ranking_a, ranking_b):
        inv = np.argsort(rank_arr, axis=1)  # inv[i, doc_idx] = 0-based position of doc_idx
        fused_scores += 1.0 / (k + inv + 1)
    return fused_scores

t0 = time.time()
rrf_scores = rrf_fuse(bge_ranking, e5_ranking, k=60)
rrf_time = time.time() - t0
rrf_ranks = gold_rank_from_sims(rrf_scores, gold_idx)
rrf_metrics = compute_metrics(rrf_ranks)
print(f"=== RRF fusion (computed in {rrf_time:.2f}s) ===")
print(json.dumps(rrf_metrics, indent=2))

# ---------------- Simple normalized-score fusion ----------------
def minmax_normalize(sims):
    mn = sims.min(axis=1, keepdims=True)
    mx = sims.max(axis=1, keepdims=True)
    return (sims - mn) / (mx - mn + 1e-8)

t0 = time.time()
fused_simple = minmax_normalize(bge_sims) + minmax_normalize(e5_sims)
simple_time = time.time() - t0
simple_ranks = gold_rank_from_sims(fused_simple, gold_idx)
simple_metrics = compute_metrics(simple_ranks)
print(f"\n=== Simple normalized-score fusion (computed in {simple_time:.2f}s) ===")
print(json.dumps(simple_metrics, indent=2))

# ---------------- Final comparison table ----------------
print("\n" + "=" * 90)
print("FINAL COMPARISON")
print("=" * 90)
print(f"{'Metric':<15}{'BGE-M3':>15}{'E5-base-v2':>15}{'RRF fusion':>15}{'Score fusion':>15}")
for key in ["recall_at_1","recall_at_3","recall_at_5","recall_at_10","recall_at_20",
            "recall_at_50","recall_at_100","ndcg_at_10","mrr_at_10"]:
    print(f"{key:<15}{bge_metrics[key]:>15.4f}{e5_metrics[key]:>15.4f}{rrf_metrics[key]:>15.4f}{simple_metrics[key]:>15.4f}")

results_all = {
    "bge_m3": bge_metrics,
    "e5_base_v2": e5_metrics,
    "rrf_fusion": rrf_metrics,
    "simple_score_fusion": simple_metrics,
    "overlap_at_100": {
        "found_by_both": both, "found_only_bge": only_bge,
        "found_only_e5": only_e5, "found_by_neither": neither,
        "union_recall_100": union_recall_100,
    },
    "timing_seconds": {
        "bge_corpus_embed": bge_corpus_time, "bge_query_embed": bge_query_time,
        "e5_corpus_embed": e5_corpus_time, "e5_query_embed": e5_query_time,
        "rrf_fusion_compute": rrf_time, "simple_fusion_compute": simple_time,
    },
}
with open("experiment1_fusion_results.json", "w") as f:
    json.dump(results_all, f, indent=2)

print("\nSaved full results to: experiment1_fusion_results.json")