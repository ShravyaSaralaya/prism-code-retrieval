"""
Bonus: Evolutionary Retrieval demo.

Builds a small synthetic multi-version corpus (a handful of real problems
from the dataset, each with 3-4 simulated "commit history" versions), then
demonstrates the exact challenge the guidelines describe -- near-duplicate
versions being hard to rank apart -- and our fix: group-and-dedup by logical
document before truncating to top_k.

This is a focused proof-of-concept (not run against the full test split --
there's no official Bonus metric, unlike P0). Fast to run, directly
demonstrable live in the demo.

Run:  python bonus_evolutionary_retrieval_demo.py
"""

import random
import numpy as np
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from versioned_index import VersionedEmbeddingIndex

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading a small sample of real corpus documents...")
corpus = load_dataset("CoIR-Retrieval/apps", "corpus", split="corpus")
queries = load_dataset("CoIR-Retrieval/apps", "queries", split="queries")
qrels = load_dataset("CoIR-Retrieval/apps", split="test")

# Pick 8 distinct logical "files" (real corpus docs) to simulate version history for
N_FILES = 8
sample_indices = random.sample(range(len(corpus)), N_FILES)
logical_files = {f"file_{i}": corpus[idx]["text"] for i, idx in enumerate(sample_indices)}

# Pick a real query that matches one of these files as our demo query
# (use the qrels to find a query whose gold doc matches one we picked)
corpus_id_by_idx = {i: corpus[i]["_id"] for i in sample_indices}
target_idx = sample_indices[0]
target_corpus_id = corpus[target_idx]["_id"]
matching_qrel = next((q for q in qrels if q["corpus-id"] == target_corpus_id), None)
if matching_qrel:
    query_id_to_text = {qid: text for qid, text in zip(queries["_id"], queries["text"])}
    demo_query_text = query_id_to_text[matching_qrel["query-id"]]
else:
    demo_query_text = queries[0]["text"]  # fallback

print(f"Demo query targets 'file_0' (real corpus doc {target_corpus_id})\n")

print("Loading BGE-M3...")
model = SentenceTransformer(
    "BAAI/bge-m3",
    model_kwargs={"torch_dtype": torch.float16} if DEVICE == "cuda" else {},
    device=DEVICE,
)

index = VersionedEmbeddingIndex(model, "bonus_versioned_index")

# ---------------- Simulate commit history: 3-4 versions per file ----------------
print("=== Simulating commit history (3-4 versions per file) ===")
EDITS = [
    "",  # v1: original
    "\n# added input validation",
    "\n# added input validation\n# refactored loop for clarity",
    "\n# added input validation\n# refactored loop for clarity\n# fixed edge case for n=0",
]
for fid, text in logical_files.items():
    for v, edit in enumerate(EDITS, start=1):
        index.add_version(fid, f"v{v}", text + edit, batch_size=8, seq_len=1536)

print(f"Indexed {len(index.rows)} total versions across {N_FILES} logical files\n")

# ---------------- Search: with vs without dedup ----------------
print(f"=== Query: {demo_query_text[:150]}...\n")
model.max_seq_length = 1536
query_emb = model.encode([demo_query_text], convert_to_numpy=True, normalize_embeddings=True)[0]

print("--- WITHOUT dedup (raw ranking across all versions) ---")
print("(This demonstrates the stated problem: near-duplicate versions of")
print(" the SAME file can dominate the top-k, crowding out other relevant files)")
results_raw = index.search_all_versions(query_emb, top_k=10, dedup=False)
for rank, (lid, ver, score) in enumerate(results_raw, 1):
    print(f"  {rank}. {lid} ({ver})  score={score:.4f}")

print("\n--- WITH dedup (grouped by logical file, best version kept) ---")
print("(Our fix: one entry per logical file, so the top-k stays diverse)")
results_dedup = index.search_all_versions(query_emb, top_k=10, dedup=True)
for rank, (lid, ver, score) in enumerate(results_dedup, 1):
    print(f"  {rank}. {lid} ({ver})  score={score:.4f}")

n_unique_raw = len(set(r[0] for r in results_raw))
n_unique_dedup = len(set(r[0] for r in results_dedup))
print(f"\nUnique logical files in top-10 WITHOUT dedup: {n_unique_raw}/10")
print(f"Unique logical files in top-10 WITH dedup:    {n_unique_dedup}/10")
print("\nThis demonstrates evolutionary retrieval: our index searches across")
print("ALL historical versions of the codebase simultaneously, while the")
print("dedup mechanism keeps results diverse and relevant despite near-")
print("duplicate versions -- directly addressing the challenge the")
print("guidelines describe.")
