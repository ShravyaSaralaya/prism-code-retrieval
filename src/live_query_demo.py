"""
Live query demo -- fixes two gaps against the hackathon guidelines:

  1. "You should show the responses for a given query" (not just numbers) --
     this answers real queries live and prints actual ranked code snippets.
  2. "how fast your solution is" -- reports REAL measured latency, not an
     assumption. Query-time defaults to CPU (not GPU) to actually prove the
     "runs on CPU with minimal GPU" claim, rather than just asserting it.

Design: index-BUILDING (one-time, ~4 min for the full corpus) can use GPU for
speed -- that's a reasonable one-time cost. Index-QUERYING (what you'd
actually demo live, and what organizers care about) defaults to CPU.

First run:  builds the full persisted index (one-time, prefers GPU if available)
Every run after: loads the index instantly, only embeds your query fresh

Usage:
  python live_query_demo.py                              # runs 3 built-in example queries on CPU
  python live_query_demo.py --query "your query here"     # your own query
  python live_query_demo.py --device cuda                 # query on GPU instead, for comparison
"""

import argparse
import time
import numpy as np
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from incremental_index import IncrementalEmbeddingIndex

EXAMPLE_QUERIES = [
    "How is the input parsed before the main processing loop?",
    "Find code that checks whether a number is prime.",
    "Which snippet performs a breadth-first search on a graph?",
]


def minmax(sims):
    mn, mx = sims.min(), sims.max()
    return (sims - mn) / (mx - mn + 1e-8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, default=None,
                         help="Run one custom query instead of the built-in examples")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                         help="Device for QUERY-TIME inference (default: cpu -- this is the number that matters)")
    parser.add_argument("--build_device", type=str, default=None, choices=["cpu", "cuda"],
                         help="Device for the one-time index BUILD, if not already built (default: cuda if available)")
    parser.add_argument("--top_k", type=int, default=10)
    args = parser.parse_args()

    query_device = args.device
    build_device = args.build_device or ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Query-time device: {query_device}  (this is the number we report)")
    print(f"Index-build device (one-time only, if needed): {build_device}\n")

    print("Loading corpus (needed for result previews either way)...")
    corpus = load_dataset("CoIR-Retrieval/apps", "corpus", split="corpus")
    doc_id_to_text = {c["_id"]: c["text"] for c in corpus}

    # Load models on the QUERY device -- plain fp32, no GPU-specific dtype tricks,
    # so this is exactly what would run on a CPU-only machine.
    print(f"Loading models on '{query_device}' (plain fp32 -- CPU-safe)...")
    bge_model = SentenceTransformer("BAAI/bge-m3", device=query_device)
    bge_model.max_seq_length = 1536
    e5_model = SentenceTransformer("intfloat/e5-base-v2", device=query_device)
    e5_model.max_seq_length = 512

    bge_index = IncrementalEmbeddingIndex(bge_model, "live_demo_index_bge", passage_prefix="")
    e5_index = IncrementalEmbeddingIndex(e5_model, "live_demo_index_e5", passage_prefix="passage: ")

    if bge_index.embeddings is None or len(bge_index.doc_ids) == 0:
        print(f"\nNo persisted index found -- building it now on '{build_device}' (one-time, ~4 minutes)...")
        if build_device != query_device:
            # Build with separate model instances on the build device for speed,
            # then reload the query-side models above will use for actual queries.
            build_bge = SentenceTransformer(
                "BAAI/bge-m3",
                model_kwargs={"torch_dtype": torch.float16} if build_device == "cuda" else {},
                device=build_device,
            )
            build_e5 = SentenceTransformer("intfloat/e5-base-v2", device=build_device)
            build_bge_index = IncrementalEmbeddingIndex(build_bge, "live_demo_index_bge", passage_prefix="")
            build_e5_index = IncrementalEmbeddingIndex(build_e5, "live_demo_index_e5", passage_prefix="passage: ")
            t0 = time.time()
            build_bge_index.sync(doc_id_to_text, batch_size=8, seq_len=1536, show_progress=True)
            build_e5_index.sync(doc_id_to_text, batch_size=32, seq_len=512, show_progress=True)
            print(f"Index built in {time.time()-t0:.1f}s.\n")
            del build_bge, build_e5
            if build_device == "cuda":
                torch.cuda.empty_cache()
            # reload the persisted index into our query-device index objects
            bge_index._load_if_exists()
            e5_index._load_if_exists()
        else:
            t0 = time.time()
            bge_index.sync(doc_id_to_text, batch_size=8, seq_len=1536, show_progress=True)
            e5_index.sync(doc_id_to_text, batch_size=32, seq_len=512, show_progress=True)
            print(f"Index built in {time.time()-t0:.1f}s.\n")
        print("Saved to disk -- future runs load this instantly, no rebuild needed.\n")
    else:
        print(f"Loaded persisted index: {len(bge_index.doc_ids)} documents (instant, no re-embedding).\n")

    print("Warming up (first CPU inference call pays a one-time cost unrelated to steady-state speed)...")
    _ = bge_model.encode(["warmup"], convert_to_numpy=True, normalize_embeddings=True)
    _ = e5_model.encode(["query: warmup"], convert_to_numpy=True, normalize_embeddings=True)
    print("Warm-up done -- timings below reflect real steady-state performance.\n")

    def answer_query(query_text, top_k):
        t0 = time.time()
        bge_q = bge_model.encode([query_text], convert_to_numpy=True, normalize_embeddings=True)[0]
        e5_q = e5_model.encode([f"query: {query_text}"], convert_to_numpy=True, normalize_embeddings=True)[0]
        t_embed = (time.time() - t0) * 1000

        t1 = time.time()
        bge_sims = bge_index.embeddings @ bge_q
        e5_sims = e5_index.embeddings @ e5_q
        fused = minmax(bge_sims) + minmax(e5_sims)
        top_idx = np.argsort(-fused)[:top_k]
        t_score = (time.time() - t1) * 1000

        total_ms = t_embed + t_score
        print(f"\nQUERY: {query_text}")
        print(f"[embed: {t_embed:.1f}ms | score vs {len(bge_index.doc_ids)} docs: {t_score:.1f}ms | "
              f"TOTAL: {total_ms:.1f}ms | device={query_device}]")
        print("-" * 78)
        for rank, i in enumerate(top_idx, 1):
            doc_id = bge_index.doc_ids[i]
            preview = doc_id_to_text.get(doc_id, "")[:150].replace("\n", " ")
            print(f"  {rank}. [{doc_id}] score={fused[i]:.4f}")
            print(f"      {preview}...")
        return total_ms

    if args.query:
        answer_query(args.query, args.top_k)
    else:
        print(f"Running {len(EXAMPLE_QUERIES)} built-in example queries "
              f"(pass --query \"...\" for your own):")
        times = [answer_query(q, args.top_k) for q in EXAMPLE_QUERIES]
        print(f"\n{'='*78}\nAverage per-query latency on {query_device.upper()}: "
              f"{sum(times)/len(times):.1f}ms across {len(times)} queries\n{'='*78}")


if __name__ == "__main__":
    main()
