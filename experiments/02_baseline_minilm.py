"""
Step 7: Run a stock baseline embedding model through MTEB's AppsRetrieval task,
end-to-end, producing the official results JSON.

Goal here is NOT a good score -- it's proving the full pipeline works:
    dataset -> embedding model -> MTEB scorer -> JSON output
so the "how do I even submit this" question is answered on day one.

Run:  python run_baseline_eval.py
"""

import mteb
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # small, fast, CPU-friendly baseline
OUTPUT_FOLDER = f"results/{MODEL_NAME.replace('/', '__')}"

print(f"Loading model: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME)

print("Loading AppsRetrieval task...")
tasks = mteb.get_tasks(tasks=["AppsRetrieval"])

print("Running evaluation (this embeds every query + every corpus snippet)...")
evaluation = mteb.MTEB(tasks=tasks)
results = evaluation.run(model, output_folder=OUTPUT_FOLDER)

print("\n" + "=" * 60)
print("DONE. Results object:")
print(results)
print("=" * 60)
print(f"\nFull results JSON written under: {OUTPUT_FOLDER}/")
print("Look for a file like: AppsRetrieval.json (or similar, nested by model/revision)")
print("Open it and check the 'ndcg_at_10' and 'mrr_at_10' (or similar) fields --")
print("that's the number you're trying to improve in later phases.")