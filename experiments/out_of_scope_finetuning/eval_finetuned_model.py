"""
Official evaluation of the fine-tuned BGE-M3 on the untouched AppsRetrieval
test split -- same mteb.evaluate() pattern as every other model we've scored,
just pointed at the local fine-tuned checkpoint instead of the HF hub.

Run:  python eval_finetuned_model.py
"""

import json
import torch
import mteb
from sentence_transformers import SentenceTransformer

MODEL_PATH = "./finetuned-bge-m3-v1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading fine-tuned model from {MODEL_PATH}...")
model = SentenceTransformer(MODEL_PATH, device=DEVICE)
model.max_seq_length = 1536  # match the eval-time seq length used for the pretrained baseline

task = mteb.get_task("AppsRetrieval")
result = mteb.evaluate(model, [task], encode_kwargs={"batch_size": 8}, cache=None)

task_result = list(result.task_results)[0]
result_dict = task_result.to_dict()

with open("finetuned_bge_m3_results.json", "w") as f:
    json.dump(result_dict, f, indent=2, default=str)

scores = result_dict["scores"]["test"][0]
print("\n" + "=" * 70)
print("FINE-TUNED BGE-M3 -- OFFICIAL TEST SPLIT RESULTS")
print("=" * 70)
print(json.dumps(result_dict, indent=2, default=str))

print("\n" + "=" * 90)
print("COMPARISON AGAINST EVERYTHING TRIED SO FAR")
print("=" * 90)
print(f"{'Model':<35}{'NDCG@10':>12}{'MRR@10':>12}{'Recall@100':>14}")
print(f"{'all-MiniLM-L6-v2 (day 1)':<35}{0.0660:>12.4f}{0.0558:>12.4f}{'--':>14}")
print(f"{'BGE-M3 pretrained (baseline)':<35}{0.1475:>12.4f}{0.1277:>12.4f}{0.4295:>14.4f}")
print(f"{'BGE-M3 + E5 fusion (banked best)':<35}{0.1761:>12.4f}{0.1527:>12.4f}{0.4744:>14.4f}")
print(f"{'BGE-M3 FINE-TUNED (this run)':<35}{scores['ndcg_at_10']:>12.4f}{scores['mrr_at_10']:>12.4f}{scores['recall_at_100']:>14.4f}")

print("\nSaved to: finetuned_bge_m3_results.json")