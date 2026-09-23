"""
Phase 1, third attempt (fixed): BAAI/bge-m3, tuned for limited GPU VRAM.

Model loaded fine last time -- it just ran out of GPU memory during encoding,
because 8192-token attention is memory-hungry and the default batch size
assumes a lot of VRAM. Fixes applied:
  1. Cap max_seq_length to 1536 tokens (comfortably covers our longest
     queries at ~1200 tokens; no need for the full 8192 ceiling here).
  2. Use fp16 on GPU to roughly halve memory use.
  3. Pass a small batch_size through encode_kwargs.

If this STILL runs out of memory, lower batch_size further (try 4, then 2)
before giving up on the GPU.

Run:  python try_bge_m3.py
"""

import torch
import mteb
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-m3"

print(f"Loading {MODEL_NAME}...")
model = SentenceTransformer(
    MODEL_NAME,
    model_kwargs={"torch_dtype": torch.float16} if torch.cuda.is_available() else {},
)
model.max_seq_length = 1536  # our longest query is ~1200 tokens; no need for 8192

tasks = mteb.get_tasks(tasks=["AppsRetrieval"])
output_folder = f"results/{MODEL_NAME.replace('/', '__')}"

evaluation = mteb.MTEB(tasks=tasks)
result = evaluation.run(
    model,
    output_folder=output_folder,
    encode_kwargs={"batch_size": 8},
)

print("\n" + "=" * 60)
print("RESULT")
print("=" * 60)
try:
    main_score = result[0].scores["test"][0]["main_score"]
    print(f"NDCG@10 for {MODEL_NAME}: {main_score}")
except Exception:
    print(f"Could not read score directly -- check the JSON under {output_folder}/")

print("\nFor comparison so far:")
print(f"{'all-MiniLM-L6-v2 (baseline)':<40} NDCG@10=0.0660  MRR@10=0.0558")
print(f"{'intfloat/e5-base-v2':<40} NDCG@10=0.1152  MRR@10=0.0988")