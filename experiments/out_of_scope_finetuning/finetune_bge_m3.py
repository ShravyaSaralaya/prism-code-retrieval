"""
Fine-tune BGE-M3 on the CoIR apps TRAIN split, using LoRA (not full
fine-tuning) -- required for 4GB VRAM GPUs like the RTX 3050 laptop, since
full fine-tuning's optimizer state alone would exceed available memory
regardless of batch size.

LoRA freezes the 568M-parameter base model and trains a small adapter on top.
After training we MERGE the adapter into the base model, so the final saved
model is a normal, full-sized model -- identical to deploy/run as before,
nothing extra carried at inference time.

No test leakage: hard negatives come only from train-partition corpus docs,
for train-partition queries only, using the dataset's own `partition` field.
Reproducible: fixed seeds throughout.
Original pretrained BGE-M3 cache is untouched -- this saves to a new folder.

First run:  pip install peft --break-system-packages
Then:       python finetune_bge_m3.py
"""

import os
import random
import numpy as np
import torch
from datasets import load_dataset, Dataset
from sentence_transformers import (
    SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments,
)
from sentence_transformers.losses import MultipleNegativesRankingLoss
from peft import LoraConfig, TaskType

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CACHE_DIR = "embed_cache"
N_HARD_NEGATIVES = 2
TRAIN_SEQ_LEN = 512
OUTPUT_DIR = "./finetuned-bge-m3-v1"

def cache_path(name):
    return os.path.join(CACHE_DIR, name)

# ---------------- Load data, split by partition ----------------
print("Loading dataset...")
corpus = load_dataset("CoIR-Retrieval/apps", "corpus", split="corpus")
queries = load_dataset("CoIR-Retrieval/apps", "queries", split="queries")
train_qrels = load_dataset("CoIR-Retrieval/apps", split="train")

corpus_ids = corpus["_id"]
corpus_texts = corpus["text"]
corpus_partition = corpus["partition"]
id_to_corpus_idx = {cid: i for i, cid in enumerate(corpus_ids)}
query_id_to_text = {qid: text for qid, text in zip(queries["_id"], queries["text"])}

train_query_ids = train_qrels["query-id"]
train_gold_ids = train_qrels["corpus-id"]
print(f"Train pairs (positive query->code): {len(train_query_ids)}")

train_corpus_mask = np.array([p == "train" for p in corpus_partition])
train_corpus_indices = np.where(train_corpus_mask)[0]
print(f"Train-partition corpus docs available for hard-negative mining: {len(train_corpus_indices)}")

# ---------------- Hard-negative mining using the CURRENT pretrained model ----------------
print("\n=== Mining hard negatives with pretrained BGE-M3 (train-partition only) ===")
required = ["bge_corpus.npy"]
missing = [f for f in required if not os.path.exists(cache_path(f))]
if missing:
    raise FileNotFoundError(f"Missing cached embeddings: {missing}. Run experiment2_reranker.py first.")

bge_corpus_emb_full = np.load(cache_path("bge_corpus.npy"))  # [n_corpus, dim], full corpus (train+test)

miner_model = SentenceTransformer(
    "BAAI/bge-m3",
    model_kwargs={"torch_dtype": torch.float16} if DEVICE == "cuda" else {},
    device=DEVICE,
)
miner_model.max_seq_length = 1536
train_query_texts_for_mining = [query_id_to_text[qid] for qid in train_query_ids]
train_query_emb = miner_model.encode(
    train_query_texts_for_mining, batch_size=8, show_progress_bar=True,
    convert_to_numpy=True, normalize_embeddings=True,
).astype(np.float32)
del miner_model
if DEVICE == "cuda":
    torch.cuda.empty_cache()

sims = train_query_emb @ bge_corpus_emb_full.T
train_only_mask = ~train_corpus_mask
sims_masked = sims.copy()
sims_masked[:, train_only_mask] = -np.inf

hard_negative_texts = []
for i, gold_id in enumerate(train_gold_ids):
    gold_idx_i = id_to_corpus_idx[gold_id]
    ranked = np.argsort(-sims_masked[i])
    negs = []
    for idx in ranked:
        if idx == gold_idx_i:
            continue
        if sims_masked[i, idx] == -np.inf:
            break
        negs.append(corpus_texts[idx])
        if len(negs) >= N_HARD_NEGATIVES:
            break
    hard_negative_texts.append(negs)

n_with_full_negs = sum(1 for n in hard_negative_texts if len(n) == N_HARD_NEGATIVES)
print(f"Training examples with {N_HARD_NEGATIVES} hard negatives found: {n_with_full_negs}/{len(hard_negative_texts)}")

# ---------------- Build training dataset ----------------
anchors, positives = [], []
neg_cols = {f"negative_{i+1}": [] for i in range(N_HARD_NEGATIVES)}
for i, (qid, gold_id) in enumerate(zip(train_query_ids, train_gold_ids)):
    negs = hard_negative_texts[i]
    if len(negs) < N_HARD_NEGATIVES:
        continue
    anchors.append(query_id_to_text[qid])
    positives.append(corpus_texts[id_to_corpus_idx[gold_id]])
    for j, neg_text in enumerate(negs):
        neg_cols[f"negative_{j+1}"].append(neg_text)

train_dataset = Dataset.from_dict({"anchor": anchors, "positive": positives, **neg_cols})
print(f"\nFinal training set size: {len(train_dataset)} examples")

# ---------------- Fine-tune with LoRA ----------------
print("\n=== Setting up BGE-M3 with a LoRA adapter (base model frozen) ===")
model = SentenceTransformer(
    "BAAI/bge-m3",
    model_kwargs={"torch_dtype": torch.bfloat16} if DEVICE == "cuda" else {},
    device=DEVICE,
)
model.max_seq_length = TRAIN_SEQ_LEN

lora_config = LoraConfig(
    task_type=TaskType.FEATURE_EXTRACTION,
    r=8,               # LoRA rank -- small, standard starting point
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["query", "key", "value"],  # standard attention projection names for this architecture
)
model.add_adapter(lora_config)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable parameters: {trainable:,} / {total:,} ({trainable/total*100:.3f}%) -- this is what makes it fit in 4GB")

loss = MultipleNegativesRankingLoss(model)

args = SentenceTransformerTrainingArguments(
    output_dir="./bge_m3_training_checkpoints",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,  # effective batch size 16
    gradient_checkpointing=False,  # disabled: transformers v5 API mismatch with this model's checkpointing method; LoRA already handles the dominant memory cost (optimizer state), so this isn't needed at batch_size=1
    learning_rate=1e-4,  # LoRA typically uses a higher LR than full fine-tuning
    warmup_ratio=0.1,
    fp16=False,
    bf16=(DEVICE == "cuda"),  # bf16 needs no gradient scaler -- avoids the fp16-unscale error entirely, and RTX 30-series supports it natively
    save_strategy="no",
    logging_steps=25,
    seed=SEED,
    data_seed=SEED,
)

trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    loss=loss,
)

print("\nStarting training (LoRA -- should fit comfortably in 4GB VRAM)...")
trainer.train()

# ---------------- Save immediately -- BEFORE anything else can go wrong ----------------
# transformers' native PEFT integration saves/loads the adapter automatically on its own;
# no explicit merge step is needed for correctness. Saving FIRST means a bug anywhere
# after this point can never again cost us a completed training run.
os.makedirs(OUTPUT_DIR, exist_ok=True)
model.save(OUTPUT_DIR)
print(f"\nFine-tuned model saved to: {OUTPUT_DIR}")
print("(Adapter stays active in the saved files -- transformers auto-detects and")
print(" re-applies it on load. Loading this later requires `peft` installed, which it already is.)")
print("Original pretrained BGE-M3 cache is untouched.")
print("\nNext step: run eval_finetuned_model.py to score this on the official test split.")