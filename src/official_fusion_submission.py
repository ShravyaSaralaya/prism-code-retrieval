"""
OFFICIAL submission JSON generator: BGE-M3 + E5-base-v2 fusion, run through
mteb's real SearchProtocol interface -- produces the actual official JSON
required for submission, not our custom-script numbers.

Fine-tuning is OUT OF SCOPE per organizer guidance (received during the
hackathon). This fusion approach -- combining two off-the-shelf pretrained
models -- is explicitly permitted by the guidelines ("free to use the models
and methods available and combine them with your own methods").

Run:  python official_fusion_submission.py
"""

import json
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import mteb
from mteb.models.model_meta import ModelMeta

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def extract_id_text(data, id_keys=("_id", "id"), text_key="text", title_key="title"):
    """Defensively handle whatever shape mteb hands us (dict-of-lists or similar)."""
    if hasattr(data, "column_names"):  # a HF Dataset
        cols = data.column_names
        id_key = next((k for k in id_keys if k in cols), id_keys[0])
        ids = data[id_key]
        texts = data[text_key]
        titles = data[title_key] if title_key in cols else [""] * len(texts)
    else:
        id_key = next((k for k in id_keys if k in data), id_keys[0])
        ids = data[id_key]
        texts = data[text_key]
        titles = data.get(title_key, [""] * len(texts))
    combined = [f"{ti} {t}".strip() if ti else t for t, ti in zip(texts, titles)]
    return list(ids), combined

def minmax(sims):
    mn = sims.min(axis=1, keepdims=True)
    mx = sims.max(axis=1, keepdims=True)
    return (sims - mn) / (mx - mn + 1e-8)


class FusionSearchModel:
    """Implements mteb's SearchProtocol directly -- full custom control over
    retrieval logic, which we need since fusion can't be expressed as a
    single embedding vector (per-query min-max normalization depends on the
    whole corpus's similarity distribution, not just two fixed vectors)."""

    def __init__(self):
        print("Loading BGE-M3...")
        self.bge = SentenceTransformer(
            "BAAI/bge-m3",
            model_kwargs={"torch_dtype": torch.float16} if DEVICE == "cuda" else {},
            device=DEVICE,
        )
        self.bge.max_seq_length = 1536

        print("Loading E5-base-v2...")
        self.e5 = SentenceTransformer("intfloat/e5-base-v2", device=DEVICE)
        self.e5.max_seq_length = 512

        self._corpus_ids = None
        self._bge_corpus_emb = None
        self._e5_corpus_emb = None

    @property
    def mteb_model_meta(self):
        return ModelMeta(
            loader=None,
            name="prism-team/bge-m3-e5-fusion",
            revision="v1",
            release_date=None,
            languages=["eng-Latn"],
            n_parameters=678_000_000,
            memory_usage_mb=None,
            max_tokens=1536,
            embed_dim=1024,
            license=None,
            open_weights=True,
            public_training_code=None,
            public_training_data=None,
            framework=["Sentence Transformers"],
            similarity_fn_name="cosine",
            use_instructions=False,
            training_datasets=None,
        )

    def index(self, corpus, *, task_metadata=None, hf_split=None, hf_subset=None,
              encode_kwargs=None, num_proc=None):
        print("Indexing corpus (embedding with both models)...")
        ids, texts = extract_id_text(corpus)
        self._corpus_ids = ids
        self._bge_corpus_emb = self.bge.encode(
            texts, batch_size=8, show_progress_bar=True,
            convert_to_numpy=True, normalize_embeddings=True,
        ).astype(np.float32)
        self._e5_corpus_emb = self.e5.encode(
            [f"passage: {t}" for t in texts], batch_size=32, show_progress_bar=True,
            convert_to_numpy=True, normalize_embeddings=True,
        ).astype(np.float32)
        print(f"Indexed {len(ids)} documents.")

    def search(self, queries, *, task_metadata=None, hf_split=None, hf_subset=None,
               top_k=100, encode_kwargs=None, top_ranked=None, num_proc=None):
        qids, qtexts = extract_id_text(queries)
        print(f"Searching {len(qids)} queries (top_k={top_k})...")

        bge_q_emb = self.bge.encode(
            qtexts, batch_size=8, show_progress_bar=True,
            convert_to_numpy=True, normalize_embeddings=True,
        ).astype(np.float32)
        e5_q_emb = self.e5.encode(
            [f"query: {t}" for t in qtexts], batch_size=32, show_progress_bar=True,
            convert_to_numpy=True, normalize_embeddings=True,
        ).astype(np.float32)

        bge_sims = bge_q_emb @ self._bge_corpus_emb.T
        e5_sims = e5_q_emb @ self._e5_corpus_emb.T
        fused = minmax(bge_sims) + minmax(e5_sims)

        results = {}
        for i, qid in enumerate(qids):
            row = fused[i]
            top_idx = np.argsort(-row)[:top_k]
            results[qid] = {self._corpus_ids[j]: float(row[j]) for j in top_idx}
        return results


print("Setting up fusion search model...")
model = FusionSearchModel()

task = mteb.get_task("AppsRetrieval")
result = mteb.evaluate(model, [task], cache=None)

task_result = list(result.task_results)[0]
result_dict = task_result.to_dict()

with open("OFFICIAL_fusion_submission_results.json", "w") as f:
    json.dump(result_dict, f, indent=2, default=str)

print("\n" + "=" * 70)
print("OFFICIAL SUBMISSION JSON -- BGE-M3 + E5 fusion")
print("=" * 70)
print(json.dumps(result_dict, indent=2, default=str))
print("\nSaved to: OFFICIAL_fusion_submission_results.json")
print("*** THIS is the file to submit / attach to the GitHub release. ***")