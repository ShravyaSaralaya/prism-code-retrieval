"""
BGE-M3 + query preprocessing, evaluated the official way (mteb.evaluate).

Key fix from the last two failures: AbsEncoder.encode() receives a DataLoader
of BATCHES (each batch a dict of lists, e.g. {"text": [8 strings], ...]}) --
NOT a flat list of strings. We must iterate it ourselves and do our own
batching into the underlying SentenceTransformer, and must NOT forward MTEB's
internal kwargs (task_metadata, hf_subset, hf_split) into
SentenceTransformer.encode(), which rejects unknown kwargs.

Run:  python run_bge_m3_preprocessed.py
"""

import re
import json
import numpy as np
import torch
import mteb
from mteb.models.abs_encoder import AbsEncoder
from sentence_transformers import SentenceTransformer

# ---- Conservative query preprocessing (validated in preview_preprocessing.py) ----
DASHED_HEADER_RE = re.compile(
    r'\n-{3,}\s*(Input|Output|Examples?|Note|Constraints?)\s*-{3,}',
    re.IGNORECASE
)
BARE_HEADER_RE = re.compile(
    r'^[ \t]*(Input|Output|Constraints?|Examples?|Note)[ \t]*\d*[ \t]*:?[ \t]*$',
    re.IGNORECASE | re.MULTILINE
)

def strip_io_boilerplate(text: str) -> str:
    candidates = []
    m = DASHED_HEADER_RE.search(text)
    if m:
        candidates.append(m.start())
    m = BARE_HEADER_RE.search(text)
    if m:
        candidates.append(m.start())
    if candidates:
        cut_at = min(candidates)
        narrative = text[:cut_at].strip()
        if len(narrative) >= 40:
            return narrative
    return text.strip()


class BGEM3PreprocessedEncoder(AbsEncoder):
    """Wraps BGE-M3. Applies query preprocessing only when prompt_type == 'query'."""

    def __init__(self, model_name="BAAI/bge-m3", max_seq_length=1536):
        self.model = SentenceTransformer(
            model_name,
            model_kwargs={"torch_dtype": torch.float16} if torch.cuda.is_available() else {},
        )
        self.model.max_seq_length = max_seq_length

    def encode(self, sentences, *, task_name=None, prompt_type=None, **kwargs):
        is_query = prompt_type is not None and getattr(prompt_type, "value", prompt_type) == "query"

        def batch_to_texts(batch):
            texts = batch["text"] if isinstance(batch, dict) else batch
            texts = list(texts) if isinstance(texts, (list, tuple)) else [texts]
            texts = [str(t) for t in texts]
            if is_query:
                texts = [strip_io_boilerplate(t) for t in texts]
            return texts

        all_embeddings = []
        # `sentences` is a DataLoader of batches -- iterate it, don't treat as a flat list.
        for batch in sentences:
            texts = batch_to_texts(batch)
            emb = self.model.encode(texts, batch_size=len(texts), show_progress_bar=False)
            all_embeddings.append(np.asarray(emb))

        return np.concatenate(all_embeddings, axis=0)


print("Loading BGE-M3 with query preprocessing wrapper...")
model = BGEM3PreprocessedEncoder()

task = mteb.get_task("AppsRetrieval")
result = mteb.evaluate(model, [task], encode_kwargs={"batch_size": 8}, cache=None)

task_result = list(result.task_results)[0]
result_dict = task_result.to_dict()

with open("bge_m3_preprocessed_results.json", "w") as f:
    json.dump(result_dict, f, indent=2, default=str)

print("\n" + "=" * 70)
print("FULL SCORES -- BGE-M3 + query preprocessing")
print("=" * 70)
print(json.dumps(result_dict, indent=2, default=str))

print("\n" + "=" * 70)
print("Saved to: bge_m3_preprocessed_results.json")
print("=" * 70)