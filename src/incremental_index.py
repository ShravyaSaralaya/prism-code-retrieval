"""
P1: Incremental embedding index -- supports fast re-indexing when the
codebase changes, instead of re-embedding the entire corpus from scratch.

Core idea: store each doc's embedding alongside a content hash. On sync(),
diff the new corpus state against stored hashes:
  - unchanged docs -> reuse cached embedding, skip re-embedding entirely
  - new/changed docs -> embed only these
  - removed docs -> drop from the index
This is the general mechanism; reused for both BGE-M3 and E5 in our actual
fusion pipeline (see p1_demo.py).
"""

import os
import json
import time
import hashlib
import numpy as np


class IncrementalEmbeddingIndex:
    def __init__(self, model, index_dir, passage_prefix=""):
        self.model = model
        self.index_dir = index_dir
        self.passage_prefix = passage_prefix
        os.makedirs(index_dir, exist_ok=True)
        self.meta_path = os.path.join(index_dir, "metadata.json")
        self.emb_path = os.path.join(index_dir, "embeddings.npy")
        self.doc_ids = []
        self.hash_by_id = {}
        self.embeddings = None
        self._load_if_exists()

    def _load_if_exists(self):
        if os.path.exists(self.meta_path) and os.path.exists(self.emb_path):
            with open(self.meta_path) as f:
                meta = json.load(f)
            self.doc_ids = meta["doc_ids"]
            self.hash_by_id = meta["hash_by_id"]
            self.embeddings = np.load(self.emb_path)

    def _save(self):
        with open(self.meta_path, "w") as f:
            json.dump({"doc_ids": self.doc_ids, "hash_by_id": self.hash_by_id}, f)
        np.save(self.emb_path, self.embeddings)

    @staticmethod
    def _hash(text):
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def sync(self, docs, batch_size=8, seq_len=1536, show_progress=False):
        """
        docs: {doc_id: text} representing the CURRENT full state of the corpus
        (this version/commit). Diffs against the stored index; only embeds
        what actually changed. Returns stats including elapsed time.
        """
        t0 = time.time()
        new_hashes = {did: self._hash(text) for did, text in docs.items()}

        added = [did for did in new_hashes if did not in self.hash_by_id]
        changed = [did for did in new_hashes
                   if did in self.hash_by_id and self.hash_by_id[did] != new_hashes[did]]
        removed = [did for did in self.hash_by_id if did not in new_hashes]
        unchanged_count = len(new_hashes) - len(added) - len(changed)

        to_embed_ids = added + changed
        to_embed_texts = [self.passage_prefix + docs[did] for did in to_embed_ids]

        dim = self.embeddings.shape[1] if self.embeddings is not None else None

        if to_embed_texts:
            self.model.max_seq_length = seq_len
            new_embs = self.model.encode(
                to_embed_texts, batch_size=batch_size, show_progress_bar=show_progress,
                convert_to_numpy=True, normalize_embeddings=True,
            ).astype(np.float32)
            dim = new_embs.shape[1]
        else:
            new_embs = np.zeros((0, dim or 1), dtype=np.float32)

        changed_set = set(changed)
        keep_ids = [did for did in self.doc_ids if did in new_hashes and did not in changed_set]
        old_index = {did: i for i, did in enumerate(self.doc_ids)}
        keep_rows = [old_index[did] for did in keep_ids]
        kept_embs = self.embeddings[keep_rows] if (self.embeddings is not None and keep_rows) \
            else np.zeros((0, dim or 1), dtype=np.float32)

        self.doc_ids = keep_ids + to_embed_ids
        self.embeddings = np.vstack([kept_embs, new_embs]) if (len(kept_embs) or len(new_embs)) else kept_embs
        self.hash_by_id = new_hashes
        self._save()

        elapsed = time.time() - t0
        return {
            "added": len(added), "changed": len(changed), "removed": len(removed),
            "unchanged": unchanged_count, "re_embedded": len(to_embed_ids),
            "total_docs": len(self.doc_ids), "elapsed_seconds": elapsed,
        }