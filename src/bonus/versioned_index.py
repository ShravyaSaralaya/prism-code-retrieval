"""
Bonus: Evolutionary Retrieval -- an extension of P1's incremental index.

Where incremental_index.py (P1) keeps only the LATEST version of each
document (fast updates, but old versions are discarded), this index keeps
the FULL VERSION HISTORY per logical document, and supports searching across
ALL versions at once.

The stated challenge: near-duplicate versions of the same file are hard to
rank apart, and can crowd out genuinely different relevant files in the
results. Our mechanism: group hits by logical document ID, keep only the
best-scoring version per group before truncating to top_k. This is
demonstrated concretely in bonus_evolutionary_retrieval_demo.py.
"""

import os
import json
import hashlib
import numpy as np


class VersionedEmbeddingIndex:
    def __init__(self, model, index_dir, passage_prefix=""):
        self.model = model
        self.index_dir = index_dir
        self.passage_prefix = passage_prefix
        os.makedirs(index_dir, exist_ok=True)
        self.meta_path = os.path.join(index_dir, "versioned_metadata.json")
        self.emb_path = os.path.join(index_dir, "versioned_embeddings.npy")
        # each row in self.embeddings corresponds to one entry in self.rows:
        # {"logical_id": ..., "version": ..., "hash": ...}
        self.rows = []
        self.embeddings = None
        self._load_if_exists()

    def _load_if_exists(self):
        if os.path.exists(self.meta_path) and os.path.exists(self.emb_path):
            with open(self.meta_path) as f:
                self.rows = json.load(f)
            self.embeddings = np.load(self.emb_path)

    def _save(self):
        with open(self.meta_path, "w") as f:
            json.dump(self.rows, f, indent=2)
        np.save(self.emb_path, self.embeddings)

    @staticmethod
    def _hash(text):
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def add_version(self, logical_id, version_label, text, batch_size=8, seq_len=1536):
        """Add a new version of a logical document. Keeps all prior versions.
        No-ops if this exact content already exists as the latest version
        (avoids indexing a no-op commit)."""
        h = self._hash(text)
        existing_versions = [r for r in self.rows if r["logical_id"] == logical_id]
        if existing_versions and existing_versions[-1]["hash"] == h:
            return False  # identical to latest version, skip

        self.model.max_seq_length = seq_len
        emb = self.model.encode(
            [self.passage_prefix + text], batch_size=batch_size,
            show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True,
        ).astype(np.float32)

        self.rows.append({"logical_id": logical_id, "version": version_label, "hash": h})
        self.embeddings = np.vstack([self.embeddings, emb]) if self.embeddings is not None else emb
        return True

    def search_all_versions(self, query_emb, top_k=10, dedup=True):
        """
        query_emb: pre-computed query embedding (1D array), same model/space
        as this index. Returns a list of (logical_id, version, score) tuples.

        dedup=True: group by logical_id, keep only the best-scoring version
        per logical document -- this is the mechanism addressing the
        "near-duplicate versions are hard to rank" challenge.
        dedup=False: raw ranking across every stored version (shows the
        problem: near-duplicates of the same file crowding the results).
        """
        sims = self.embeddings @ query_emb  # [n_versions]
        order = np.argsort(-sims)

        if not dedup:
            return [(self.rows[i]["logical_id"], self.rows[i]["version"], float(sims[i]))
                    for i in order[:top_k]]

        seen_logical_ids = set()
        results = []
        for i in order:
            lid = self.rows[i]["logical_id"]
            if lid in seen_logical_ids:
                continue
            seen_logical_ids.add(lid)
            results.append((lid, self.rows[i]["version"], float(sims[i])))
            if len(results) >= top_k:
                break
        return results
