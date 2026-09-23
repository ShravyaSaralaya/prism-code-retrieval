"""
Step 6 (fixed): Peek at the CoIR 'apps' retrieval dataset.

Your previous run already downloaded and cached everything via MTEB's loader
(MTEB v2 changed its internal task.queries/task.corpus API, which is why that
attribute access failed -- but the underlying data is already on your disk).
So instead of poking at MTEB's internal task object, we load the same
HuggingFace dataset directly, which is stable across mteb versions.

Run:  python peek_dataset.py
"""

from datasets import load_dataset

print("Loading corpus (code snippets)...")
corpus = load_dataset("CoIR-Retrieval/apps", "corpus", split="corpus")

print("Loading queries (natural language questions)...")
queries = load_dataset("CoIR-Retrieval/apps", "queries", split="queries")

print("Loading qrels (relevance labels, test split)...")
qrels = load_dataset("CoIR-Retrieval/apps", split="test")

print("\n" + "=" * 60)
print(f"Number of corpus snippets: {len(corpus)}")
print(f"Number of queries: {len(queries)}")
print(f"Number of qrel (relevance) entries: {len(qrels)}")
print("=" * 60)

print("\n--- Column names ---")
print("corpus columns :", corpus.column_names)
print("queries columns:", queries.column_names)
print("qrels columns  :", qrels.column_names)

print("\n--- Example corpus entry #0 ---")
print(corpus[0])

print("\n--- Example query entry #0 ---")
print(queries[0])

print("\n--- Example qrel entry #0 (which query maps to which correct snippet) ---")
print(qrels[0])

# Rough size stats, useful for deciding chunking/model context length later
def get_text(entry):
    # be tolerant of either "text" or a combination of "title" + "text"
    if "text" in entry:
        t = entry.get("text", "") or ""
        if "title" in entry and entry.get("title"):
            t = entry["title"] + " " + t
        return t
    return str(entry)

query_lens = [len(get_text(q)) for q in queries.select(range(min(200, len(queries))))]
corpus_lens = [len(get_text(c)) for c in corpus.select(range(min(200, len(corpus))))]

print("\n--- Rough character-length stats (first 200 samples) ---")
print(f"Query length   -> min: {min(query_lens)}, max: {max(query_lens)}, avg: {sum(query_lens)//len(query_lens)}")
print(f"Snippet length -> min: {min(corpus_lens)}, max: {max(corpus_lens)}, avg: {sum(corpus_lens)//len(corpus_lens)}")
