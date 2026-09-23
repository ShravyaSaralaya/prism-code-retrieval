"""
Preview the query preprocessing transformation on real examples BEFORE running
the full (expensive) evaluation. No model loading needed -- just inspecting text.

Conservative design:
  - We only ever CUT text at a recognized "-----Input-----" / "-----Output-----" /
    "-----Examples-----" / "-----Note-----" / "-----Constraints-----" style marker
    (this is APPS's actual format, confirmed from real samples we already saw).
  - We keep everything BEFORE the first such marker (the narrative/problem
    description -- this is where the distinguishing signal lives).
  - Safety net: if the kept narrative ends up suspiciously short (<40 chars),
    we assume the format didn't match as expected and fall back to the FULL
    ORIGINAL text, unmodified. We never blindly truncate something we don't
    recognize.

Run:  python preview_preprocessing.py
"""

import re
from datasets import load_dataset

# Pattern A: Codeforces-style dashed headers, e.g. "-----Input-----"
DASHED_HEADER_RE = re.compile(
    r'\n-{3,}\s*(Input|Output|Examples?|Note|Constraints?)\s*-{3,}',
    re.IGNORECASE
)

# Pattern B: plain "bare line" headers common on LeetCode/CodeWars/AtCoder-style
# problems, e.g. a line that is JUST "Input", "Constraints:", "Example 1:".
# Anchored to start-of-line and end-of-line (via MULTILINE) so it won't match
# the word "input" appearing mid-sentence in the narrative.
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
        cut_at = min(candidates)  # cut at the EARLIEST recognized marker
        narrative = text[:cut_at].strip()
        if len(narrative) >= 40:  # safety net -- never over-truncate
            return narrative
    return text.strip()  # fallback: format not recognized, keep everything


print("Loading queries...")
queries = load_dataset("CoIR-Retrieval/apps", "queries", split="queries")

# Show the transformation on the first 3 queries, in full, so you can
# actually verify nothing important got cut.
for i in range(3):
    original = queries[i]["text"]
    processed = strip_io_boilerplate(original)
    print("\n" + "=" * 70)
    print(f"QUERY #{i} -- id={queries[i]['_id']}")
    print("=" * 70)
    print(f"Original length: {len(original)} chars")
    print(f"Processed length: {len(processed)} chars  ({len(processed)/len(original)*100:.0f}% kept)")
    print("\n--- ORIGINAL (last 300 chars, to show what gets cut) ---")
    print(original[-300:])
    print("\n--- PROCESSED (full) ---")
    print(processed)

# Aggregate stats across a larger sample, so we know the overall effect
# before committing to a full eval run.
sample = queries.select(range(min(500, len(queries))))
orig_lens = [len(q["text"]) for q in sample]
proc_lens = [len(strip_io_boilerplate(q["text"])) for q in sample]
unchanged_count = sum(1 for o, p in zip(orig_lens, proc_lens) if o == p)

print("\n" + "=" * 70)
print(f"AGGREGATE STATS (first {len(sample)} queries)")
print("=" * 70)
print(f"Avg original length : {sum(orig_lens)//len(orig_lens)} chars")
print(f"Avg processed length: {sum(proc_lens)//len(proc_lens)} chars")
print(f"Avg reduction        : {(1 - sum(proc_lens)/sum(orig_lens))*100:.1f}%")
print(f"Queries where marker was NOT found (kept unchanged, safety fallback): {unchanged_count}/{len(sample)}")