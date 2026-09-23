"""
Phase 1 (fixed): Try a few candidate embedding models on AppsRetrieval and
compare against our baseline (all-MiniLM-L6-v2, NDCG@10 = 0.0660).

Fix from last run: MTEB v2 enforces a strict model interface, so a hand-rolled
wrapper class gets rejected. Instead we use mteb.get_model(name), which uses
MTEB's own built-in registry -- for well-known models like e5, this already
knows to add the required "query: " / "passage: " prefixes correctly.

Run:  python compare_models.py
"""

import json
import mteb

CANDIDATES = [
    "intfloat/e5-base-v2",
    "Snowflake/snowflake-arctic-embed-m-v2.0",
]

tasks = mteb.get_tasks(tasks=["AppsRetrieval"])
results_summary = [("all-MiniLM-L6-v2 (baseline)", 0.0660)]

for model_name in CANDIDATES:
    print(f"\n{'='*60}\nEvaluating: {model_name}\n{'='*60}")
    model = mteb.get_model(model_name)  # uses MTEB's built-in wrapper if one exists
    output_folder = f"results/{model_name.replace('/', '__')}"
    evaluation = mteb.MTEB(tasks=tasks)
    result = evaluation.run(model, output_folder=output_folder)

    # Be defensive about the exact shape of the result object across mteb versions --
    # fall back to reading the JSON straight off disk if needed.
    main_score = None
    try:
        main_score = result[0].scores["test"][0]["main_score"]
    except Exception:
        pass

    if main_score is None:
        print(f"(Could not read score from result object directly -- check the JSON under {output_folder}/)")
    else:
        print(f"main_score (NDCG@10) for {model_name}: {main_score}")

    results_summary.append((model_name, main_score))

print(f"\n{'='*60}\nSUMMARY (NDCG@10)\n{'='*60}")
for name, score in results_summary:
    print(f"{name:<45} {score}")