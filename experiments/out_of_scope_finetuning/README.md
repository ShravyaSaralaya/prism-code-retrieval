# Fine-tuning attempt (out of scope)

This folder contains a working LoRA fine-tuning pipeline for `bge-m3`
(hard-negative mining from the train split only, no test leakage, trained
successfully — final NDCG@10/MRR@10 were not evaluated after the organizers'
clarification below).

**The organizers confirmed that fine-tuning the embedding model is out of
scope for this problem statement:**

> "We don't think that fine-tuning the embedding model is the right
> direction here... fine-tuning the embedding model is out of scope.
> However, if you want to come up with a different approach (modifying the
> attention heads for a new feature which is used for retrieval or a custom
> architecture), you can fine-tune that. Keep in mind that the time allotted
> for the hackathon is not sufficient for approaches like these, so we would
> recommend against them."

This code is kept in the repo for transparency about what was explored, but
**is not part of the final submitted pipeline** — see `src/` and the main
README for the actual submission.

## Files
- `finetune_bge_m3.py` — LoRA fine-tuning with hard-negative mining
- `eval_finetuned_model.py` — evaluation script (not run to completion, given
  the scope clarification above)
