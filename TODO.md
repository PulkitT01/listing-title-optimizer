# TODO

Where things stand (3 Nov 2026): I've run the full pipeline once on my M4 Mac (24 GB), and the
results are in the README. Training outputs, data and the judge cache are git-ignored, so they only
exist on that Mac.

## Next

- [ ] **Measure judge–human agreement.** Label 200 pairs myself, then compare them with the Sonnet
  judge (about 400 judge calls):
  ```bash
  uv run title-opt label data/preferences/dpo_train.jsonl   # writes data/human_labels.jsonl
  uv run title-opt judge-agreement
  ```
  Then put the agreement and Cohen's κ in the README's "Judge reliability" line. `data/` is
  git-ignored, so back up `human_labels.jsonl` somewhere else.
- [ ] **Publish to the Hugging Face Hub** and fill in the README's Demo, Model and Dataset links:
  - adapters: `outputs/sft`, `outputs/dpo`, `outputs/grpo` (LoRA on `Qwen/Qwen3-1.7B`, chained:
    DPO and GRPO apply on top of SFT)
  - dataset: `data/processed/` plus `data/raw/MANIFEST.json`. The README already says the processed
    data is on the Hub, but it isn't yet. The September 2026 snapshots rotate off Inside Airbnb
    around September 2027, and after that the results can't be reproduced without this upload.
  - demo: deploy `app/app.py` as a Space with `MODEL_ID=Qwen/Qwen3-1.7B` and `ADAPTERS` set to the
    SFT and GRPO repos
- [ ] **Merge `mac-training`** through a PR.

## Don't delete

- `.cache/judge.jsonl` (6,353 judgments). Reruns of preferences or eval skip anything in it, which
  saves hours and Claude usage.
- `outputs/` (about 1 GB). Retraining everything takes about 17 hours.

## Ideas

- **GRPO runs out of signal.** After about step 200, 80–95% of groups scored every title the same,
  so the last 300 steps (about 6 hours) changed little. Options: stop at 200 steps, set
  `judge` > 0 in `configs/grpo.yaml` for a judge reward (one judge call per completion), or use
  `num_generations: 8`.
- **DPO gained little in training.** Pair accuracy rose only from 0.62 to 0.66 over one epoch of 625
  pairs. More preference listings (500 now) or a second epoch might help, but DPO already has the
  best win rate, so measure before spending the time.
- **DPO and GRPO are tied.** Win rates are 68.5% and 67.7% with overlapping CIs. Separating them
  would need a larger eval set or a second judge.
- **Training can't resume.** A crash restarts the stage from scratch. Resuming from the saved
  checkpoints would make long Mac runs safer.
- From the README: an NLI-based hallucination checker, and a reward model trained on engagement.
