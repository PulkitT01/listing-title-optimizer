# TODO

Where things stand (5 Nov 2026): the project is done. The full pipeline has run on my M4 Mac
(24 GB), the results and judge–human agreement are in the README, and the adapters and dataset are
on the Hugging Face Hub. There's no demo, because Gradio Spaces need a paid plan. Training outputs,
data and the judge cache are git-ignored, so they only exist on that Mac.

## Don't delete

- `.cache/judge.jsonl`. Reruns of preferences or eval skip anything in it, which saves hours and
  Claude usage.
- `outputs/` (about 1 GB). Retraining everything takes about 17 hours.
- `data/human_labels.jsonl` (my 50 labels behind the README's judge agreement; backed up in
  `~/human_labels_run2.backup.jsonl`).

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
