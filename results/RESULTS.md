# Results

Held-out test listings: **500** (hosts never seen in training). Judge: `claude-sonnet-5-5`.

| System | Win rate vs original (95% CI) | Hallucination ↓ | ≤50 chars ↑ | Style issues ↓ |
|---|---|---|---|---|
| original (host) | — | 3.0% | 100.0% | 2.4% |
| base | 36.1% (32.1%–40.2%) | 4.0% | 23.8% | 0.4% |
| sft | 65.3% (61.3%–69.4%) | 0.8% | 97.2% | 0.0% |
| sft+dpo | 68.5% (64.4%–72.3%) | 0.2% | 99.8% | 0.0% |
| sft+grpo | 67.7% (63.8%–71.4%) | 0.2% | 100.0% | 0.0% |
