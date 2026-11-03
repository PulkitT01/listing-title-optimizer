# Design decisions

The reasoning behind choices that aren't obvious from the code. Each entry says what was decided, why,
and what it costs. Performance numbers were measured on an Apple M4 Mac with 24 GB of memory.

## Model and hardware

### Train on `Qwen/Qwen3-1.7B`

**Decision:** Use the 2-billion-parameter Qwen3 model instead of `Qwen3-4B-Instruct-2507`.

**Why:** The project should be trainable on a 24 GB Mac. Non-CUDA devices load the model as float32,
so the 4B model needs about 16 GB for weights alone, before any training overhead. The 1.7B model
needs about 8 GB and peaked around 13 GB of process memory in short SFT, DPO and GRPO runs. It is also
Apache-2.0 and needs no access approval.

**Cost:** A smaller model may write weaker titles. Results describe this model, not the 4B one.

### Turn thinking off in `load_tokenizer`

**Decision:** Prepend `{%- set enable_thinking = false -%}` to the chat template of any model that
supports thinking.

**Why:** Titles get 32 tokens. With thinking on, Qwen3-1.7B spent them on reasoning and
`generate_titles` returned the literal title `<think>`, with no error. Every stage loads its tokenizer
through this one function, so one change covers generation, SFT, DPO, GRPO and evaluation, and the
prompt stays a prefix of the full training text.

### Enable gradient checkpointing on every accelerator

**Decision:** Use `device() != "cpu"` instead of tying checkpointing to bf16 support.

**Why:** It was previously NVIDIA-only, which left Mac training without its main memory saving. The
SFT loss was identical with checkpointing on and off. CPU stays off, which keeps tiny-model tests fast.

### Reduce the workload for a Mac

**Decision:** Use smaller device batches with more gradient accumulation, keeping the same effective
batch size for SFT and DPO. Cap SFT at 5,000 samples, use 500 preference listings, and lower GRPO to
4 generations per prompt.

**Why:** The full pipeline has to finish on one Mac in about a day. Measured times on real data:

| Stage | Work | Time |
|---|---|---|
| SFT | 313 steps, ~45 s each | 3 h 55 min |
| Preferences | 2,000 samples + 3,007 Haiku judgments | 1 h |
| DPO | 625 pairs, 40 steps | 1 h 9 min |
| GRPO | 500 steps, ~75 s each | 10 h 26 min |

**Cost:** Less data, and fewer GRPO generations per prompt mean less reward variation to learn from.

## Data

### Shuffle rows in `prepare`

**Decision:** Apply one seeded shuffle after cleaning, before splitting.

**Why:** Raw files load one city at a time, so `max_train_samples` took its rows from whichever city
sorted first. Fixing it at the source covers every stage that limits samples.

### Quote YAML values that look like booleans

**Decision:** Write Toronto's region as `region: "on"`, and test that every city slug loads as a string.

**Why:** YAML reads a bare `on` as `True`, so the download looked for `canada/True/toronto` and failed.

### Anchor the ignore rule to `/data/`

**Decision:** Ignore only the top-level `data/` folder.

**Why:** The unanchored `data/` rule also matched the `title_optimizer/data` package, which kept that
source out of git and hid it from ruff in CI.

## Judge

### Use isolated `claude -p` calls

**Decision:** Models named `claude-*` run through `ClaudeCliJudge`, which starts a fresh `claude -p`
process for each judgment. Each call runs in a neutral temp folder with no tools, settings, MCP
servers, slash commands or saved session.

**Why:** It needs no API key, and it keeps each judgment independent. Judging inside one long Claude
conversation would break the order-swap bias control, because the second order would remember the
first. It would also expose project context and allow scores to drift over thousands of judgments.
The temp folder keeps the project's name out of what the judge can see.

**Cost:** It uses the Claude Code plan's usage limits, it depends on a logged-in CLI, and `claude -p`
has no temperature setting. Exact model IDs and the judge cache keep reruns consistent. Other people
can only reproduce results with the same judge models.

### Use different judges for training and evaluation

**Decision:** Haiku judges preference pairs and Sonnet judges evaluation and `judge-agreement`.

**Why:** DPO learns to please the judge that picked its pairs, so reusing that judge to grade the
result would inflate win rates. A different model reduces this, but the two are related, so it doesn't
remove it. The human agreement check (`title-opt judge-agreement`) remains the independent test.

### Fail loudly on bad judge calls

**Decision:** Each Claude call must exit cleanly and return only a number from 1 to 10. Otherwise it
retries twice, then stops the run. Failed calls are never cached.

**Why:** `parse_score` pulls the first digit from any text, so "resets at 5pm" would otherwise be saved
as a score of 5. One silent bad value is worse than a stopped stage, because cached judgments survive
a rerun.

## Tooling

### Target Python 3.12 only

**Decision:** Set `requires-python = ">=3.12"` and ruff's target to `py312`.

**Why:** CI and `.python-version` already use 3.12, and untested 3.11 support was only a promise.

### Run light mypy on `src/`

**Decision:** Run mypy in default mode on `src/`, in CI and through `make typecheck`, but not in
pre-commit.

**Why:** It keeps the existing type hints accurate at low cost. Strict mode would mostly fight the
incomplete types in ML libraries, and leaving it out of pre-commit keeps commits fast.

### Test the CLI with fake stage functions

**Decision:** `tests/test_cli.py` runs every command with its slow function replaced by a fake.

**Why:** Commands import their code only when run, so `--help` misses broken imports. These tests catch
bad imports and argument wiring in about 2 seconds. `test_smoke_train.py` covers the training itself.

## Known limitations

- **No training resume:** checkpoints are saved, but `title-opt train` can't continue from one, so a
  crash restarts that stage. Finished stages are kept.
- **GRPO runs out of signal:** with the judge reward off, GRPO needs titles to score differently. On
  real data the share of groups where every title scored the same rose from 5% at the start to
  80–95% after step 200. Length was learned first, then most of what remained came from rare
  faithfulness failures. The later steps changed little. Turning on the judge reward
  (`judge` > 0 in `grpo.yaml`) would add variation, at the cost of a judge call per completion.
