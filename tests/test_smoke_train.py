"""End-to-end smoke test: prepare -> SFT -> preferences -> DPO -> GRPO -> evaluate, on a tiny model.

It checks the plumbing (data formats, TRL APIs, adapter chaining), not model quality.
"""

import pytest

from title_optimizer.config import DataConfig, EvalConfig, TrainConfig
from title_optimizer.data.prepare import prepare, read_jsonl, write_jsonl
from title_optimizer.evaluate import evaluate
from title_optimizer.judge import RuleJudge
from title_optimizer.modeling import generate_titles, load_model, load_tokenizer
from title_optimizer.preferences import build_pairs, pair_example
from title_optimizer.train import train

pytestmark = pytest.mark.slow


def _train_cfg(stage, model_dir, tmp_path, **kw):
    return TrainConfig(
        stage=stage,
        model={"base_model": model_dir, "lora_r": 4, "lora_alpha": 8},
        output_dir=str(tmp_path / stage),
        max_steps=2,
        per_device_train_batch_size=2,
        logging_steps=1,
        save_steps=1000,
        max_length=512,
        **kw,
    )


def test_full_pipeline(raw_listings, tiny_model_dir, tmp_path, monkeypatch):
    monkeypatch.setenv("JUDGE_MODEL", "rules")
    processed = tmp_path / "processed"
    prepare(DataConfig(raw_dir=str(raw_listings.parent), processed_dir=str(processed)))

    train(
        _train_cfg("sft", tiny_model_dir, tmp_path, train_file=str(processed / "sft_train.jsonl"))
    )
    sft_dir = str(tmp_path / "sft")

    records = read_jsonl(processed / "train.jsonl")[:4]
    tok = load_tokenizer(tiny_model_dir)
    model = load_model(tiny_model_dir, adapter=sft_dir)
    cands = generate_titles(model, tok, records, num_return_sequences=2, temperature=1.0)
    assert len(cands) == 4 and all(len(c) == 2 for c in cands)

    pairs = build_pairs(
        records, [c + [r["title"]] for c, r in zip(cands, records, strict=True)], RuleJudge()
    )
    # Random tiny model may not yield pairs; guarantee a few so DPO has data.
    pairs += [pair_example(r, r["title"], r["title"] + " with hot tub!!", "rule") for r in records]
    write_jsonl(pairs, tmp_path / "dpo.jsonl")

    train(
        _train_cfg(
            "dpo",
            tiny_model_dir,
            tmp_path,
            train_file=str(tmp_path / "dpo.jsonl"),
            init_adapter=sft_dir,
        )
    )
    train(
        _train_cfg(
            "grpo",
            tiny_model_dir,
            tmp_path,
            train_file=str(processed / "train.jsonl"),
            init_adapter=sft_dir,
            num_generations=2,
            max_completion_length=8,
            max_train_samples=8,
        )
    )

    metrics = evaluate(
        EvalConfig(
            test_file=str(processed / "train.jsonl"),
            num_examples=4,
            base_model=tiny_model_dir,
            systems={
                "sft+dpo": [sft_dir, str(tmp_path / "dpo")],
                "sft+grpo": [sft_dir, str(tmp_path / "grpo")],
            },
            max_new_tokens=8,
            output_dir=str(tmp_path / "results"),
        )
    )
    assert {"original (host)", "sft+dpo", "sft+grpo"} <= set(metrics)
    assert (tmp_path / "results" / "RESULTS.md").exists()
