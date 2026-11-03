"""SFT, DPO and GRPO training with TRL + LoRA. One function per stage, driven by YAML configs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset
from rich.console import Console

from title_optimizer.config import JudgeConfig, TrainConfig
from title_optimizer.data.prepare import read_jsonl
from title_optimizer.modeling import device, load_model, load_tokenizer, lora_config, use_bf16
from title_optimizer.prompts import build_messages
from title_optimizer.rewards import REWARD_FUNCS, make_judge_reward

console = Console()


def _common_args(cfg: TrainConfig) -> dict[str, Any]:
    return {
        "output_dir": cfg.output_dir,
        "learning_rate": cfg.learning_rate,
        "num_train_epochs": cfg.num_train_epochs,
        "max_steps": cfg.max_steps,
        "per_device_train_batch_size": cfg.per_device_train_batch_size,
        "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
        "logging_steps": cfg.logging_steps,
        "save_steps": cfg.save_steps,
        "save_total_limit": 2,
        "seed": cfg.seed,
        "report_to": cfg.report_to,
        "bf16": cfg.bf16 and use_bf16(),
        "gradient_checkpointing": device() != "cpu",
        "lr_scheduler_type": "cosine",
        "warmup_steps": 0.05,  # float < 1 = ratio of total steps
    }


def _load_jsonl(path: str, limit: int | None = None) -> Dataset:
    ds = load_dataset("json", data_files=path, split="train")
    return ds.select(range(min(limit, len(ds)))) if limit else ds


def _model_and_tok(cfg: TrainConfig):
    mc = cfg.model_config()
    tok = load_tokenizer(mc.base_model)
    # Merge the previous stage's adapter so it becomes both the policy init and the reference.
    model = load_model(mc.base_model, adapter=cfg.init_adapter, load_in_4bit=mc.load_in_4bit)
    if mc.load_in_4bit:
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(model)
    return model, tok, lora_config(mc)


def _finish(trainer: Any, cfg: TrainConfig) -> None:
    trainer.train()
    trainer.save_model(cfg.output_dir)
    trainer.processing_class.save_pretrained(cfg.output_dir)
    meta = {"stage": cfg.stage, "base_model": cfg.model_config().base_model, **vars(cfg)}
    Path(cfg.output_dir, "run_config.json").write_text(json.dumps(meta, indent=2, default=str))
    console.print(f"[green]saved[/] {cfg.output_dir}")


def train_sft(cfg: TrainConfig) -> None:
    from trl import SFTConfig, SFTTrainer

    model, tok, peft_cfg = _model_and_tok(cfg)
    args = SFTConfig(**_common_args(cfg), max_length=cfg.max_length)
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=_load_jsonl(cfg.train_file, cfg.max_train_samples),
        eval_dataset=_load_jsonl(cfg.eval_file) if cfg.eval_file else None,
        processing_class=tok,
        peft_config=peft_cfg,
    )
    _finish(trainer, cfg)


def train_dpo(cfg: TrainConfig) -> None:
    from trl import DPOConfig, DPOTrainer

    model, tok, peft_cfg = _model_and_tok(cfg)
    args = DPOConfig(**_common_args(cfg), beta=cfg.beta, max_length=cfg.max_length)
    ds = _load_jsonl(cfg.train_file, cfg.max_train_samples)
    ds = ds.select_columns(["prompt", "chosen", "rejected"])
    trainer = DPOTrainer(
        model=model, args=args, train_dataset=ds, processing_class=tok, peft_config=peft_cfg
    )
    _finish(trainer, cfg)


def grpo_dataset(path: str, limit: int | None = None) -> Dataset:
    """Prompt-only dataset; extra columns are passed to reward functions as kwargs."""
    rows = read_jsonl(path)[:limit] if limit else read_jsonl(path)
    return Dataset.from_list(
        [
            {
                "prompt": build_messages(r),
                "description": r["description"],
                "amenities": r["amenities"],
            }
            for r in rows
        ]
    )


def train_grpo(cfg: TrainConfig, judge_cfg: JudgeConfig | None = None) -> None:
    from trl import GRPOConfig, GRPOTrainer

    model, tok, peft_cfg = _model_and_tok(cfg)
    weights = {k: v for k, v in cfg.reward_weights.items() if v}
    funcs, w = [], []
    for name, weight in weights.items():
        if name == "judge":
            from title_optimizer.judge import make_judge

            funcs.append(make_judge_reward(make_judge(judge_cfg or JudgeConfig())))
        else:
            funcs.append(REWARD_FUNCS[name])
        w.append(weight)

    args = GRPOConfig(
        **_common_args(cfg),
        num_generations=cfg.num_generations,
        max_completion_length=cfg.max_completion_length,
        temperature=cfg.temperature,
        beta=cfg.beta,
        reward_weights=w,
    )
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=funcs,
        args=args,
        train_dataset=grpo_dataset(cfg.train_file, cfg.max_train_samples),
        processing_class=tok,
        peft_config=peft_cfg,
    )
    _finish(trainer, cfg)


def train(cfg: TrainConfig, judge_cfg: JudgeConfig | None = None) -> None:
    if cfg.stage == "sft":
        train_sft(cfg)
    elif cfg.stage == "dpo":
        train_dpo(cfg)
    elif cfg.stage == "grpo":
        train_grpo(cfg, judge_cfg)
    else:
        raise ValueError(f"Unknown stage {cfg.stage!r}")
