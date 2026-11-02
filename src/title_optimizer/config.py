"""Typed configs loaded from YAML. One file per stage lives in `configs/`."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def load_config[T](path: str | Path, cls: type[T], overrides: dict[str, Any] | None = None) -> T:
    """Load a YAML file into a dataclass, rejecting unknown keys so typos fail loudly."""
    raw = yaml.safe_load(Path(path).read_text()) or {}
    raw.update(overrides or {})
    known = {f.name for f in dataclasses.fields(cls)}  # type: ignore[arg-type]
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"Unknown keys in {path}: {sorted(unknown)}")
    return cls(**raw)


@dataclass
class DataConfig:
    # Inside Airbnb snapshots: (country, region, city, date). See https://insideairbnb.com/get-the-data/
    cities: list[dict[str, str]] = field(default_factory=list)
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    language: str = "en"
    min_description_chars: int = 200
    max_description_chars: int = 2000
    min_title_chars: int = 10
    max_title_chars: int = 50
    min_reviews: int = 5
    min_rating: float = 4.7
    # Listings in the top `engagement_quantile` of reviews_per_month count as "good titles" for SFT
    engagement_quantile: float = 0.66
    test_fraction: float = 0.1
    val_fraction: float = 0.05
    seed: int = 42


@dataclass
class ModelConfig:
    base_model: str = "Qwen/Qwen3-4B-Instruct-2507"
    load_in_4bit: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: str | list[str] = "all-linear"


@dataclass
class TrainConfig:
    stage: str = "sft"  # sft | dpo | grpo
    model: dict[str, Any] = field(default_factory=dict)
    train_file: str = ""
    eval_file: str | None = None
    # Previous stage adapter(s) to merge into the base before training, e.g. "outputs/sft"
    init_adapter: str | list[str] | None = None
    output_dir: str = "outputs/run"
    learning_rate: float = 2e-4
    num_train_epochs: float = 1.0
    max_steps: int = -1
    per_device_train_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    max_length: int = 1024
    logging_steps: int = 10
    save_steps: int = 200
    seed: int = 42
    report_to: str = "none"
    bf16: bool = True
    # DPO
    beta: float = 0.1
    # GRPO
    num_generations: int = 8
    max_completion_length: int = 32
    temperature: float = 0.9
    reward_weights: dict[str, float] = field(
        default_factory=lambda: {"length": 1.0, "style": 0.5, "faithfulness": 2.0, "judge": 0.0}
    )
    max_train_samples: int | None = None

    def model_config(self) -> ModelConfig:
        return ModelConfig(**self.model)


@dataclass
class JudgeConfig:
    # Any OpenAI-compatible endpoint: OpenAI, Anthropic's OpenAI-compatible API, vLLM, Ollama, ...
    base_url: str | None = None  # falls back to $JUDGE_BASE_URL
    model: str = "gpt-4.1-mini"  # falls back to $JUDGE_MODEL if set
    api_key_env: str = "JUDGE_API_KEY"
    max_workers: int = 8
    temperature: float = 0.0


@dataclass
class PreferenceConfig:
    policy_model: str = "Qwen/Qwen3-4B-Instruct-2507"
    policy_adapter: str | None = "outputs/sft"
    input_file: str = "data/processed/train.jsonl"
    output_file: str = "data/preferences/dpo_train.jsonl"
    num_listings: int = 3000
    num_candidates: int = 4
    temperature: float = 0.9
    max_new_tokens: int = 32
    batch_size: int = 16
    include_original_title: bool = True
    judge: dict[str, Any] = field(default_factory=dict)
    seed: int = 42

    def judge_config(self) -> JudgeConfig:
        return JudgeConfig(**self.judge)


@dataclass
class EvalConfig:
    test_file: str = "data/processed/test.jsonl"
    num_examples: int = 500
    base_model: str = "Qwen/Qwen3-4B-Instruct-2507"
    # name -> adapter chain (null = base model). Each system is compared against the original titles.
    systems: dict[str, str | list[str] | None] = field(default_factory=lambda: {"base": None})
    max_new_tokens: int = 32
    batch_size: int = 16
    judge: dict[str, Any] = field(default_factory=dict)
    output_dir: str = "results"
    seed: int = 42

    def judge_config(self) -> JudgeConfig:
        return JudgeConfig(**self.judge)
