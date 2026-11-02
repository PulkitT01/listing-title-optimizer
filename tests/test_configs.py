from pathlib import Path

import pytest

from title_optimizer.config import (
    DataConfig,
    EvalConfig,
    JudgeConfig,
    PreferenceConfig,
    TrainConfig,
    load_config,
)

CONFIGS = Path(__file__).parents[1] / "configs"


@pytest.mark.parametrize(
    "name,cls",
    [
        ("data.yaml", DataConfig),
        ("judge.yaml", JudgeConfig),
        ("sft.yaml", TrainConfig),
        ("dpo.yaml", TrainConfig),
        ("grpo.yaml", TrainConfig),
        ("preferences.yaml", PreferenceConfig),
        ("eval.yaml", EvalConfig),
    ],
)
def test_shipped_configs_load(name, cls):
    cfg = load_config(CONFIGS / name, cls)
    if isinstance(cfg, TrainConfig):
        assert cfg.stage in name
        cfg.model_config()
    if isinstance(cfg, PreferenceConfig | EvalConfig):
        cfg.judge_config()


def test_unknown_keys_fail(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("stage: sft\nlearning_rat: 1e-4\n")
    with pytest.raises(ValueError, match="learning_rat"):
        load_config(p, TrainConfig)


def test_grpo_batch_divisible_by_generations():
    cfg = load_config(CONFIGS / "grpo.yaml", TrainConfig)
    assert cfg.per_device_train_batch_size % cfg.num_generations == 0
