"""CLI wiring: stage functions are faked, so these check arguments and imports, not training."""

from pathlib import Path
from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

from title_optimizer.cli import app
from title_optimizer.config import (
    DataConfig,
    EvalConfig,
    JudgeConfig,
    PreferenceConfig,
    TrainConfig,
)
from title_optimizer.data.prepare import read_jsonl, write_jsonl

runner = CliRunner()


@pytest.fixture(autouse=True)
def repo_root(monkeypatch):
    """Default config paths such as `configs/data.yaml` are relative to the repo root."""
    monkeypatch.chdir(Path(__file__).parents[1])


def fake(monkeypatch, target: str) -> Mock:
    mock = Mock()
    monkeypatch.setattr(target, mock)
    return mock


@pytest.mark.parametrize(
    "command,target,cfg_cls",
    [
        ("download", "title_optimizer.data.download.download", DataConfig),
        ("prepare", "title_optimizer.data.prepare.prepare", DataConfig),
        ("preferences", "title_optimizer.preferences.generate_preferences", PreferenceConfig),
        ("evaluate", "title_optimizer.evaluate.evaluate", EvalConfig),
    ],
)
def test_commands_load_their_default_config(monkeypatch, command, target, cfg_cls):
    stage = fake(monkeypatch, target)
    result = runner.invoke(app, [command])
    assert result.exit_code == 0, result.output
    stage.assert_called_once()
    assert isinstance(stage.call_args.args[0], cfg_cls)


def test_download_passes_force(monkeypatch):
    stage = fake(monkeypatch, "title_optimizer.data.download.download")
    assert runner.invoke(app, ["download", "--force"]).exit_code == 0
    assert stage.call_args.kwargs == {"force": True}


def test_train_applies_max_steps_override(monkeypatch):
    stage = fake(monkeypatch, "title_optimizer.train.train")
    result = runner.invoke(app, ["train", "-c", "configs/sft.yaml", "--max-steps", "5"])
    assert result.exit_code == 0, result.output
    cfg, judge_cfg = stage.call_args.args
    assert isinstance(cfg, TrainConfig)
    assert (cfg.stage, cfg.max_steps, judge_cfg) == ("sft", 5, None)


def test_train_loads_judge_config(monkeypatch):
    stage = fake(monkeypatch, "title_optimizer.train.train")
    args = ["train", "-c", "configs/grpo.yaml", "--judge-config", "configs/judge.yaml"]
    assert runner.invoke(app, args).exit_code == 0
    cfg, judge_cfg = stage.call_args.args
    assert cfg.stage == "grpo"
    assert isinstance(judge_cfg, JudgeConfig)


def test_missing_config_is_a_usage_error(monkeypatch):
    stage = fake(monkeypatch, "title_optimizer.train.train")
    assert runner.invoke(app, ["train", "-c", "configs/nope.yaml"]).exit_code == 2
    stage.assert_not_called()


def test_merge_saves_model_and_tokenizer(monkeypatch):
    load_model = fake(monkeypatch, "title_optimizer.modeling.load_model")
    load_tokenizer = fake(monkeypatch, "title_optimizer.modeling.load_tokenizer")
    result = runner.invoke(app, ["merge", "base", "adapter", "out"])
    assert result.exit_code == 0, result.output
    load_model.assert_called_once_with("base", adapter="adapter")
    load_model.return_value.save_pretrained.assert_called_once_with("out")
    load_tokenizer.return_value.save_pretrained.assert_called_once_with("out")


def test_judge_agreement_uses_labels_and_judge(monkeypatch):
    agreement = fake(monkeypatch, "title_optimizer.evaluate.judge_agreement")
    make_judge = fake(monkeypatch, "title_optimizer.judge.make_judge")
    result = runner.invoke(app, ["judge-agreement", "--labels-file", "labels.jsonl"])
    assert result.exit_code == 0, result.output
    assert isinstance(make_judge.call_args.args[0], JudgeConfig)
    agreement.assert_called_once_with("labels.jsonl", make_judge.return_value)


def test_label_records_answers_and_resumes(tmp_path):
    pairs = tmp_path / "pairs.jsonl"
    write_jsonl(
        [
            {
                "prompt": [{"content": f"listing {i}"}],
                "chosen": [{"content": f"good {i}"}],
                "rejected": [{"content": f"bad {i}"}],
            }
            for i in range(3)
        ],
        pairs,
    )
    output = tmp_path / "labels.jsonl"
    args = ["label", str(pairs), "--output", str(output)]

    assert runner.invoke(app, args, input="a\nq\n").exit_code == 0  # quit after one answer
    assert [r["human"] for r in read_jsonl(output)] == ["A"]

    assert runner.invoke(app, args, input="b\nt\n").exit_code == 0  # picks up at the second pair
    rows = read_jsonl(output)
    assert [r["human"] for r in rows] == ["A", "B", "tie"]
    assert len({r["description"] for r in rows}) == 3
    for r in rows:
        i = r["description"].removeprefix("listing ")
        assert {r["title_a"], r["title_b"]} == {f"good {i}", f"bad {i}"}
