"""Command-line entry point: `title-opt --help`."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from title_optimizer.config import (
    DataConfig,
    EvalConfig,
    JudgeConfig,
    PreferenceConfig,
    TrainConfig,
    load_config,
)

app = typer.Typer(no_args_is_help=True, add_completion=False)
console = Console()
ConfigOpt = Annotated[Path, typer.Option("--config", "-c", exists=True, dir_okay=False)]


@app.command()
def download(config: ConfigOpt = Path("configs/data.yaml"), force: bool = False) -> None:
    """Download Inside Airbnb listings for the cities in the data config."""
    from title_optimizer.data.download import download as _download

    _download(load_config(config, DataConfig), force=force)


@app.command()
def prepare(config: ConfigOpt = Path("configs/data.yaml")) -> None:
    """Clean listings, drop auto-generated titles, split by host, write JSONL."""
    from title_optimizer.data.prepare import prepare as _prepare

    _prepare(load_config(config, DataConfig))


@app.command()
def train(
    config: ConfigOpt,
    judge_config: Annotated[Path | None, typer.Option(exists=True)] = None,
    max_steps: int | None = None,
) -> None:
    """Train one stage (sft | dpo | grpo) as set in the config."""
    from title_optimizer.train import train as _train

    overrides = {"max_steps": max_steps} if max_steps is not None else None
    cfg = load_config(config, TrainConfig, overrides)
    _train(cfg, load_config(judge_config, JudgeConfig) if judge_config else None)


@app.command()
def preferences(config: ConfigOpt = Path("configs/preferences.yaml")) -> None:
    """Sample candidate titles and build DPO preference pairs (rules + LLM judge)."""
    from title_optimizer.preferences import generate_preferences

    generate_preferences(load_config(config, PreferenceConfig))


@app.command()
def evaluate(config: ConfigOpt = Path("configs/eval.yaml")) -> None:
    """Evaluate systems on the held-out test split and write results/RESULTS.md."""
    from title_optimizer.evaluate import evaluate as _evaluate

    _evaluate(load_config(config, EvalConfig))


@app.command()
def merge(base_model: str, adapter: str, output_dir: str) -> None:
    """Merge a LoRA adapter into its base model (needed before 4-bit training on top of it)."""
    from title_optimizer.modeling import load_model, load_tokenizer

    model = load_model(base_model, adapter=adapter)
    model.save_pretrained(output_dir)
    load_tokenizer(base_model).save_pretrained(output_dir)
    console.print(f"[green]merged[/] -> {output_dir}")


@app.command()
def label(
    pairs_file: Path,
    output: Path = Path("data/human_labels.jsonl"),
    n: int = 200,
    seed: int = 0,
) -> None:
    """Label title pairs yourself in the terminal, to measure judge-human agreement."""
    from title_optimizer.data.prepare import read_jsonl

    rows = read_jsonl(pairs_file)
    random.Random(seed).shuffle(rows)
    done = len(read_jsonl(output)) if output.exists() else 0
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a") as f:
        for row in rows[done:n]:
            a, b = row["chosen"][-1]["content"], row["rejected"][-1]["content"]
            if random.random() < 0.5:
                a, b = b, a
            desc = row["prompt"][-1]["content"]
            console.rule()
            console.print(desc[:1500])
            console.print(f"\n[bold]A:[/] {a}\n[bold]B:[/] {b}")
            choice = typer.prompt("Better? [a/b/t(ie)/q(uit)]").strip().lower()
            if choice == "q":
                break
            human = {"a": "A", "b": "B"}.get(choice, "tie")
            rec = {"description": desc, "title_a": a, "title_b": b, "human": human}
            f.write(json.dumps(rec) + "\n")


@app.command("judge-agreement")
def judge_agreement(
    labels_file: Path = Path("data/human_labels.jsonl"),
    judge_config: Annotated[Path, typer.Option(exists=True)] = Path("configs/judge.yaml"),
) -> None:
    """Report agreement and Cohen's kappa between the LLM judge and your labels."""
    from title_optimizer.evaluate import judge_agreement as _agreement
    from title_optimizer.judge import make_judge

    _agreement(str(labels_file), make_judge(load_config(judge_config, JudgeConfig)))


if __name__ == "__main__":
    app()
