"""Evaluate systems on held-out listings and write `results/RESULTS.md` + `metrics.json`.

Metrics per system:
- win rate vs the host's original title (LLM judge, order-swapped; ties count as half), 95% bootstrap CI
- hallucination rate: share of titles with at least one unsupported feature claim
- length compliance (<= 50 chars) and style-violation rate
"""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from rich.console import Console
from rich.table import Table

from title_optimizer.config import EvalConfig
from title_optimizer.data.prepare import read_jsonl, write_jsonl
from title_optimizer.judge import make_judge
from title_optimizer.modeling import generate_titles, load_model, load_tokenizer
from title_optimizer.rewards import MAX_CHARS, style_violations, unsupported_claims

console = Console()


def bootstrap_ci(values: list[float], n: int = 2000, seed: int = 0) -> tuple[float, float]:
    if not values:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    arr = np.asarray(values)
    means = rng.choice(arr, size=(n, len(arr)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def title_metrics(titles: list[str], records: list[dict[str, Any]]) -> dict[str, float]:
    halluc = [
        bool(unsupported_claims(t, r["description"], r.get("amenities", [])))
        for t, r in zip(titles, records, strict=True)
    ]
    return {
        "hallucination_rate": float(np.mean(halluc)),
        "length_ok_rate": float(np.mean([0 < len(t) <= MAX_CHARS for t in titles])),
        "style_violation_rate": float(np.mean([bool(style_violations(t)) for t in titles])),
        "mean_chars": float(np.mean([len(t) for t in titles])),
    }


def win_rate(verdicts: Sequence[str]) -> dict[str, float]:
    points = [1.0 if v == "A" else 0.5 if v == "tie" else 0.0 for v in verdicts]
    lo, hi = bootstrap_ci(points)
    return {
        "win_rate_vs_original": float(np.mean(points)),
        "win_rate_ci_low": lo,
        "win_rate_ci_high": hi,
        "wins": verdicts.count("A"),
        "ties": verdicts.count("tie"),
        "losses": verdicts.count("B"),
    }


def render_markdown(metrics: dict[str, dict[str, float]], n: int, judge_name: str) -> str:
    lines = [
        "# Results",
        "",
        f"Held-out test listings: **{n}** (hosts never seen in training). Judge: `{judge_name}`.",
        "",
        "| System | Win rate vs original (95% CI) | Hallucination ↓ | ≤50 chars ↑ | Style issues ↓ |",
        "|---|---|---|---|---|",
    ]
    for name, m in metrics.items():
        wr = (
            f"{m['win_rate_vs_original']:.1%} ({m['win_rate_ci_low']:.1%}–{m['win_rate_ci_high']:.1%})"
            if "win_rate_vs_original" in m
            else "—"
        )
        lines.append(
            f"| {name} | {wr} | {m['hallucination_rate']:.1%} | {m['length_ok_rate']:.1%} "
            f"| {m['style_violation_rate']:.1%} |"
        )
    return "\n".join(lines) + "\n"


def evaluate(cfg: EvalConfig) -> dict[str, Any]:
    records = read_jsonl(cfg.test_file)
    random.Random(cfg.seed).shuffle(records)
    records = records[: cfg.num_examples]
    originals = [r["title"] for r in records]
    judge = make_judge(cfg.judge_config())
    tok = load_tokenizer(cfg.base_model)

    metrics: dict[str, dict[str, float]] = {"original (host)": title_metrics(originals, records)}
    samples = [{"listing_id": r["listing_id"], "original": r["title"]} for r in records]

    for name, adapter in cfg.systems.items():
        console.print(f"[bold]generating[/] {name}")
        model = load_model(cfg.base_model, adapter=adapter)
        titles = [
            t[0]
            for t in generate_titles(
                model, tok, records, max_new_tokens=cfg.max_new_tokens, batch_size=cfg.batch_size
            )
        ]
        del model
        verdicts = judge.compare_many([r["description"] for r in records], titles, originals)
        metrics[name] = {**title_metrics(titles, records), **win_rate(verdicts)}
        for s, t, v in zip(samples, titles, verdicts, strict=True):
            s[name] = t
            s[f"{name}_vs_original"] = v

    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    write_jsonl(samples, out / "samples.jsonl")
    judge_name = getattr(judge, "model", "rules")
    (out / "RESULTS.md").write_text(render_markdown(metrics, len(records), judge_name))

    table = Table("system", "win vs orig", "halluc.", "≤50ch", "style")
    for name, m in metrics.items():
        table.add_row(
            name,
            f"{m['win_rate_vs_original']:.1%}" if "win_rate_vs_original" in m else "—",
            f"{m['hallucination_rate']:.1%}",
            f"{m['length_ok_rate']:.1%}",
            f"{m['style_violation_rate']:.1%}",
        )
    console.print(table)
    return metrics


# ---- judge vs human agreement ----------------------------------------------------------------


def cohen_kappa(a: list[str], b: list[str]) -> float:
    labels = sorted(set(a) | set(b))
    n = len(a)
    po = sum(x == y for x, y in zip(a, b, strict=True)) / n
    pe = sum((a.count(lbl) / n) * (b.count(lbl) / n) for lbl in labels)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def judge_agreement(labels_file: str, judge: Any) -> dict[str, float]:
    """`labels_file` rows: {description, title_a, title_b, human: "A"|"B"|"tie"}."""
    rows = read_jsonl(labels_file)
    pred = judge.compare_many(
        [r["description"] for r in rows], [r["title_a"] for r in rows], [r["title_b"] for r in rows]
    )
    human = [r["human"] for r in rows]
    decided = [(h, p) for h, p in zip(human, pred, strict=True) if h != "tie" and p != "tie"]
    result = {
        "n": len(rows),
        "agreement": sum(h == p for h, p in zip(human, pred, strict=True)) / len(rows),
        "agreement_excluding_ties": (
            sum(h == p for h, p in decided) / len(decided) if decided else float("nan")
        ),
        "cohen_kappa": cohen_kappa(human, pred),
    }
    console.print(result)
    return result
