"""Build DPO preference pairs: sample candidates, apply verifiable checks, then ask the judge.

Per listing:
1. sample `num_candidates` titles from the SFT policy (plus the host's original title),
2. anything with an unsupported claim or broken format is a "hard negative" and can only be rejected,
3. the judge scores the remaining candidates (pointwise), and the best and worst become a pair,
4. the pair is confirmed with an order-swapped pairwise comparison; ties are dropped.

That's ~k+2 judge calls per listing instead of k*(k-1) for a full round robin.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.progress import track

from title_optimizer.config import PreferenceConfig
from title_optimizer.data.prepare import read_jsonl, write_jsonl
from title_optimizer.judge import make_judge
from title_optimizer.modeling import generate_titles, load_model, load_tokenizer
from title_optimizer.prompts import build_messages
from title_optimizer.rewards import length_score, style_violations, unsupported_claims

console = Console()


def is_hard_negative(title: str, record: dict[str, Any]) -> bool:
    return (
        bool(unsupported_claims(title, record["description"], record.get("amenities", [])))
        or length_score(title) < 0
        or bool(style_violations(title))
    )


def pair_example(record: dict[str, Any], chosen: str, rejected: str, source: str) -> dict[str, Any]:
    return {
        "prompt": build_messages(record),
        "chosen": [{"role": "assistant", "content": chosen}],
        "rejected": [{"role": "assistant", "content": rejected}],
        "listing_id": record["listing_id"],
        "pair_source": source,
    }


def build_pairs(
    records: list[dict[str, Any]], candidates: list[list[str]], judge: Any, seed: int = 0
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    pairs, to_score = [], []

    for rec, cands in zip(records, candidates, strict=True):
        cands = list(dict.fromkeys(c for c in cands if c))  # dedupe, keep order
        good = [c for c in cands if not is_hard_negative(c, rec)]
        bad = [c for c in cands if is_hard_negative(c, rec)]
        if good and bad:
            # Verifiable preference: no judge needed.
            pairs.append(pair_example(rec, rng.choice(good), rng.choice(bad), "rule"))
        if len(good) >= 2:
            to_score.append((rec, good))

    if not to_score:
        return pairs

    flat_desc = [rec["description"] for rec, good in to_score for _ in good]
    flat_titles = [t for _, good in to_score for t in good]
    flat_scores = judge.score_many(flat_desc, flat_titles)

    best_worst, i = [], 0
    for rec, good in to_score:
        scores = flat_scores[i : i + len(good)]
        i += len(good)
        ranked = sorted(zip(scores, good, strict=True), key=lambda x: x[0])
        if ranked[-1][0] - ranked[0][0] >= 2:
            best_worst.append((rec, ranked[-1][1], ranked[0][1]))

    verdicts = judge.compare_many(
        [r["description"] for r, _, _ in best_worst],
        [b for _, b, _ in best_worst],
        [w for _, _, w in best_worst],
    )
    for (rec, best, worst), v in zip(best_worst, verdicts, strict=True):
        if v == "A":
            pairs.append(pair_example(rec, best, worst, "judge"))
    return pairs


def generate_preferences(cfg: PreferenceConfig) -> Path:
    rng = random.Random(cfg.seed)
    records = read_jsonl(cfg.input_file)
    rng.shuffle(records)
    records = records[: cfg.num_listings]

    tok = load_tokenizer(cfg.policy_model)
    model = load_model(cfg.policy_model, adapter=cfg.policy_adapter)
    candidates: list[list[str]] = []
    for i in track(range(0, len(records), cfg.batch_size), description="sampling"):
        candidates += generate_titles(
            model,
            tok,
            records[i : i + cfg.batch_size],
            num_return_sequences=cfg.num_candidates,
            temperature=cfg.temperature,
            max_new_tokens=cfg.max_new_tokens,
            batch_size=cfg.batch_size,
        )
    if cfg.include_original_title:
        candidates = [c + [r["title"]] for c, r in zip(candidates, records, strict=True)]

    pairs = build_pairs(records, candidates, make_judge(cfg.judge_config()), cfg.seed)
    out = Path(cfg.output_file)
    write_jsonl(pairs, out)
    n_rule = sum(p["pair_source"] == "rule" for p in pairs)
    console.print(f"[green]{len(pairs)} pairs[/] ({n_rule} rule-based) -> {out}")
    return out
