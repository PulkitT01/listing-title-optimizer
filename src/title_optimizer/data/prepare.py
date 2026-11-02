"""Clean Inside Airbnb listings and build leakage-free train/val/test splits.

Important data caveat: since mid-2023 the `name` column in many Inside Airbnb snapshots holds
Airbnb's auto-generated card subtitle (e.g. "Rental unit in Brooklyn · ★4.9 · 1 bedroom · 1 bed")
instead of the host-written title. Those rows are useless as title targets, so we detect and drop
them and report how many were removed. `description` is still host-written.
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rich.console import Console

from title_optimizer.config import DataConfig
from title_optimizer.prompts import build_messages

console = Console()

COLUMNS = [
    "id",
    "host_id",
    "name",
    "description",
    "amenities",
    "property_type",
    "room_type",
    "neighbourhood_cleansed",
    "number_of_reviews",
    "reviews_per_month",
    "review_scores_rating",
]

_AUTO_TITLE_HINTS = re.compile(r"★|\b(bedrooms?|beds?|baths?|studio|half-bath)\b", re.IGNORECASE)
_TAGS = re.compile(r"<[^>]+>")
_WS = re.compile(r"[ \t]+")
_EN_STOPWORDS = {
    "the", "and", "a", "to", "of", "in", "is", "with", "for", "you", "this", "on", "our", "your",
    "it", "from", "are", "an", "be", "has", "at", "all", "we", "room", "apartment",
}  # fmt: skip


def is_auto_generated_title(name: str) -> bool:
    """True for Airbnb's generated subtitles like 'Condo in Naples · ★4.89 · 2 bedrooms'."""
    if not isinstance(name, str):
        return True
    return " · " in name and bool(_AUTO_TITLE_HINTS.search(name))


def clean_text(text: Any) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = html.unescape(_TAGS.sub(" ", text))
    lines = [_WS.sub(" ", ln).strip() for ln in text.splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def looks_english(text: str, min_ratio: float = 0.08) -> bool:
    """Cheap language filter: ASCII-heavy and contains common English function words."""
    if not text:
        return False
    ascii_ratio = sum(ch.isascii() for ch in text) / len(text)
    words = re.findall(r"[a-zA-Z']+", text.lower())
    if not words or ascii_ratio < 0.9:
        return False
    return sum(w in _EN_STOPWORDS for w in words) / len(words) >= min_ratio


def parse_amenities(raw: Any) -> list[str]:
    if isinstance(raw, list):
        return [str(a) for a in raw]
    if not isinstance(raw, str) or not raw.strip():
        return []
    try:
        return [str(a) for a in json.loads(raw)]
    except json.JSONDecodeError:
        return [a.strip(' "{}') for a in raw.split(",") if a.strip(' "{}')]


def load_raw(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for p in paths:
        df = pd.read_csv(p, usecols=lambda c: c in COLUMNS, low_memory=False)
        df["city"] = p.name.split("_")[0]
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def clean(df: pd.DataFrame, cfg: DataConfig) -> tuple[pd.DataFrame, dict[str, int]]:
    stats = {"raw": len(df)}

    df = df.dropna(subset=["name", "description"]).copy()
    stats["has_name_and_description"] = len(df)

    auto = df["name"].map(is_auto_generated_title)
    stats["dropped_auto_generated_titles"] = int(auto.sum())
    df = df[~auto]

    df["title"] = df["name"].astype(str).str.strip()
    df["description"] = df["description"].map(clean_text)
    df = df[df["title"].str.len().between(cfg.min_title_chars, cfg.max_title_chars)]
    df = df[df["description"].str.len() >= cfg.min_description_chars]
    df["description"] = df["description"].str.slice(0, cfg.max_description_chars)
    stats["length_filtered"] = len(df)

    if cfg.language == "en":
        df = df[df["description"].map(looks_english) & df["title"].map(looks_english_title)]
    stats["language_filtered"] = len(df)

    df = df.drop_duplicates(subset=["title", "description"])
    stats["deduplicated"] = len(df)

    df["amenities"] = df["amenities"].map(parse_amenities)
    df["reviews_per_month"] = pd.to_numeric(df["reviews_per_month"], errors="coerce").fillna(0.0)
    df["review_scores_rating"] = pd.to_numeric(df["review_scores_rating"], errors="coerce")

    # "High engagement" = enough reviews, well rated, and in the top part of reviews/month *within
    # its city* (cities differ a lot in base demand). Confounded by price, photos and location:
    # treat it as a weak signal, not ground truth.
    eligible = (df["number_of_reviews"] >= cfg.min_reviews) & (
        df["review_scores_rating"] >= cfg.min_rating
    )
    threshold = (
        df[eligible]
        .groupby("city")["reviews_per_month"]
        .transform(lambda s: s.quantile(cfg.engagement_quantile))
    )
    df["high_engagement"] = eligible & (df["reviews_per_month"] >= threshold.reindex(df.index))
    stats["high_engagement"] = int(df["high_engagement"].sum())
    return df, stats


def looks_english_title(title: str) -> bool:
    return bool(title) and sum(ch.isascii() for ch in title) / len(title) >= 0.9


def group_split(
    df: pd.DataFrame, test_fraction: float, val_fraction: float, seed: int
) -> dict[str, pd.DataFrame]:
    """Split by host so a host's templated titles can't leak from train into test."""
    hosts = np.array(sorted(df["host_id"].astype(str).unique()))  # sorted: order-independent
    rng = np.random.default_rng(seed)
    rng.shuffle(hosts)
    n_test = int(len(hosts) * test_fraction)
    n_val = int(len(hosts) * val_fraction)
    test_h, val_h = set(hosts[:n_test]), set(hosts[n_test : n_test + n_val])
    host = df["host_id"].astype(str)
    return {
        "train": df[~host.isin(test_h | val_h)],
        "val": df[host.isin(val_h)],
        "test": df[host.isin(test_h)],
    }


def to_record(row: pd.Series) -> dict[str, Any]:
    rating = row.get("review_scores_rating")
    return {
        "listing_id": str(row["id"]),
        "host_id": str(row["host_id"]),
        "city": row["city"],
        "title": row["title"],
        "description": row["description"],
        "amenities": list(row["amenities"]),
        "property_type": row.get("property_type") or "",
        "room_type": row.get("room_type") or "",
        "neighbourhood": row.get("neighbourhood_cleansed") or "",
        "reviews_per_month": float(row["reviews_per_month"]),
        "rating": None if pd.isna(rating) else float(rating),
        "high_engagement": bool(row["high_engagement"]),
    }


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def sft_example(record: dict[str, Any]) -> dict[str, Any]:
    """Conversational prompt/completion format understood by TRL's SFTTrainer."""
    return {
        "prompt": build_messages(record),
        "completion": [{"role": "assistant", "content": record["title"]}],
    }


def prepare(cfg: DataConfig, raw_paths: list[Path] | None = None) -> dict[str, int]:
    raw_paths = raw_paths or sorted(Path(cfg.raw_dir).glob("*_listings.csv.gz"))
    if not raw_paths:
        raise FileNotFoundError(f"No *_listings.csv.gz in {cfg.raw_dir}. Run `title-opt download`.")

    df, stats = clean(load_raw(raw_paths), cfg)
    splits = group_split(df, cfg.test_fraction, cfg.val_fraction, cfg.seed)
    out = Path(cfg.processed_dir)

    for name, part in splits.items():
        records = [to_record(r) for _, r in part.iterrows()]
        write_jsonl(records, out / f"{name}.jsonl")
        stats[f"split_{name}"] = len(records)
        if name in ("train", "val"):
            sft = [sft_example(r) for r in records if r["high_engagement"]]
            write_jsonl(sft, out / f"sft_{name}.jsonl")
            stats[f"sft_{name}"] = len(sft)

    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    for k, v in stats.items():
        console.print(f"{k:>32}: {v:,}")
    return stats
