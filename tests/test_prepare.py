import json

import pandas as pd
import pytest

from title_optimizer.config import DataConfig
from title_optimizer.data.download import listings_url, resolve_latest_date
from title_optimizer.data.prepare import (
    clean_text,
    is_auto_generated_title,
    looks_english,
    parse_amenities,
    prepare,
    read_jsonl,
)


@pytest.mark.parametrize(
    "name,expected",
    [
        ("Rental unit in Brooklyn · ★4.9 · 1 bedroom · 1 bed · 1 bath", True),
        ("Condo in Naples · ★4.89 · 2 bedrooms · 6 beds · 1 bath", True),
        ("Home in Austin · Studio · 1 bed · 1 bath", True),
        ("Sunny loft · rooftop terrace", False),
        ("Cozy bedroom close to 'Farol da Barra'", False),
        (None, True),
    ],
)
def test_auto_generated_titles(name, expected):
    assert is_auto_generated_title(name) is expected


def test_clean_text_strips_html():
    assert clean_text("Nice flat.<br /><br /><b>The space</b><br />Big &amp; bright") == (
        "Nice flat.\n\nThe space\nBig & bright"
    )


def test_language_filter():
    assert looks_english("A lovely flat in the heart of the city with a view of the river.")
    assert not looks_english("Ein schönes Zimmer mit Blick auf den Fluss und die Altstadt.")


def test_parse_amenities():
    assert parse_amenities('["Wifi", "Hot tub"]') == ["Wifi", "Hot tub"]
    assert parse_amenities(None) == []


def test_resolve_latest_date():
    page = (
        '<a href="https://data.insideairbnb.com/united-kingdom/england/london/2026-03-14/data/'
        'listings.csv.gz">x</a><a href="https://data.insideairbnb.com/united-kingdom/england/'
        'london/2026-09-16/data/listings.csv.gz">y</a>'
    )
    assert resolve_latest_date("united-kingdom", "england", "london", page) == "2026-09-16"
    with pytest.raises(ValueError):
        resolve_latest_date("france", "ile-de-france", "paris", page)
    assert listings_url("a", "b", "c", "2026-01-01").endswith(
        "/a/b/c/2026-01-01/data/listings.csv.gz"
    )


def test_prepare_end_to_end(raw_listings, tmp_path):
    cfg = DataConfig(
        raw_dir=str(raw_listings.parent),
        processed_dir=str(tmp_path / "processed"),
        test_fraction=0.2,
        val_fraction=0.1,
    )
    stats = prepare(cfg)
    assert stats["dropped_auto_generated_titles"] == 1
    assert stats["split_train"] + stats["split_val"] + stats["split_test"] == stats["deduplicated"]

    splits = {
        s: read_jsonl(tmp_path / "processed" / f"{s}.jsonl") for s in ("train", "val", "test")
    }
    titles = {r["title"] for rows in splits.values() for r in rows}
    assert "Apartamento luminoso en el centro" not in titles  # non-English dropped
    assert "Too short" not in titles
    hosts = {s: {r["host_id"] for r in rows} for s, rows in splits.items()}
    assert not (hosts["train"] & hosts["test"]) and not (hosts["train"] & hosts["val"])
    assert all("<b>" not in r["description"] for rows in splits.values() for r in rows)

    sft = read_jsonl(tmp_path / "processed" / "sft_train.jsonl")
    assert sft, "expected some high-engagement SFT examples"
    assert sft[0]["prompt"][0]["role"] == "system"
    assert sft[0]["completion"][0]["role"] == "assistant"
    json.loads((tmp_path / "processed" / "stats.json").read_text())


def test_prepare_mixes_cities(raw_listings, tmp_path, monkeypatch):
    by_city = pd.DataFrame({"city": ["dublin"] * 50 + ["london"] * 50})
    monkeypatch.setattr("title_optimizer.data.prepare.clean", lambda df, cfg: (by_city, {}))
    seen = []

    def capture_split(df, *args):
        seen.append(df)
        return {}

    monkeypatch.setattr("title_optimizer.data.prepare.group_split", capture_split)
    processed = tmp_path / "processed"
    processed.mkdir()
    prepare(DataConfig(raw_dir=str(raw_listings.parent), processed_dir=str(processed)))
    assert set(seen[0]["city"].head(20)) == {"dublin", "london"}
