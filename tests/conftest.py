"""Offline fixtures: a fake Inside Airbnb snapshot and a tiny randomly initialised chat model.

Nothing here touches the network, so CI is fast and never rate-limited.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

DESC = (
    "Bright one bedroom flat with a private balcony overlooking the canal. "
    "Fast wifi and a dedicated workspace, five minutes to the station and the market. "
    "The kitchen is fully equipped and there is a washing machine for longer stays."
)

ROWS = [
    # (id, host_id, name, description, amenities, n_reviews, reviews_per_month, rating)
    (1, 10, "Canal-view flat with private balcony", DESC, ["Wifi", "Dedicated workspace"], 50, 3.1, 4.9),
    (2, 11, "Rental unit in London · ★4.85 · 1 bedroom · 1 bed · 1 bath", DESC, ["Wifi"], 40, 2.0, 4.85),
    (3, 12, "Quiet studio near the market", DESC, ["Wifi"], 12, 0.4, 4.8),
    (4, 13, "Apartamento luminoso en el centro", "Apartamento muy luminoso con balcón, cerca de todo y con wifi rápido para trabajar en casa. " * 3, ["Wifi"], 30, 2.5, 4.9),
    (5, 14, "Sunny loft with workspace & fast wifi", DESC, ["Wifi", "Dedicated workspace"], 80, 4.0, 4.95),
    (6, 15, "Too short", "short", ["Wifi"], 5, 1.0, 4.9),
    (7, 16, "Garden flat close to the station", DESC.replace("balcony", "garden"), ["Garden"], 25, 1.2, 4.75),
    (8, 17, "Balcony flat by the canal and market", DESC, ["Wifi"], 9, 0.9, 4.6),
] + [
    (100 + i, 200 + i, f"Comfortable canal flat number {i}", DESC, ["Wifi"], 10 + i, 0.5 + i / 10, 4.8)
    for i in range(24)
]  # fmt: skip


@pytest.fixture
def raw_listings(tmp_path: Path) -> Path:
    df = pd.DataFrame(
        [
            {
                "id": r[0],
                "host_id": r[1],
                "name": r[2],
                "description": r[3].replace(". ", ".<br /><b>") + "</b>",
                "amenities": json.dumps(r[4]),
                "property_type": "Entire rental unit",
                "room_type": "Entire home/apt",
                "neighbourhood_cleansed": "Hackney",
                "number_of_reviews": r[5],
                "reviews_per_month": r[6],
                "review_scores_rating": r[7],
                "price": "$120.00",  # extra column that must be ignored
            }
            for r in ROWS
        ]
    )
    raw = tmp_path / "raw"
    raw.mkdir()
    path = raw / "london_2026-09-16_listings.csv.gz"
    df.to_csv(path, index=False, compression="gzip")
    return path


CHAT_TEMPLATE = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n"
    "{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory: pytest.TempPathFactory) -> str:
    """A 2-layer Qwen2-architecture model with a byte-level BPE tokenizer trained on the fly."""
    from tokenizers import ByteLevelBPETokenizer
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    from title_optimizer.prompts import SYSTEM_PROMPT

    out = tmp_path_factory.mktemp("tiny-model")
    specials = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
    bpe = ByteLevelBPETokenizer()
    corpus = [SYSTEM_PROMPT, DESC, *[r[2] for r in ROWS], "user assistant system"] * 5
    bpe.train_from_iterator(corpus, vocab_size=600, min_frequency=1, special_tokens=specials)
    tok = PreTrainedTokenizerFast(
        tokenizer_object=bpe._tokenizer,
        eos_token="<|im_end|>",
        pad_token="<|endoftext|>",
        bos_token=None,
    )
    tok.chat_template = CHAT_TEMPLATE
    tok.save_pretrained(out)

    config = Qwen2Config(
        vocab_size=len(tok),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=2048,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
        tie_word_embeddings=True,
    )
    Qwen2ForCausalLM(config).save_pretrained(out)
    return str(out)
