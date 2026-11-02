"""Verifiable rewards for titles. Used three ways:

1. as GRPO reward functions (TRL signature: `fn(completions, **dataset_columns) -> list[float]`),
2. as hard filters when building DPO preference pairs,
3. as evaluation metrics (hallucination rate, length compliance, style violations).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from title_optimizer.prompts import completion_text, extract_title

MAX_CHARS = 50
TARGET_RANGE = (20, 45)

# (claim name, pattern that detects the claim in a title, pattern that counts as evidence in the
# description/amenities). A title claim without evidence is a hallucination.
CLAIMS: list[tuple[str, str, str]] = [
    ("hot tub", r"hot ?tub|jacuzzi|\bspa\b", r"hot ?tub|jacuzzi|whirlpool|\bspa\b"),
    ("pool", r"\bpool\b", r"\bpool\b"),
    ("sauna", r"\bsauna\b", r"\bsauna\b"),
    ("fireplace", r"fire ?place|wood ?stove", r"fire ?place|wood[- ]?(burning )?stove"),
    ("balcony", r"\bbalcon(y|ies)\b", r"\bbalcon(y|ies)\b"),
    ("terrace", r"\bterrace\b|\bpatio\b|\bdeck\b", r"\bterrace\b|\bpatio\b|\bdeck\b"),
    ("rooftop", r"\brooftop\b|\broof ?top\b", r"\broof ?top\b|\broof terrace\b"),
    ("garden", r"\bgarden\b|\bbackyard\b", r"\bgarden\b|\bbackyard\b|\byard\b"),
    ("parking", r"\bparking\b|\bgarage\b", r"\bparking\b|\bgarage\b|\bdriveway\b"),
    ("gym", r"\bgym\b|fitness", r"\bgym\b|fitness"),
    ("ev charger", r"\bev\b|charger", r"\bev charger\b|electric vehicle|\bcharger\b"),
    ("bbq", r"\bbbq\b|barbecue|\bgrill\b", r"\bbbq\b|barbecue|\bgrill\b"),
    (
        "beach",
        r"beach ?front|on the beach|beach access|steps to the beach",
        r"beach ?front|beach access|on the beach|steps (from|to) the beach|minutes? to the beach",
    ),
    ("waterfront", r"water ?front|lake ?front|river ?front", r"water ?front|lake ?front|river ?front"),
    ("pet friendly", r"pet[- ]friendly|dogs? welcome", r"pets? allowed|pet[- ]friendly|dogs?"),
    ("workspace", r"work ?space|\bwfh\b|home office", r"workspace|\bdesk\b|\boffice\b"),
    ("air conditioning", r"\ba/?c\b|air[- ]con", r"air[- ]?condition|\ba/?c\b"),
    ("king bed", r"\bking[- ]?(size )?bed\b", r"\bking\b"),
    ("sea view", r"\b(sea|ocean) ?views?\b", r"\b(sea|ocean)\b.{0,20}\bviews?\b|\bviews?\b.{0,25}\b(sea|ocean)\b"),
    ("lake view", r"\blake ?views?\b", r"\blake\b.{0,20}\bviews?\b|\bviews?\b.{0,25}\blake\b"),
    ("mountain view", r"\bmountain ?views?\b", r"\bmountains?\b.{0,20}\bviews?\b|\bviews?\b.{0,25}\bmountains?\b"),
    ("city view", r"\b(city|skyline) ?views?\b", r"\b(city|skyline)\b.{0,20}\bviews?\b|\bviews?\b.{0,25}\b(city|skyline)\b"),
]  # fmt: skip
_COMPILED = [
    (name, re.compile(t, re.IGNORECASE), re.compile(e, re.IGNORECASE)) for name, t, e in CLAIMS
]

_EMOJI = re.compile(
    "[\U0001f300-\U0001faff\U00002600-\U000027bf\U0001f000-\U0001f2ff\U00002b00-\U00002bff]"
)


def unsupported_claims(title: str, description: str, amenities: Sequence[str] = ()) -> list[str]:
    evidence = description + "\n" + "\n".join(amenities)
    return [
        name
        for name, in_title, in_evidence in _COMPILED
        if in_title.search(title) and not in_evidence.search(evidence)
    ]


def style_violations(raw_completion: str) -> list[str]:
    """Checks follow Airbnb's title guidance: no emojis/symbol spam, sentence case, one line."""
    text = raw_completion.strip()
    title = extract_title(text)
    v = []
    if not title:
        v.append("empty")
    if len([ln for ln in text.splitlines() if ln.strip()]) > 1:
        v.append("multiline")
    if title != text.strip(" \"'"):
        v.append("extra_text")  # prefixes like "Title:", quotes or markdown
    if _EMOJI.search(title):
        v.append("emoji")
    if re.search(r"([!?*#~])\1", title):
        v.append("repeated_punctuation")
    letters = [c for c in title if c.isalpha()]
    caps_words = [w for w in re.findall(r"[A-Za-z]{4,}", title) if w.isupper()]
    if letters and (sum(c.isupper() for c in letters) / len(letters) > 0.6 or len(caps_words) >= 2):
        v.append("all_caps")
    return v


def length_score(title: str, max_chars: int = MAX_CHARS, target=TARGET_RANGE) -> float:
    n = len(title)
    if n == 0 or n > max_chars:
        return -1.0
    lo, hi = target
    if lo <= n <= hi:
        return 1.0
    if n < lo:
        return n / lo
    return 1.0 - (n - hi) / (max_chars - hi + 1)


# ---- TRL GRPO reward functions ------------------------------------------------------------


def length_reward(completions: list[Any], **_: Any) -> list[float]:
    return [length_score(extract_title(completion_text(c))) for c in completions]


def style_reward(completions: list[Any], **_: Any) -> list[float]:
    return [max(-1.0, 1.0 - 0.5 * len(style_violations(completion_text(c)))) for c in completions]


def faithfulness_reward(
    completions: list[Any], description: list[str], amenities: list[list[str]], **_: Any
) -> list[float]:
    """+1 if every feature claimed in the title is backed by the listing, -1 per unsupported one."""
    out = []
    for c, desc, am in zip(completions, description, amenities, strict=True):
        bad = unsupported_claims(extract_title(completion_text(c)), desc, am or [])
        out.append(1.0 if not bad else max(-2.0, -1.0 * len(bad)))
    return out


def make_judge_reward(judge: Any):
    """Wrap an LLM judge (pointwise 1-10 score) as a GRPO reward in [-1, 1]."""

    def judge_reward(completions: list[Any], description: list[str], **_: Any) -> list[float]:
        titles = [extract_title(completion_text(c)) for c in completions]
        scores = judge.score_many(description, titles)
        return [(s - 5.5) / 4.5 for s in scores]

    judge_reward.__name__ = "judge_reward"
    return judge_reward


REWARD_FUNCS = {
    "length": length_reward,
    "style": style_reward,
    "faithfulness": faithfulness_reward,
}
