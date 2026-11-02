"""Prompt construction and output parsing, shared by training, evaluation and the demo."""

from __future__ import annotations

import re
from typing import Any

SYSTEM_PROMPT = (
    "You write Airbnb listing titles. Given a listing, reply with ONE title and nothing else.\n"
    "Rules:\n"
    "- At most 50 characters; 25-45 is ideal.\n"
    "- Lead with the most distinctive, appealing feature (view, location, standout amenity).\n"
    "- Only mention features that the listing actually states. Never invent amenities.\n"
    "- Sentence case. No emojis, no ALL CAPS, no '!!!', no quotes, no prefix like 'Title:'.\n"
    "- Don't repeat what guests already see (city name, bed count) unless it's the hook."
)

MAX_AMENITIES_IN_PROMPT = 20


def format_listing(record: dict[str, Any]) -> str:
    parts = []
    if record.get("property_type"):
        parts.append(f"Property type: {record['property_type']}")
    if record.get("neighbourhood"):
        parts.append(f"Neighbourhood: {record['neighbourhood']}")
    amenities = record.get("amenities") or []
    if amenities:
        parts.append("Amenities: " + ", ".join(amenities[:MAX_AMENITIES_IN_PROMPT]))
    parts.append(f"Description:\n{record['description']}")
    return "\n".join(parts)


def build_messages(record: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": format_listing(record)},
    ]


_PREFIX = re.compile(r"^\s*(?:\*\*)?(?:listing\s+)?title\s*[:\-]\s*", re.IGNORECASE)
_WRAPPING = "\"'`*“”‘’ "


def extract_title(text: str) -> str:
    """Turn a raw completion into a clean title (first non-empty line, no prefix/quotes)."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    line = next((ln for ln in text.strip().splitlines() if ln.strip()), "")
    line = _PREFIX.sub("", line)
    return line.strip(_WRAPPING).strip()


def completion_text(completion: Any) -> str:
    """TRL passes completions as plain strings or as [{'role': 'assistant', 'content': ...}]."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion and isinstance(completion[-1], dict):
        return str(completion[-1].get("content", ""))
    return str(completion)
