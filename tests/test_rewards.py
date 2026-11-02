import pytest

from title_optimizer.prompts import extract_title
from title_optimizer.rewards import (
    faithfulness_reward,
    length_reward,
    length_score,
    style_reward,
    style_violations,
    unsupported_claims,
)

DESC = "Bright flat with a private balcony and views of the river. Fast wifi, dedicated workspace."


@pytest.mark.parametrize(
    "title,claims",
    [
        ("Bright flat with private balcony", []),
        ("Riverside flat with hot tub", ["hot tub"]),
        ("Flat with pool, sauna and balcony", ["pool", "sauna"]),
        ("Work-ready flat with dedicated workspace", []),
        ("Sea view flat by the river", ["sea view"]),
        ("Flat near King's Cross", []),
    ],
)
def test_unsupported_claims(title, claims):
    assert unsupported_claims(title, DESC) == claims


def test_amenities_count_as_evidence():
    assert unsupported_claims("Flat with hot tub", DESC, ["Hot tub", "Wifi"]) == []


def test_length_score():
    assert length_score("") == -1.0
    assert length_score("x" * 51) == -1.0
    assert length_score("x" * 30) == 1.0
    assert 0 < length_score("x" * 10) < 1
    assert 0 < length_score("x" * 48) < 1


@pytest.mark.parametrize(
    "raw,violation",
    [
        ("Title: Sunny loft", "extra_text"),
        ('"Sunny loft"', None),
        ("Sunny loft 🌞", "emoji"),
        ("AMAZING LOFT WITH VIEWS", "all_caps"),
        ("Sunny loft!!", "repeated_punctuation"),
        ("Sunny loft\nAlso: great views", "multiline"),
        ("Sunny loft near the NYC subway", None),
    ],
)
def test_style_violations(raw, violation):
    v = style_violations(raw)
    assert (violation in v) if violation else v == []


def test_extract_title():
    assert (
        extract_title('**Title:** "Sunny loft with views"\n\nWhy: ...') == "Sunny loft with views"
    )
    assert extract_title("<think>hmm</think>\nCanal flat") == "Canal flat"


def test_trl_reward_signatures_accept_conversational_completions():
    completions = [
        [{"role": "assistant", "content": "Bright flat with private balcony"}],
        [{"role": "assistant", "content": "Flat with hot tub and pool!!"}],
    ]
    kw = {"description": [DESC, DESC], "amenities": [[], []], "prompts": [None, None]}
    assert faithfulness_reward(completions, **kw) == [1.0, -2.0]
    assert style_reward(completions, **kw)[0] == 1.0
    assert style_reward(completions, **kw)[1] < 1.0
    assert all(-1 <= x <= 1 for x in length_reward(completions, **kw))
