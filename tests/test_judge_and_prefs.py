from title_optimizer.evaluate import bootstrap_ci, cohen_kappa, render_markdown, win_rate
from title_optimizer.judge import RuleJudge, _combine, parse_choice, parse_score
from title_optimizer.preferences import build_pairs, is_hard_negative

DESC = "Bright flat with a private balcony overlooking the canal. Fast wifi."
REC = {"listing_id": "1", "description": DESC, "amenities": ["Wifi"], "title": "Canal flat"}


def test_position_swap_combination():
    assert _combine(1, 2) == "A"  # A won in both orders
    assert _combine(2, 1) == "B"
    assert _combine(1, 1) == "tie"  # judge always picked the first slot -> position bias -> tie
    assert _combine(None, 2) == "tie"


def test_parsers():
    assert parse_choice("1") == 1 and parse_choice("Title 2 is better") == 2
    assert parse_choice("neither") is None
    assert parse_score("8") == 8.0 and parse_score("Score: 10") == 10.0


def test_rule_judge_prefers_faithful_titles():
    j = RuleJudge()
    assert j.compare(DESC, "Canal flat with private balcony", "Canal flat with hot tub") == "A"


def test_build_pairs_rule_and_judge():
    cands = [["Canal flat with private balcony", "Canal flat with hot tub", "Flat", "Cozy flat"]]
    pairs = build_pairs([REC], cands, RuleJudge(), seed=0)
    sources = {p["pair_source"] for p in pairs}
    assert "rule" in sources
    for p in pairs:
        assert not is_hard_negative(p["chosen"][0]["content"], REC)
        assert p["prompt"][-1]["role"] == "user"


def test_metrics_helpers():
    wr = win_rate(["A", "A", "tie", "B"])
    assert wr["win_rate_vs_original"] == 0.625 and wr["wins"] == 2
    lo, hi = bootstrap_ci([0.0, 1.0] * 50)
    assert lo < 0.5 < hi
    assert cohen_kappa(["A", "B", "A"], ["A", "B", "A"]) == 1.0
    md = render_markdown(
        {
            "sft": {
                **wr,
                "hallucination_rate": 0.1,
                "length_ok_rate": 1.0,
                "style_violation_rate": 0,
            }
        },
        4,
        "rules",
    )
    assert "| sft | 62.5%" in md
