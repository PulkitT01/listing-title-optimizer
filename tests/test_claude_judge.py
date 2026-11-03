import subprocess

import pytest

from title_optimizer.config import JudgeConfig
from title_optimizer.judge import ClaudeCliJudge, RuleJudge, make_judge


@pytest.fixture
def judge(tmp_path, monkeypatch):
    monkeypatch.delenv("JUDGE_MODEL", raising=False)
    cfg = JudgeConfig(model="claude-haiku-5-5", max_workers=1)
    return ClaudeCliJudge(cfg, tmp_path / "cache.jsonl")


def fake_claude(monkeypatch, *replies):
    calls = []
    outputs = iter(replies)

    def run(command, **kwargs):
        calls.append((command, kwargs))
        code, stdout = next(outputs)
        return subprocess.CompletedProcess(command, code, stdout=stdout, stderr="")

    monkeypatch.setattr("title_optimizer.judge.subprocess.run", run)
    monkeypatch.setattr("title_optimizer.judge.time.sleep", lambda seconds: None)
    return calls


def test_calls_are_isolated_and_cached(judge, monkeypatch):
    calls = fake_claude(monkeypatch, (0, "7\n"))
    assert judge.score_many(["listing"], ["title"]) == [7.0]
    assert judge.score_many(["listing"], ["title"]) == [7.0]
    assert len(calls) == 1
    command, kwargs = calls[0]
    assert command[command.index("--model") + 1] == "claude-haiku-5-5"
    assert command[command.index("--tools") + 1] == ""
    assert command[command.index("--setting-sources") + 1] == ""
    assert "judge-" in kwargs["cwd"] and "listing-title-optimizer" not in kwargs["cwd"]


def test_each_pairwise_order_is_a_separate_call(judge, monkeypatch):
    calls = fake_claude(monkeypatch, (0, "1"), (0, "2"))
    assert judge.compare_many(["listing"], ["title a"], ["title b"]) == ["A"]
    assert len(calls) == 2
    assert calls[0][0][2] != calls[1][0][2]


def test_transient_failure_is_retried(judge, monkeypatch):
    calls = fake_claude(monkeypatch, (1, ""), (0, "8"))
    assert judge.score_many(["listing"], ["title"]) == [8.0]
    assert len(calls) == 2


@pytest.mark.parametrize("reply", [(1, ""), (0, "Limit reached, resets at 5pm")])
def test_failed_calls_stop_the_run_and_are_not_cached(judge, monkeypatch, reply):
    fake_claude(monkeypatch, reply, reply, reply)
    with pytest.raises(RuntimeError, match="Claude judge call failed"):
        judge.score_many(["listing"], ["title"])
    assert not judge.cache_path.exists()


def test_make_judge_picks_the_claude_judge_without_an_api_key(monkeypatch):
    monkeypatch.delenv("JUDGE_MODEL", raising=False)
    monkeypatch.delenv("JUDGE_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert isinstance(make_judge(JudgeConfig(model="claude-haiku-5-5")), ClaudeCliJudge)


def test_judge_model_rules_still_overrides(monkeypatch):
    monkeypatch.setenv("JUDGE_MODEL", "rules")
    assert isinstance(make_judge(JudgeConfig(model="claude-haiku-5-5")), RuleJudge)
