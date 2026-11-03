"""LLM-as-judge for titles, plus a rule-based judge for offline runs and tests.

Pairwise comparisons are asked twice with the order swapped; if the two answers disagree the
result is a tie. This removes most position bias. Measure the judge against your own labels with
`title-opt judge-agreement` before trusting it.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Literal

from title_optimizer.config import JudgeConfig
from title_optimizer.rewards import length_score, style_violations, unsupported_claims

Verdict = Literal["A", "B", "tie"]

RUBRIC = (
    "You are an expert Airbnb listing copywriter judging listing titles.\n"
    "Judge on, in order of importance:\n"
    "1. Accuracy: the title must not claim anything the listing doesn't state.\n"
    "2. Appeal: would it make a traveller click? Leads with the most distinctive feature.\n"
    "3. Specificity: concrete details beat generic words like 'cozy' or 'nice'.\n"
    "4. Form: at most 50 characters, sentence case, no emojis or ALL CAPS."
)


def _pairwise_prompt(description: str, first: str, second: str) -> str:
    return (
        f"{RUBRIC}\n\nLISTING:\n{description}\n\nTITLE 1: {first}\nTITLE 2: {second}\n\n"
        "Which title is better? Answer with exactly one character: 1 or 2."
    )


def _pointwise_prompt(description: str, title: str) -> str:
    return (
        f"{RUBRIC}\n\nLISTING:\n{description}\n\nTITLE: {title}\n\n"
        "Rate the title from 1 (bad) to 10 (excellent). Answer with just the number."
    )


def parse_choice(text: str) -> int | None:
    m = re.search(r"\b([12])\b", text)
    return int(m.group(1)) if m else None


def parse_score(text: str) -> float | None:
    m = re.search(r"\b(10|[1-9])(?:\.\d+)?\b", text)
    return float(m.group(0)) if m else None


def _combine(first_order: int | None, swapped_order: int | None) -> Verdict:
    a_first = first_order == 1
    a_swapped = swapped_order == 2
    if first_order is None or swapped_order is None or a_first != a_swapped:
        return "tie"
    return "A" if a_first else "B"


class LLMJudge:
    def __init__(self, cfg: JudgeConfig, cache_path: str | Path = ".cache/judge.jsonl"):
        self.cfg = cfg
        self.model = os.environ.get("JUDGE_MODEL", cfg.model)
        self.client = self._make_client()
        self.cache_path = Path(cache_path)
        self.cache: dict[str, str] = {}
        self._lock = threading.Lock()
        if self.cache_path.exists():
            for line in self.cache_path.read_text().splitlines():
                item = json.loads(line)
                self.cache[item["k"]] = item["v"]

    def _make_client(self) -> Any:
        from openai import OpenAI

        api_key = os.environ.get(self.cfg.api_key_env) or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(f"Set ${self.cfg.api_key_env} (or use judge model 'rules' offline).")
        return OpenAI(
            api_key=api_key, base_url=self.cfg.base_url or os.environ.get("JUDGE_BASE_URL")
        )

    def _complete(self, prompt: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.cfg.temperature,
            max_tokens=5,
        )
        return resp.choices[0].message.content or ""

    def _ask(self, prompt: str) -> str:
        key = hashlib.sha256(f"{self.model}\n{prompt}".encode()).hexdigest()
        if key in self.cache:
            return self.cache[key]
        answer = self._complete(prompt)
        with self._lock:
            self.cache[key] = answer
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with self.cache_path.open("a") as f:
                f.write(json.dumps({"k": key, "v": answer}) + "\n")
        return answer

    def _map(self, fn, items):
        with ThreadPoolExecutor(self.cfg.max_workers) as pool:
            return list(pool.map(fn, items))

    def compare(self, description: str, title_a: str, title_b: str) -> Verdict:
        return self.compare_many([description], [title_a], [title_b])[0]

    def compare_many(self, descriptions, titles_a, titles_b) -> list[Verdict]:
        prompts = []
        for d, a, b in zip(descriptions, titles_a, titles_b, strict=True):
            prompts += [_pairwise_prompt(d, a, b), _pairwise_prompt(d, b, a)]
        answers = [parse_choice(x) for x in self._map(self._ask, prompts)]
        return [_combine(answers[i], answers[i + 1]) for i in range(0, len(answers), 2)]

    def score_many(self, descriptions, titles) -> list[float]:
        prompts = [_pointwise_prompt(d, t) for d, t in zip(descriptions, titles, strict=True)]
        return [parse_score(x) or 1.0 for x in self._map(self._ask, prompts)]


class ClaudeCliJudge(LLMJudge):
    """Judge through the Claude Code login. No API key, and no temperature control.

    Every prompt is a fresh `claude -p` process in a neutral folder, so the two orders of a
    pairwise question stay independent and no project context reaches the judge. A failed call
    raises instead of returning, because the parsers would read digits in an error as an answer.
    """

    ISOLATION = (
        "--tools",
        "",
        "--no-session-persistence",
        "--strict-mcp-config",
        "--setting-sources",
        "",
        "--disable-slash-commands",
        "--system-prompt",
        "Follow the user's instructions exactly.",
    )

    def _make_client(self) -> None:
        return None

    def _complete(self, prompt: str) -> str:
        command = ["claude", "-p", prompt, "--model", self.model, *self.ISOLATION]
        error = ""
        for attempt in range(3):
            time.sleep(30 * attempt)
            with tempfile.TemporaryDirectory(prefix="judge-") as cwd:
                proc = subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=120)
            answer = proc.stdout.strip()
            if proc.returncode == 0 and re.fullmatch(r"10|[1-9]", answer):
                return answer
            error = (proc.stderr or proc.stdout).strip()[:200]
        raise RuntimeError(f"Claude judge call failed: {error}")


class RuleJudge:
    """Deterministic stand-in judge built from the verifiable rewards. No API needed."""

    def score_many(self, descriptions, titles) -> list[float]:
        out = []
        for d, t in zip(descriptions, titles, strict=True):
            s = 5.5 + 2.0 * length_score(t) - 1.5 * len(style_violations(t))
            s -= 3.0 * len(unsupported_claims(t, d))
            out.append(min(10.0, max(1.0, s)))
        return out

    def compare_many(self, descriptions, titles_a, titles_b) -> list[Verdict]:
        sa, sb = self.score_many(descriptions, titles_a), self.score_many(descriptions, titles_b)
        return ["A" if a > b else "B" if b > a else "tie" for a, b in zip(sa, sb, strict=True)]

    def compare(self, description: str, title_a: str, title_b: str) -> Verdict:
        return self.compare_many([description], [title_a], [title_b])[0]


def make_judge(cfg: JudgeConfig) -> LLMJudge | ClaudeCliJudge | RuleJudge:
    model = os.environ.get("JUDGE_MODEL", cfg.model)
    if model == "rules":
        return RuleJudge()
    if model.startswith("claude-"):
        return ClaudeCliJudge(cfg)
    return LLMJudge(cfg)
