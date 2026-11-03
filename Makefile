.PHONY: setup data sft preferences dpo grpo eval all test test-all lint typecheck format demo

setup:            ## install everything (add `--extra cuda` on NVIDIA GPUs)
	uv sync --all-extras

data:             ## download Inside Airbnb snapshots and build splits
	uv run title-opt download
	uv run title-opt prepare

sft:
	uv run title-opt train -c configs/sft.yaml

preferences:      ## needs a Claude Code login (or JUDGE_MODEL=rules for an offline dry run)
	uv run title-opt preferences -c configs/preferences.yaml

dpo:
	uv run title-opt train -c configs/dpo.yaml

grpo:
	uv run title-opt train -c configs/grpo.yaml --judge-config configs/judge.yaml

eval:
	uv run title-opt evaluate -c configs/eval.yaml

all: data sft preferences dpo grpo eval

test:             ## fast unit tests
	uv run pytest -m "not slow"

test-all:         ## includes the end-to-end tiny-model training smoke test (~1 min on CPU)
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .

typecheck:
	uv run mypy

format:
	uv run ruff check --fix .
	uv run ruff format .

demo:
	uv run --extra app python app/app.py
