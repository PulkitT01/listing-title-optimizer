import pytest
from transformers import AutoTokenizer

from title_optimizer.config import TrainConfig
from title_optimizer.modeling import load_tokenizer
from title_optimizer.train import _common_args


def test_load_tokenizer_turns_off_thinking(tiny_model_dir, tmp_path):
    tok = AutoTokenizer.from_pretrained(tiny_model_dir)
    tok.chat_template = (
        "{% if enable_thinking is defined and enable_thinking is false %}direct"
        "{% else %}thinking{% endif %}"
    )
    tok.save_pretrained(tmp_path)
    messages = [{"role": "user", "content": "x"}]
    assert load_tokenizer(str(tmp_path)).apply_chat_template(messages, tokenize=False) == "direct"


@pytest.mark.parametrize("dev,expected", [("cuda", True), ("mps", True), ("cpu", False)])
def test_gradient_checkpointing_runs_on_accelerators(monkeypatch, dev, expected):
    monkeypatch.setattr("title_optimizer.train.device", lambda: dev)
    assert _common_args(TrainConfig())["gradient_checkpointing"] is expected
