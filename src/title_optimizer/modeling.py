"""Model loading (optionally 4-bit), LoRA config, adapter merging and batched generation."""

from __future__ import annotations

from typing import Any

import torch
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

from title_optimizer.config import ModelConfig
from title_optimizer.prompts import build_messages, extract_title


def device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def use_bf16() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()


def load_tokenizer(name: str) -> PreTrainedTokenizerBase:
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model(
    name: str,
    adapter: str | list[str] | None = None,
    load_in_4bit: bool = False,
    merge: bool = True,
) -> Any:
    """Load a base model and optionally apply LoRA adapters in order (e.g. [sft, dpo]).

    Later stages are trained on top of the merged earlier stages, so evaluation must apply the
    same chain.
    """
    kwargs: dict[str, Any] = {"dtype": torch.bfloat16 if use_bf16() else torch.float32}
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        kwargs["device_map"] = "auto"
    model: Any = AutoModelForCausalLM.from_pretrained(name, **kwargs)
    adapters = [adapter] if isinstance(adapter, str) else list(adapter or [])
    if adapters and load_in_4bit and merge:
        raise ValueError(
            "Can't merge a LoRA adapter into a 4-bit model. Run `title-opt merge` first and "
            "use the merged checkpoint as base_model."
        )
    for a in adapters:
        model = PeftModel.from_pretrained(model, a)
        if merge:
            model = model.merge_and_unload()
    if not load_in_4bit:
        model.to(device())
    return model


def lora_config(cfg: ModelConfig) -> LoraConfig:
    return LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.lora_target_modules,
        task_type="CAUSAL_LM",
    )


@torch.inference_mode()
def generate_titles(
    model: Any,
    tok: PreTrainedTokenizerBase,
    records: list[dict[str, Any]],
    num_return_sequences: int = 1,
    temperature: float = 0.0,
    max_new_tokens: int = 32,
    batch_size: int = 16,
) -> list[list[str]]:
    """Return `num_return_sequences` cleaned titles per record."""
    tok.padding_side = "left"
    model.eval()
    results: list[list[str]] = []
    for i in range(0, len(records), batch_size):
        batch = records[i : i + batch_size]
        prompts = [
            tok.apply_chat_template(build_messages(r), tokenize=False, add_generation_prompt=True)
            for r in batch
        ]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(
            model.device
        )
        sample = temperature > 0
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=sample,
            temperature=temperature if sample else None,
            top_p=0.95 if sample else None,
            top_k=None,
            num_return_sequences=num_return_sequences,
            pad_token_id=tok.pad_token_id,
        )
        texts = tok.batch_decode(out[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)
        for j in range(len(batch)):
            chunk = texts[j * num_return_sequences : (j + 1) * num_return_sequences]
            results.append([extract_title(t) for t in chunk])
    return results
