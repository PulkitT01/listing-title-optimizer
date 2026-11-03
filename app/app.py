"""Gradio demo: paste a listing description, get title suggestions with a faithfulness check.

Deploy to Hugging Face Spaces by copying this file and setting MODEL_ID / ADAPTERS, e.g.
MODEL_ID=Qwen/Qwen3-1.7B ADAPTERS=<your-hf-user>/title-optimizer-sft,<your-hf-user>/title-optimizer-grpo
"""

from __future__ import annotations

import os

import gradio as gr

from title_optimizer.modeling import generate_titles, load_model, load_tokenizer
from title_optimizer.rewards import style_violations, unsupported_claims

MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3-1.7B")
ADAPTERS = [a for a in os.environ.get("ADAPTERS", "outputs/sft,outputs/grpo").split(",") if a]

tok = load_tokenizer(MODEL_ID)
# Local adapters are used if they exist; anything else is treated as a Hugging Face Hub repo id.
model = load_model(
    MODEL_ID,
    adapter=[a for a in ADAPTERS if os.path.exists(a) or not a.startswith(("outputs/", ".", "/"))],
)


def suggest(description: str, amenities: str, n: int, temperature: float):
    record = {
        "description": description.strip(),
        "amenities": [a.strip() for a in amenities.split(",") if a.strip()],
    }
    titles = generate_titles(
        model, tok, [record], num_return_sequences=int(n), temperature=temperature
    )[0]
    rows = []
    for t in dict.fromkeys(titles):
        claims = unsupported_claims(t, record["description"], record["amenities"])
        issues = style_violations(t)
        status = "✅ ok" if not claims and not issues else "⚠️ " + ", ".join(claims + issues)
        rows.append([t, len(t), status])
    return rows


demo = gr.Interface(
    fn=suggest,
    inputs=[
        gr.Textbox(lines=10, label="Listing description"),
        gr.Textbox(label="Amenities (comma-separated, optional)"),
        gr.Slider(1, 8, value=4, step=1, label="Number of suggestions"),
        gr.Slider(0.1, 1.2, value=0.8, label="Temperature"),
    ],
    outputs=gr.Dataframe(headers=["Title", "Chars", "Check"], label="Suggestions"),
    title="Listing Title Optimizer",
    description="Titles from a small open model preference-tuned with DPO/GRPO. "
    "The check column flags features the title claims but the listing never mentions.",
)

if __name__ == "__main__":
    demo.launch()
