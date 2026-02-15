# Tasks: medqa_4options, drop, groundcocoa, truthfulqa_mc2, coqa, medmcqa, mmlu_pro

import argparse
import json
import os
import tempfile

from lm_eval import evaluator
from transformers import AutoTokenizer

# Tasks that should use vllm backend
VLLM_TASKS = {"mmlu_pro"}

# Tulu-style chat template
TULU_CHAT_TEMPLATE_JINJA = r"""{% for message in messages %}
{% if message['role'] == 'system' %}
<|system|>
{{ message['content'] }}
{% elif message['role'] == 'user' %}
<|user|>
{{ message['content'] }}
{% elif message['role'] == 'assistant' %}
<|assistant|>
{{ message['content'] | trim }}{{ eos_token }}
{% else %}
{{ raise_exception("Tulu chat template only supports 'system', 'user' and 'assistant' roles. Got: " ~ message['role']) }}
{% endif %}
{% endfor %}
<|assistant|>
"""


def prepare_tokenizer_with_tulu_template(model_path: str) -> str:
    """Load tokenizer, add Tulu template, save to temp dir."""
    tok = AutoTokenizer.from_pretrained(model_path, use_fast=True)

    if tok.eos_token is None:
        tok.eos_token = "</s>"

    tok.chat_template = TULU_CHAT_TEMPLATE_JINJA

    tmp_dir = tempfile.mkdtemp(prefix="tulu_tokenizer_")
    tok.save_pretrained(tmp_dir)
    return tmp_dir


def run_eval(
    model_path: str,
    dataset_name: str,
    output_dir: str,
    apply_chat_template: bool = False,
):
    """Run lm-eval on one dataset and save JSON."""
    os.makedirs(output_dir, exist_ok=True)

    if apply_chat_template:
        print("Using chat template")
        tok_dir = prepare_tokenizer_with_tulu_template(model_path)
        model_args = f"pretrained={model_path},tokenizer={tok_dir}"
    else:
        print("Not using chat template")
        model_args = f"pretrained={model_path}"

    # Pick backend
    if dataset_name in VLLM_TASKS:
        backend = "vllm"
    else:
        backend = "hf"

    print(f"Backend: {backend}")
    print(f"Dataset: {dataset_name}")

    results = evaluator.simple_evaluate(
        model=backend,
        model_args=model_args,
        tasks=[dataset_name],
        num_fewshot=0,
        batch_size="auto",
        apply_chat_template=apply_chat_template,
        log_samples=False,
    )

    out_path = os.path.join(output_dir, f"{dataset_name}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="HF model ID or local path",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Single dataset to evaluate",
    )
    parser.add_argument(
        "--apply_chat_template",
        action="store_true",
        help="Enable Tulu chat template",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory for JSON output",
    )

    args = parser.parse_args()

    run_eval(
        model_path=args.model_path,
        dataset_name=args.dataset_name,
        output_dir=args.output_dir,
        apply_chat_template=args.apply_chat_template,
    )
