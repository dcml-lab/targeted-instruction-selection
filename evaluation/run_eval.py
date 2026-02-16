#!/usr/bin/env python3
import argparse
import shlex
import subprocess
from pathlib import Path

PYTHON = "python3"


def run(cmd: list[str]) -> None:
    """Run a command, printing it in a copy-pastable form."""
    print("\n>>", " ".join(shlex.quote(c) for c in cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified evaluation runner for minimal_multitask + custom_eval."
    )
    parser.add_argument(
        "--model_name_or_path",
        required=True,
        help="Path or HF name to the model",
    )
    parser.add_argument(
        "--save_dir",
        required=True,
        help="Base output directory for evaluation results.",
    )
    parser.add_argument(
        "--eval_dataset",
        required=True,
        choices=["gsm8k", "tydiqa", "bbh", "codex", "mmlu_pro"],
        help="Which evaluation to run.",
    )
    parser.add_argument(
        "--eval_data_dir",
        default="data/eval",
        help='Base eval data directory (default: "data/eval").',
    )

    # Behavior toggles
    parser.add_argument(
        "--zero_shot",
        action="store_true",
        help="If set: do NOT pass --use_chat_format or --apply_chat_template.",
    )

    # vLLM toggle for minimal_multitask evals (NOT for custom_eval)
    parser.add_argument(
        "--use_vllm",
        action="store_true",
        default=True,
        help="Use vLLM for minimal_multitask evals (default: enabled).",
    )

    # Allow forwarding extra args to the underlying module.
    args, unknown = parser.parse_known_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # vLLM args
    vllm_args = ["--use_vllm"]

    # Chat formatting only when NOT zero-shot (only relevant to minimal_multitask)
    chat_args: list[str] = []
    if not args.zero_shot:
        chat_args = [
            "--use_chat_format",
            "--chat_formatting_function",
            "evaluation.templates.create_prompt_with_tulu_chat_format",
        ]

    # Dataset dispatch
    if args.eval_dataset == "gsm8k":
        out_dir = save_dir / "results_gsm8k"
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            PYTHON,
            "-m",
            "evaluation.gsm.run_eval",
            "--data_dir",
            str(Path(args.eval_data_dir) / "gsm/"),
            "--save_dir",
            str(out_dir),
            "--model_name_or_path",
            args.model_name_or_path,
            "--n_shot",
            "8",
            *chat_args,
            *vllm_args,
        ]
        run(cmd)

    elif args.eval_dataset == "tydiqa":
        out_dir = save_dir / "results_tydiqa"
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            PYTHON,
            "-m",
            "evaluation.tydiqa.run_eval",
            "--data_dir",
            str(Path(args.eval_data_dir) / "tydiqa/"),
            "--n_shot",
            "1",
            "--max_context_length",
            "512",
            "--save_dir",
            str(out_dir),
            "--model_name_or_path",
            args.model_name_or_path,
            "--eval_batch_size",
            "20",
            *chat_args,
            *vllm_args,
        ]
        run(cmd)

    elif args.eval_dataset == "bbh":
        out_dir = save_dir / "results_bbh"
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            PYTHON,
            "-m",
            "evaluation.bbh.run_eval",
            "--data_dir",
            str(Path(args.eval_data_dir) / "bbh"),
            "--save_dir",
            str(out_dir),
            "--model_name_or_path",
            args.model_name_or_path,
            *chat_args,
            *vllm_args,
        ]
        run(cmd)

    elif args.eval_dataset == "codex":
        out_dir = save_dir / "results_codex"
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            PYTHON,
            "-m",
            "evaluation.codex_humaneval.run_eval",
            "--data_file",
            str(Path(args.eval_data_dir) / "codex_humaneval/HumanEval.jsonl.gz"),
            "--data_file_hep",
            str(Path(args.eval_data_dir) / "codex_humaneval/humanevalpack.jsonl"),
            *chat_args,
            "--eval_pass_at_ks",
            "10",
            "--unbiased_sampling_size_n",
            "10",
            "--temperature",
            "0.8",
            "--save_dir",
            str(out_dir),
            "--model_name_or_path",
            args.model_name_or_path,
            *vllm_args,
        ]
        run(cmd)

    elif args.eval_dataset == "mmlu_pro":
        # Requirement: run custom_eval only for mmlu_pro.
        # Also: if zero-shot, do NOT pass --apply_chat_template.
        out_dir = save_dir / "results_mmlu_pro"
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            PYTHON,
            "-m",
            "evaluation.custom_eval",
            "--model_name_or_path",
            args.model_name_or_path,
            "--output_dir",
            str(out_dir),
            "--dataset_name",
            "mmlu_pro",
        ]
        if not args.zero_shot:
            cmd.append("--apply_chat_template")
        run(cmd)

    else:
        raise ValueError(f"Unknown eval_dataset: {args.eval_dataset}")


if __name__ == "__main__":
    main()
