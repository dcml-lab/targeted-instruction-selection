import argparse
import json
import os

import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from common.data import construct_test_sample


def compute_loss(model, dataset, batch_size=1):
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False
    )
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(dataloader):
            outputs = model(
                input_ids=batch["input_ids"].to(model.device),
                attention_mask=batch["attention_mask"].to(model.device),
                labels=batch["labels"].to(model.device),
            )
            total_loss += outputs.loss.item() * batch["input_ids"].size(0)
    return total_loss / len(dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="Path to the model name on Huggingface or the local path to the model checkpoint.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size for computing the loss."
    )
    parser.add_argument(
        "--eval_dataset_name",
        type=str,
        default="mmlu_pro",
        help="Name of the evaluation dataset.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the output loss file.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )

    args = parser.parse_args()

    set_seed(args.seed)

    print(args)

    kwargs = {"device_map": "auto", "torch_dtype": "auto", "low_cpu_mem_usage": True}
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, **kwargs)
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **kwargs)
    model.eval()

    loss_dict = {}

    for split in ["dev"]:
        eval_dataset = load_dataset(
            "Harvard-DCML/targeted-query-set-processed",
            args.eval_dataset_name,
            split=split,
        )
        eval_dataset = eval_dataset.map(
            lambda x: construct_test_sample(
                sample=x,
                tokenizer=tokenizer,
                max_length=2048,
            )
        )
        eval_dataset.set_format(
            type="torch", columns=["input_ids", "attention_mask", "labels"]
        )

        # print stats
        print(
            f"Computing loss for {args.eval_dataset_name} {split} dataset with {len(eval_dataset)} samples..."
        )
        ce_loss = compute_loss(model, eval_dataset, batch_size=args.batch_size)
        loss_dict[split] = ce_loss
        print(f"loss for {split} dataset: {ce_loss:.4f}")

    # if the args.output_path directory is not present, create it
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    with open(args.output_path, "w+") as f:
        json.dump(loss_dict, f, indent=4)

    print(f"losses saved to {args.output_path}")
