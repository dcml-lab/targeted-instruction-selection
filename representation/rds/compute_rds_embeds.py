import argparse
import logging
import os
import pickle
import sys
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from common.data import construct_test_sample, encode_with_messages_format
from representation.helper import batch_cosine_similarity

logger = logging.getLogger(__name__)


def load_train_dataset(
    train_dataset_path: str,
    tokenizer: AutoTokenizer,
    start_index: int = 0,
    end_index: int = None,
    debug: bool = False,
) -> torch.utils.data.Dataset:
    if train_dataset_path is not None and os.path.exists(train_dataset_path):
        # assuming it is json
        train_dataset = load_dataset("json", data_files=[train_dataset_path])["train"]
    else:
        train_dataset = load_dataset(
            "Harvard-DCML/tulu-v2-197K-processed", split="train"
        )

    if end_index is not None:
        train_dataset = train_dataset.select(range(start_index, end_index))
        logger.info(
            "Selected training dataset from index %d to %d", start_index, end_index
        )

    if debug:
        train_dataset = train_dataset.select(range(100))

    train_dataset = train_dataset.map(
        lambda x: encode_with_messages_format(
            example=x, tokenizer=tokenizer, max_seq_length=2048, include_response=True
        ),
        num_proc=16,
    )
    logger.info("Number of training examples: %d", len(train_dataset))

    return train_dataset


def compute_rds_embeddings(
    model, dataset, pooling="weighted_mean", batch_size=1
) -> torch.Tensor:
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False
    )
    model.eval()
    all_embeddings = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            outputs = model(
                input_ids=batch["input_ids"].to(model.device),
                attention_mask=batch["attention_mask"].to(model.device),
                output_hidden_states=True,
            )
            hidden_states = outputs.hidden_states[-1]
            if pooling == "mean":
                embeddings = torch.mean(hidden_states, dim=1)
            elif pooling == "weighted_mean":
                # SGPT idea: https://arxiv.org/abs/2202.08904
                weighting_mask = (
                    torch.arange(
                        hidden_states.size(1), device=hidden_states.device
                    ).unsqueeze(0)
                    + 1
                )
                weighting_mask = weighting_mask / weighting_mask.sum(
                    dim=1, keepdim=True
                )
                embeddings = torch.sum(
                    hidden_states * weighting_mask.unsqueeze(-1), dim=1
                )
            else:
                raise ValueError(f"Unsupported pooling method: {pooling}")
            # normalize the embeddings
            embeddings = F.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings.cpu())

    all_embeddings = torch.cat(all_embeddings, dim=0)
    return all_embeddings


def compute_train_embeddings(
    model,
    tokenizer,
    train_dataset_path=None,
    pooling="weighted_mean",
    batch_size=1,
    start_index=0,
    end_index=None,
    debug=False,
) -> torch.Tensor:
    train_dataset = load_train_dataset(
        train_dataset_path,
        tokenizer=tokenizer,
        start_index=start_index,
        end_index=end_index,
        debug=debug,
    )
    train_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"]
    )
    train_embeddings = compute_rds_embeddings(
        model=model, dataset=train_dataset, pooling=pooling, batch_size=batch_size
    )
    return train_embeddings


def compute_eval_embeddings(
    model,
    tokenizer,
    eval_dataset_name,
    split="dev",
    pooling="weighted_mean",
    batch_size=1,
) -> torch.Tensor:
    """
    Compute embeddings for the evaluation dataset.
    """
    eval_dataset = load_dataset(
        "Harvard-DCML/targeted-query-set-processed", eval_dataset_name, split=split
    )

    # log one sample
    logger.info("Example eval sample[0]: %s", eval_dataset[0])

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
    eval_embeddings = compute_rds_embeddings(
        model=model, dataset=eval_dataset, pooling=pooling, batch_size=batch_size
    )
    return eval_embeddings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-3B")
    parser.add_argument("--save_dir", type=str, default="files/index/rds_embeds")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--train_dataset_name",
        type=str,
        default="Harvard-DCML/tulu-v2-197K-processed",
    )
    parser.add_argument("--train_index_path", type=str, default=None)

    parser.add_argument("--dev_dataset_name", type=str, default=None)
    parser.add_argument("--dev_index_path", type=str, default=None)

    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--pooling", type=str, default="weighted_mean"
    )  # none, mean, weighted_mean

    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=None)
    parser.add_argument("--debug", action="store_true")

    # NOTE: Your original code references args.train_dataset_path but never defines it.
    # If you have it elsewhere, add the argument back. If not, this prevents AttributeError.
    parser.add_argument("--train_dataset_path", type=str, default=None)

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logger.setLevel(logging.INFO)

    assert args.pooling in ["none", "mean", "weighted_mean"]

    if args.dtype == "bf16":
        kwargs = {"torch_dtype": torch.bfloat16}
    elif args.dtype == "fp16":
        kwargs = {"torch_dtype": torch.float16}
    elif args.dtype == "fp32":
        kwargs = {"torch_dtype": torch.float32}
    else:
        raise ValueError(f"Unsupported dtype: {args.dtype}")

    logger.info("Loading model %s with dtype %s", args.model_name, args.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        **kwargs,
        device_map="auto",  # use multiple gpus if you can
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    logger.info(
        "Model loaded with %.2f billion parameters",
        sum(p.numel() for p in model.parameters()) / 1e9,
    )

    if args.end_index is not None and args.end_index <= args.start_index:
        raise ValueError("end_index must be greater than start_index")

    os.makedirs(args.save_dir, exist_ok=True)
    if args.start_index == 0 and args.end_index is None:
        if not args.train_index_path:
            args.train_index_path = os.path.join(args.save_dir, "train_embeds.pt")
    else:
        if not args.train_index_path:
            args.train_index_path = os.path.join(
                args.save_dir,
                f"train_embeds_{args.start_index}_{args.end_index}.pt",
            )

    if args.train_index_path is not None and os.path.exists(args.train_index_path):
        all_train_embeds = torch.load(args.train_index_path)
        logger.info(
            "Loaded train embeddings from %s; shape: %s",
            args.train_index_path,
            tuple(all_train_embeds.shape),
        )
    else:
        all_train_embeds = compute_train_embeddings(
            model=model,
            tokenizer=tokenizer,
            train_dataset_path=args.train_dataset_path,
            pooling=args.pooling,
            batch_size=args.batch_size,
            start_index=args.start_index,
            end_index=args.end_index,
            debug=args.debug,
        )
        if not args.train_index_path:
            args.train_index_path = os.path.join(args.save_dir, "train_embeds.pt")

        # save the train embeddings
        os.makedirs(os.path.dirname(args.train_index_path), exist_ok=True)
        with open(args.train_index_path, "wb") as f:
            torch.save(all_train_embeds, f)

        logger.info(
            "Train embeddings computed and saved to %s; shape: %s",
            args.train_index_path,
            tuple(all_train_embeds.shape),
        )

    if args.dev_index_path is not None and os.path.exists(args.dev_index_path):
        all_dev_embeds = torch.load(args.dev_index_path)
        logger.info(
            "Loaded dev embeddings from %s; shape: %s",
            args.dev_index_path,
            tuple(all_dev_embeds.shape),
        )
    else:
        all_dev_embeds = compute_eval_embeddings(
            model=model,
            tokenizer=tokenizer,
            eval_dataset_name=args.dev_dataset_name,
            split="dev",
            pooling=args.pooling,
            batch_size=args.batch_size,
        )
        if not args.dev_index_path:
            args.dev_index_path = os.path.join(
                args.save_dir, f"{args.dev_dataset_name}_dev_embeds.pt"
            )

        with open(args.dev_index_path, "wb") as f:
            torch.save(all_dev_embeds, f)

        logger.info(
            "Dev embeddings computed and saved to %s; shape: %s",
            args.dev_index_path,
            tuple(all_dev_embeds.shape),
        )

    if args.start_index == 0 and args.end_index is None:
        out_path = os.path.join(args.save_dir, f"{args.dev_dataset_name}_cossim.npy")
    else:
        out_path = os.path.join(
            args.save_dir,
            f"{args.dev_dataset_name}_cossim_{args.start_index}_{args.end_index}.npy",
        )

    if os.path.exists(out_path):
        logger.info("Cosine similarity matrix already exists at %s", out_path)
        return

    cossim_matrix = batch_cosine_similarity(
        dev_reps=all_dev_embeds,
        train_reps=all_train_embeds,
        chunk_size=256,
        device=None,
        normalize=True,  # cosine similarity
    )

    # Save similarity matrix
    np.save(out_path, cossim_matrix)
    logger.info("Saved similarity matrix to %s", out_path)


if __name__ == "__main__":
    main()
