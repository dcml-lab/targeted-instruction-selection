import argparse
import logging
import os
import pickle
import sys
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
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

    if debug:
        train_dataset = train_dataset.select(range(100))

    def _process_example(example):
        messages = example["messages"]

        if len(messages) == 0:
            raise ValueError("messages field is empty.")

        def _concat_messages(messages):
            message_text = ""
            for message in messages:
                if message["role"] == "system":
                    message_text += "<|system|>\n" + message["content"].strip() + "\n"
                elif message["role"] == "user":
                    message_text += "<|user|>\n" + message["content"].strip() + "\n"
                elif message["role"] == "assistant":
                    message_text += (
                        "<|assistant|>\n"
                        + message["content"].strip()
                        + tokenizer.eos_token
                        + "\n"
                    )
                else:
                    raise ValueError("Invalid role: {}".format(message["role"]))
            # add bos token if needed
            if add_bos_token:
                message_text = tokenizer.bos_token + message_text
            return message_text

        add_bos_token = tokenizer.bos_token is not None
        text = _concat_messages(messages).strip()
        return {"text": text}

    train_dataset = train_dataset.map(
        lambda x: _process_example(x),
        num_proc=1,
        load_from_cache_file=False,
        remove_columns=train_dataset.column_names,
    )

    return train_dataset


def l2_normalize_batch(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Row-wise L2 normalize a 2D tensor.
    Args:
        x: [N, D] tensor
        eps: small value to avoid division by zero
    Returns:
        [N, D] tensor with unit L2 norm per row
    """
    if not torch.is_tensor(x):
        x = torch.as_tensor(x)
    x = x.float()
    return F.normalize(x, p=2, dim=1, eps=eps)


def compute_train_embeddings(
    model,
    tokenizer,
    train_dataset_path: str,
    start_index: int = 0,
    end_index: int = None,
    batch_size: int = 1,
    debug: bool = False,
) -> torch.Tensor:
    """
    Loads the train split and returns L2-normalized embeddings as a torch.Tensor [N, D].
    `model` is expected to support SentenceTransformers-style `.encode`.
    """
    train_dataset = load_train_dataset(
        train_dataset_path,
        tokenizer,
        start_index,
        end_index,
        debug,
    )

    texts: List[str] = train_dataset["text"]

    if debug:
        # Keep a small slice to speed things up in debug mode
        texts = texts[:1024]

    logger.info("Train dataset loaded with %d samples.", len(texts))

    all_train_embeds_np: np.ndarray = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,  # we'll normalize ourselves
    )

    all_train_embeds = torch.from_numpy(all_train_embeds_np)
    all_train_embeds = l2_normalize_batch(all_train_embeds)

    return all_train_embeds


def compute_eval_embeddings(
    model, tokenizer, eval_dataset_name: str, split: str, batch_size: int
) -> torch.Tensor:
    eval_dataset = load_dataset(
        "Harvard-DCML/targeted-query-set-processed", eval_dataset_name, split=split
    )

    # log one sample
    logger.info("Example eval sample[0]: %s", eval_dataset[0])

    PROMPT = (
        "Instruct: Given a sample, find the passages closest to that sample.\nQuery:"
    )

    def process_example(example):
        return {"text": f'{PROMPT} {example["prompts"]} {example["labels"]}'.strip()}

    eval_dataset = eval_dataset.map(
        lambda x: process_example(x),
        num_proc=16,
        load_from_cache_file=False,
        remove_columns=eval_dataset.column_names,
    )
    texts: List[str] = eval_dataset["text"]
    logger.info("Eval dataset loaded with %d samples.", len(texts))

    eval_embeddings_np: np.ndarray = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,  # we'll normalize ourselves
    )
    eval_embeddings = torch.from_numpy(eval_embeddings_np)
    eval_embeddings = l2_normalize_batch(eval_embeddings)

    return eval_embeddings


def _ensure_unit_norm(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / x.norm(dim=1, keepdim=True).clamp_min(eps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", type=str, default="sentence-transformers/gtr-t5-base"
    )
    parser.add_argument("--save_dir", type=str, default="files/index/gtr-t5-base")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_dataset_path", type=str, default=None)
    parser.add_argument(
        "--train_dataset_name",
        type=str,
        default="Harvard-DCML/tulu-v2-197K-processed",
    )
    parser.add_argument("--train_index_path", type=str, default=None)

    parser.add_argument("--dev_dataset_name", type=str, default="mmlu_pro")
    parser.add_argument("--dev_index_path", type=str, default=None)

    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument(
        "--start_index",
        type=int,
        default=0,
        help="start index for the training dataset",
    )
    parser.add_argument(
        "--end_index", type=int, default=None, help="end index for the training dataset"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logger.setLevel(logging.INFO)

    logger.info("Loading model %s with dtype %s", args.model_name, args.dtype)
    kwargs = {"torch_dtype": torch.bfloat16, "device_map": "auto"}
    model = SentenceTransformer(args.model_name, model_kwargs=kwargs)

    logger.info(
        "Model loaded with %.2f billion parameters",
        sum(p.numel() for p in model.parameters()) / 1e9,
    )

    if args.end_index is not None and args.end_index <= args.start_index:
        raise ValueError("end_index must be greater than start_index")

    tokenizer = model.tokenizer

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
            batch_size=args.batch_size,
            start_index=args.start_index,
            end_index=args.end_index,
            debug=args.debug,
        )

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

    logger.info(
        "Computing cosine similarity between dev and train embeddings (batched)..."
    )

    if args.start_index == 0 and args.end_index is None:
        out_path = os.path.join(args.save_dir, f"{args.dev_dataset_name}_cossim.npy")
    else:
        out_path = os.path.join(
            args.save_dir,
            f"{args.dev_dataset_name}_cossim_{args.start_index}_{args.end_index}.npy",
        )

    cossim_matrix = batch_cosine_similarity(
        dev_reps=all_dev_embeds,
        train_reps=all_train_embeds,
        chunk_size=1024,  # tune based on GPU RAM
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        normalize=False,  # already normalized
    )
    np.save(out_path, cossim_matrix.cpu().numpy())
    logger.info(
        "Cosine similarity matrix saved to %s with shape %s and dtype %s",
        out_path,
        tuple(cossim_matrix.shape),
        cossim_matrix.dtype,
    )


if __name__ == "__main__":
    main()
