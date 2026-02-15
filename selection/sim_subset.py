import argparse
import logging
import os

import numpy as np
from datasets import load_dataset
from transformers import set_seed

from selection.doubly_greedy import doubly_greedy_selection
from selection.round_robin import round_robin_selection
from selection.uot import uot_selection

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_dataset_name",
        type=str,
        default="Harvard-DCML/tulu-v2-197K-processed",
    )
    parser.add_argument("--dev_dataset_name", type=str, required=True)
    parser.add_argument("--selection_method", type=str, default="doubly_greedy")
    parser.add_argument("--subset_dataset_dir", type=str, required=True)
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[500, 1000, 2500, 5000, 10000, 25000],
        help="List of subset sizes to create",
    )
    parser.add_argument("--similarity_matrix_path", type=str, required=False)
    parser.add_argument("--eps", type=float, default=0.01)
    parser.add_argument("--reg_m1", type=float, default=float("inf"))
    parser.add_argument("--reg_m2", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    set_seed(args.seed)

    train_dataset = load_dataset(args.train_dataset_name, split="train")
    dev_dataset = load_dataset(
        "Harvard-DCML/targeted-query-set-processed", args.dev_dataset_name
    )["dev"]

    # load the similarity matrix
    if args.similarity_matrix_path is None:
        raise ValueError(
            "Either similarity_matrix_path or distance_matrix_path must be provided"
        )

    logger.info("Loading similarity matrix from %s", args.similarity_matrix_path)
    sim_matrix = np.load(args.similarity_matrix_path)

    max_size = max(args.sizes)

    if args.selection_method == "doubly_greedy":
        logger.info("Creating ordered stratification using doubly greedy method...")
        ordered_indices = doubly_greedy_selection(sim_matrix, max_size)
    elif args.selection_method == "uot":
        logger.info("Creating ordered stratification using UOT sum-based method...")
        ordered_indices = uot_selection(
            cossim_clipped=sim_matrix,
            eps=args.eps,
            reg_m1=args.reg_m1,
            reg_m2=args.reg_m2,
            num_samples=max_size,
        )
    elif args.selection_method == "round_robin":
        logger.info("Creating ordered stratification using round robin method...")
        ordered_indices = round_robin_selection(sim_matrix, max_size)
    else:
        raise ValueError(f"Unknown selection method: {args.selection_method}")

    for k in args.sizes:
        logger.info("Creating subset dataset for top %d samples...", k)
        topk_indices = ordered_indices[:k]
        subset = train_dataset.select(topk_indices)
        subset_dataset_path = os.path.join(
            args.subset_dataset_dir, f"{args.dev_dataset_name}_subset_top{k}.jsonl"
        )
        subset.to_json(subset_dataset_path)
        logger.info(
            "Saved subsampled dataset for method %s for top %d samples to %s",
            args.selection_method,
            k,
            subset_dataset_path,
        )


if __name__ == "__main__":
    main()
