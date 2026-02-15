"""
Creates the quantiles for LESS, RDS+, and EMBED.
"""

import argparse
import logging
import os

import numpy as np
from datasets import load_dataset

from selection.round_robin import round_robin_selection

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_dataset",
        type=str,
        default="Harvard-DCML/tulu-v2-197K-processed",
    )
    parser.add_argument("--dev_dataset_name", type=str, required=True)
    parser.add_argument("--subset_dataset_dir", type=str, required=True)
    parser.add_argument(
        "--num_samples",
        type=int,
        default=500,
        help="Number of samples to select",
    )
    parser.add_argument(
        "--num_quantiles",
        type=int,
        default=10,
        help=(
            "Number of quantiles to create. "
            "When --focus_quantile0 is used, this is both the number of coarse quantiles "
            "over the full dataset and the number of sub-quantiles within quantile 0."
        ),
    )
    parser.add_argument("--similarity_matrix_path", type=str, required=True)
    parser.add_argument(
        "--focus_quantile0",
        action="store_true",
        help=(
            "If set, first conceptually partition the ordered data into num_quantiles coarse quantiles "
            "and then only further split coarse quantile 0 into num_quantiles sub-quantiles."
        ),
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logger.setLevel(logging.INFO)

    train_dataset = load_dataset(args.train_dataset, split="train")
    num_train = len(train_dataset)

    dev_dataset = load_dataset(
        "Harvard-DCML/targeted-query-set-processed", args.dev_dataset_name
    )["dev"]
    num_dev = len(dev_dataset)

    # Similarity matrix is required by CLI, so this will always run, but we keep the guard.
    if args.similarity_matrix_path is not None:
        logger.info("Loading similarity matrix from %s", args.similarity_matrix_path)
        sim_matrix = np.load(args.similarity_matrix_path)
    else:
        raise ValueError("similarity_matrix_path must be provided")

    logger.info("Creating ordered stratification using round robin method...")
    ordered_indices = round_robin_selection(sim_matrix, num_train)

    # Optional: focus only on coarse quantile 0 and then re-quantile that subset.
    if args.focus_quantile0:
        coarse_num_quantiles = args.num_quantiles
        coarse_quant_size = len(ordered_indices) // coarse_num_quantiles

        if coarse_num_quantiles == 1:
            # trivial case: whole dataset is quantile 0
            quant0_indices = ordered_indices
        else:
            # quantile 0 is the first coarse_quant_size elements
            quant0_indices = ordered_indices[:coarse_quant_size]

        logger.info("=== Focus-on-quantile 0 mode enabled ===")
        logger.info("Total ordered indices (full dataset): %d", len(ordered_indices))
        logger.info(
            "Using num_quantiles=%d as coarse quantiles, coarse quantile 0 size: %d",
            args.num_quantiles,
            len(quant0_indices),
        )

        # Now only work with quantile 0 and re-split it into args.num_quantiles quantiles
        ordered_indices = quant0_indices
        num_quantiles = args.num_quantiles
    else:
        num_quantiles = args.num_quantiles

    quantiles: list[list[int]] = []
    quant_size = len(ordered_indices) // num_quantiles

    logger.info("Creating quantiles...")
    logger.info("Total ordered indices used for this run: %d", len(ordered_indices))
    logger.info("Quantile size: %d", quant_size)

    for q in range(num_quantiles):
        start_idx = q * quant_size
        if q == num_quantiles - 1:
            end_idx = len(ordered_indices)
        else:
            end_idx = (q + 1) * quant_size
        quant_indices = ordered_indices[start_idx:end_idx]
        quantiles.append(quant_indices)

    # From each quantile, get the top num_samples samples
    for q in range(num_quantiles):
        quant_indices = quantiles[q]
        logger.info("Quantile %d: %d samples", q, len(quant_indices))
        top_indices = quant_indices[: args.num_samples]
        logger.info("Top %d indices from quantile %d", args.num_samples, q)

        subsampled_dataset = train_dataset.select(top_indices)

        # Different naming when refining only quantile 0
        if args.focus_quantile0:
            quant_label = f"quantile0_subquantile{q}"
        else:
            quant_label = f"quantile{q}"

        subset_dataset_path = os.path.join(
            args.subset_dataset_dir,
            f"{args.dev_dataset_name}_subset_{quant_label}_top{args.num_samples}.jsonl",
        )
        subsampled_dataset.to_json(subset_dataset_path)
        logger.info("Saved subset dataset to %s", subset_dataset_path)


if __name__ == "__main__":
    main()
