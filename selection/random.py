import argparse
import os

import numpy as np
from datasets import Dataset, load_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_dataset",
        type=str,
        default="Harvard-DCML/tulu-v2-197K-processed",
    )
    parser.add_argument("--subset_dataset_dir", type=str, required=True)
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[500, 1000, 2500, 5000, 10000],
        help="List of subset sizes to create",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for shuffling")
    args = parser.parse_args()

    train_dataset = load_dataset(args.train_dataset, split="train")
    num_train = len(train_dataset)

    # shuffle the dataset
    print("Shuffling the dataset...")
    train_dataset = train_dataset.shuffle(seed=args.seed)

    for size in args.sizes:
        print(f"Creating subset of size {size}...")
        subset_dataset = train_dataset.select(range(size))
        subset_path = os.path.join(
            args.subset_dataset_dir, f"subset_size_{size}_seed_{args.seed}.jsonl"
        )
        print(f"Saving subset to {subset_path}...")
        subset_dataset.to_json(subset_path)
