import argparse
import json
import logging
import os
from typing import Optional

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import set_seed

from representation.helper import batch_cosine_similarity as calculate_influence_score

logger = logging.getLogger(__name__)


def get_train_grads(
    train_dataset, output_dir, gradient_type, ckpt_step, proj_dim, step_size
):
    # check train_grads_{args.gradient_type}_{args.ckpt_step}_dim{args.proj_dim}.pt if present in the output_dir
    full_train_grads_path = os.path.join(
        output_dir,
        f"train_grads_{gradient_type}_ckpt{ckpt_step}_dim{proj_dim}_normalized.pt",
    )
    if os.path.exists(full_train_grads_path):
        logger.info("Loading train gradients from %s", full_train_grads_path)
        train_grads = torch.load(full_train_grads_path, map_location="cpu")
        return train_grads
    else:
        logger.info("Train gradients file %s not found.", full_train_grads_path)
        # checking for paths with start and index
        all_train_grads = []
        all_files = []
        for start_idx in range(0, len(train_dataset), step_size):
            end_idx = min(start_idx + step_size, len(train_dataset))
            train_grads_path = os.path.join(
                output_dir,
                f"train_grads_{gradient_type}_ckpt{ckpt_step}_dim{proj_dim}_{start_idx}_{end_idx}_normalized.pt",
            )
            if os.path.exists(train_grads_path):
                logger.info("Loading train gradients from %s", train_grads_path)
                part_grads = torch.load(train_grads_path, map_location="cpu")
                all_train_grads.append(part_grads)
                all_files.append(train_grads_path)
            else:
                # throw an error
                raise FileNotFoundError(
                    f"Train gradients file {train_grads_path} not found."
                )
        # concatenate all parts
        train_grads = torch.cat(all_train_grads, dim=0)

        # save the full train grads for future use
        torch.save(train_grads, full_train_grads_path)
        logger.info("Saved full train gradients to %s", full_train_grads_path)

        # delete the part files ONLY after successful save
        for fpath in all_files:
            try:
                os.remove(fpath)
                logger.info("Deleted shard file %s", fpath)
            except OSError as e:
                logger.warning("Could not delete %s: %s", fpath, e)

        return train_grads


def get_dev_grads(
    dev_dataset_name,
    output_dir,
    gradient_type,
    ckpt_step,
    proj_dim,
    step_size,
):
    dev_grads_path = os.path.join(
        output_dir,
        f"{dev_dataset_name}_grads_{gradient_type}_ckpt{ckpt_step}_dim{proj_dim}_normalized.pt",
    )
    if os.path.exists(dev_grads_path):
        logger.info("Loading dev gradients from %s", dev_grads_path)
        dev_grads = torch.load(dev_grads_path)
        return dev_grads

    raise FileNotFoundError(f"Dev gradients file {dev_grads_path} not found.")


def compute_avg_lr(trainer_state, num_epochs: int = 4):
    pos = 0
    avg_lrs = []
    for epoch in range(num_epochs):
        # collect epoch learning rates
        end_epoch = epoch + 1
        lr_values = []
        epoch_values = []
        while pos < len(trainer_state["log_history"]) and (
            "epoch" in trainer_state["log_history"][pos]
            and trainer_state["log_history"][pos]["epoch"] <= end_epoch
        ):
            # collect lr values
            if "learning_rate" in trainer_state["log_history"][pos]:
                lr_values.append(trainer_state["log_history"][pos]["learning_rate"])
            # collect epoch values
            epoch_values.append(trainer_state["log_history"][pos]["epoch"])
            pos += 1
        if lr_values:
            logger.info(
                "Epoch %d: Average learning rate: %s", epoch, float(np.mean(lr_values))
            )
            avg_lrs.append(np.mean(lr_values))
    return avg_lrs


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logger.setLevel(logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_dataset_name", type=str, default="Harvard-DCML/tulu-v2-197K-processed"
    )
    parser.add_argument("--dev_dataset_name", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--checkpoint_steps", type=int, nargs="+", required=True)

    parser.add_argument("--step_size", type=int, default=10000)
    parser.add_argument("--proj_dim", type=int, default=8192)
    parser.add_argument(
        "--train_gradient_type", type=str, choices=["adam", "sgd"], default="adam"
    )
    parser.add_argument(
        "--dev_gradient_type", type=str, choices=["adam", "sgd"], default="sgd"
    )
    parser.add_argument("--num_epochs", type=int, default=4)
    parser.add_argument(
        "--random_seed", type=int, default=42, help="Random seed for shuffling rows."
    )

    args = parser.parse_args()

    # get the train dataset
    train_dataset = load_dataset(args.train_dataset_name, split="train")

    dev_dataset = load_dataset(
        "Harvard-DCML/targeted-query-set-processed", args.dev_dataset_name
    )["dev"]

    # load the learning rates from the last checkpoint trainer_state.json
    args.checkpoint_steps = sorted(args.checkpoint_steps)
    trainer_state_path = os.path.join(
        args.ckpt_dir, f"checkpoint-{args.checkpoint_steps[-1]}", "trainer_state.json"
    )
    with open(trainer_state_path, "r") as f:
        trainer_state = json.load(f)

    # get the learning rates
    avg_lrs = compute_avg_lr(trainer_state)
    logger.info("Average learning rates per epoch: %s", avg_lrs)

    # normalize the avglrs to sum to 1
    total_lr = sum(avg_lrs)
    avg_lrs = [lr / total_lr for lr in avg_lrs]
    logger.info("Normalized average learning rates per epoch: %s", avg_lrs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # compute the similarity matrix
    inf_matrix = torch.zeros((len(dev_dataset), len(train_dataset)), device=device)
    for t in range(args.num_epochs):
        # get train gradients
        avg_lr = avg_lrs[t]
        logger.info("Processing epoch %d with average learning rate %s", t, avg_lr)

        train_grads = get_train_grads(
            train_dataset,
            args.output_dir,
            args.train_gradient_type,
            args.checkpoint_steps[t],
            args.proj_dim,
            args.step_size,
        )

        dev_grads = get_dev_grads(
            args.dev_dataset_name,
            args.output_dir,
            args.dev_gradient_type,
            args.checkpoint_steps[t],
            args.proj_dim,
            args.step_size,
        )

        # compute similarity
        out = calculate_influence_score(
            dev_reps=dev_grads,
            train_reps=train_grads,
            chunk_size=256,
            device=device,
            normalize=False,  # already normalized when saving
        )

        # move avg_lr to the same device as out
        avg_lr_t = torch.tensor(avg_lr, device=device)
        inf_matrix += avg_lr_t * out
        logger.info("Completed epoch %d", t)

    # move to numpy and save as {dev_dataset_name}_cossim.npy
    inf_matrix = inf_matrix.cpu().numpy()
    sim_matrix_path = os.path.join(
        args.output_dir, f"{args.dev_dataset_name}_cossim.npy"
    )

    # shape is (num_dev, num_train)
    logger.info("Influence matrix shape: %s", inf_matrix.shape)

    np.save(sim_matrix_path, inf_matrix)
    logger.info("Saved similarity matrix to %s", sim_matrix_path)
