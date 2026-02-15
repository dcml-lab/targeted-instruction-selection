# KNN KDE is adapted from https://github.com/ZifanL/TSDS/blob/main/tsds.py.
# This implementation used cosine distance instead of L2.
import argparse
import heapq
import json
import logging
import os
from typing import List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from datasets import load_dataset
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,  # Set the logging level to INFO
    format="%(asctime)s - %(levelname)s - %(message)s",
)


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
            logging.info(f"Epoch {epoch}: Average learning rate: {np.mean(lr_values)}")
            avg_lrs.append(np.mean(lr_values))

    # normalize avg_lrs to sum to 1
    total_lr = sum(avg_lrs)
    avg_lrs = [lr / total_lr for lr in avg_lrs]

    return avg_lrs


def _prefetch_knn_argsort(cossim_matrix: np.ndarray, K: int):
    M, N = cossim_matrix.shape
    K = int(min(max(K, 2), N))

    dist_matrix = (1.0 - cossim_matrix).astype(np.float32)
    dist_matrix = np.maximum(dist_matrix, 0.0)

    sorted_indices = np.argsort(dist_matrix, axis=1, kind="mergesort")
    sorted_dist = np.take_along_axis(dist_matrix, sorted_indices, axis=1)

    return sorted_dist[:, :K], sorted_indices[:, :K]


def _knn_weighted_cosine_dist_topk_chunked(
    train_embeds: Union[torch.Tensor, Sequence[torch.Tensor]],
    avg_lrs: Union[Sequence[float], torch.Tensor],
    K: int,
    chunk_size: int = 256,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    embeds_by_epoch = (
        list(train_embeds)
        if isinstance(train_embeds, (list, tuple))
        else list(train_embeds)
    )
    T = len(embeds_by_epoch)
    P = embeds_by_epoch[0].shape[0]
    K = int(min(max(K, 1), P))

    lrs = avg_lrs if isinstance(avg_lrs, torch.Tensor) else torch.tensor(avg_lrs)
    lrs = lrs.to(dtype=torch.float32)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lrs = lrs.to(device=device, dtype=torch.float32)

    top_sim = torch.empty((P, K), device=device, dtype=torch.float32)

    for start in tqdm(
        range(0, P, chunk_size),
        desc="Computing weighted cosine similarity",
        unit="chunk",
    ):
        end = min(start + chunk_size, P)
        sim = torch.zeros((end - start, P), device=device)
        for t in range(T):
            chunk_embed = embeds_by_epoch[t][start:end].to(device=device)
            sim += lrs[t] * (chunk_embed @ embeds_by_epoch[t].to(device).T)

        vals, _ = torch.topk(sim, k=K, dim=1, largest=True, sorted=True)
        top_sim[start:end] = vals

    top_dist = (1.0 - top_sim).clamp_min(0.0)
    top_dist, _ = torch.sort(top_dist, dim=1, descending=False)
    return top_dist.cpu().numpy().astype(np.float32)


def _knn_weighted_concat_sq_l2_topk_chunked(
    train_embeds: Union[torch.Tensor, Sequence[torch.Tensor]],
    avg_lrs: Union[Sequence[float], torch.Tensor],
    K: int,
    chunk_size: int = 256,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    embeds_by_epoch = (
        list(train_embeds)
        if isinstance(train_embeds, (list, tuple))
        else list(train_embeds)
    )
    T = len(embeds_by_epoch)
    P, D = embeds_by_epoch[0].shape
    K = int(min(max(K, 1), P))

    lrs = avg_lrs if isinstance(avg_lrs, torch.Tensor) else torch.tensor(avg_lrs)
    lrs = lrs.to(dtype=torch.float32)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lrs = lrs.to(device=device, dtype=torch.float32)

    # Build concatenated embedding: [lr0*E0 | lr1*E1 | ...]  -> (P, T*D)
    parts = []
    for t in range(T):
        e = embeds_by_epoch[t].to(device=device, dtype=torch.float32)
        parts.append(e * lrs[t])
    concat_emb = torch.cat(parts, dim=1)  # (P, T*D)

    # Normalize
    concat_emb = F.normalize(concat_emb, p=2, dim=1, eps=1e-12)

    top_d2 = torch.empty((P, K), device=device, dtype=torch.float32)

    for start in range(0, P, chunk_size):
        end = min(start + chunk_size, P)
        q = concat_emb[start:end]  # (q, TD)
        d2 = torch.cdist(q, concat_emb, p=2)  # (q, P) L2
        d2 = d2 * d2  # squared L2

        vals, _ = torch.topk(
            d2, k=K, dim=1, largest=False, sorted=True
        )  # smallest distances
        top_d2[start:end] = vals

    return top_d2.cpu().numpy().astype(np.float32)


def run_tsds_kde(
    cossim_matrix: np.ndarray, train_embeddings_list: list[torch.Tensor], avg_lrs: list
) -> List[int]:
    # Hardcoded config
    # SAMPLE_SIZE = 100  # optional: if you want to return only top-K
    ALPHA = 0.01  # matches the knn unif.
    C = 5.0
    SIGMA = 0.75
    MAX_K = 5000
    KDE_K = 1000

    # Caps
    MAX_K = min(MAX_K, max(1, train_embeddings_list[0].shape[0] // 10))
    # MAX_K = train_embeddings_list[0].shape[0]
    KDE_K = min(KDE_K, max(1, train_embeddings_list[0].shape[0] // 10))

    # ---- (1) Prefetch neighbors (argsort-based) ----
    logging.info(
        f"Start prefetching {MAX_K}-nearest neighbors for each query example (argsort-based)."
    )

    # top_dists, top_indices = _prefetch_knn_argsort(xq, xb, K=MAX_K)
    # prefetch knn
    top_dists, top_indices = _prefetch_knn_argsort(cossim_matrix, K=MAX_K)
    logging.info("Completed prefetching neighbors.")

    u = len(np.unique(top_indices))
    N = train_embeddings_list[0].shape[0]
    logging.info("unique top_indices:", u, "out of", N)
    logging.info(
        "unique in first 1000 per row:", len(np.unique(top_indices.reshape(-1)))
    )

    # ---- KDE over the prefetched candidates (NO FAISS) ----
    if SIGMA == 0:
        logging.info("Sigma is zero, KDE set to 1 for all points.")
        top_kdes = np.ones_like(top_indices, dtype=np.float32)
    else:
        logging.info(f"Start computing KDE, neighborhood size: {KDE_K}.")
        top_indices_set = np.unique(top_indices.reshape(-1))
        with torch.no_grad():
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Gather top features for KDE
            top_features_by_epoch = [
                train_embeddings_list[i][top_indices_set]
                for i in range(len(train_embeddings_list))
            ]
            logging.info("Computing weighted cosine distances for KDE.")
            knn_dist = _knn_weighted_cosine_dist_topk_chunked(
                train_embeds=top_features_by_epoch,  # (T,P,D) tensor OR list of T tensors (P,D)
                avg_lrs=avg_lrs,  # length T
                K=KDE_K,
                chunk_size=256,
                device=device,
            )  # (P, KDE_K) cosine distances

        # multiplying with 2 to preserve the equivalence with squared L2 distance
        D2_equiv = 2.0 * knn_dist
        kernel = 1.0 - D2_equiv / (SIGMA**2)
        logging.info(
            f"A point has {(kernel > 0).sum(axis=-1).mean() - 1} near-duplicates on average."
        )
        kernel = kernel * (kernel > 0)
        kde = kernel.sum(axis=-1).astype(np.float32)

        kde_map = {
            int(top_indices_set[i]): float(kde[i]) for i in range(len(top_indices_set))
        }
        kde_mapfunc = np.vectorize(lambda t: kde_map[int(t)], otypes=[np.float32])
        top_kdes = kde_mapfunc(top_indices)

    # ---- Probability assignment (unchanged) ----
    logging.info("Start computing the probability assignment.")
    M, N = top_indices.shape[0], train_embeddings_list[0].shape[0]
    lastK = [0] * M
    heap = [(1.0 / top_kdes[j][0], 0, j) for j in range(M)]
    heapq.heapify(heap)
    dist_weighted_sum = [top_dists[j][0] / top_kdes[j][0] for j in range(M)]
    s = 0.0
    cost = np.zeros(M, dtype=np.float64)
    total_cost = 0.0

    while heap:
        count, curr_k, curr_j = heapq.heappop(heap)
        s = count

        total_cost -= cost[curr_j]
        cost[curr_j] = top_dists[curr_j][curr_k + 1] * count - dist_weighted_sum[curr_j]
        total_cost += cost[curr_j]

        if (ALPHA / C) * total_cost >= (1 - ALPHA) * M:
            break

        lastK[curr_j] = curr_k

        if curr_k < MAX_K - 2:
            count += 1.0 / top_kdes[curr_j][curr_k + 1]
            heapq.heappush(heap, (count, curr_k + 1, curr_j))
            dist_weighted_sum[curr_j] += (
                top_dists[curr_j][curr_k + 1] / top_kdes[curr_j][curr_k + 1]
            )
    logging.info(
        f"mean lastK+1: {np.mean(np.array(lastK)+1):.2f}, max: {np.max(np.array(lastK)+1)}"
    )

    global_probs = np.zeros(N, dtype=np.float64)
    for j in range(M):
        prob_sum = 0.0
        for k in range(lastK[j] + 1):
            global_probs[top_indices[j][k]] += 1.0 / M / s / top_kdes[j][k]
            prob_sum += 1.0 / M / s / top_kdes[j][k]
        global_probs[top_indices[j][lastK[j] + 1]] += max(1.0 / M - prob_sum, 0.0)

    logging.info(
        f"Global probabilities stats: min {global_probs.min()}, max {global_probs.max()}"
    )

    # number of non-zero probabilities
    non_zero_count = np.sum(global_probs > 0)
    logging.info(
        f"Number of candidates with non-zero probabilities: {non_zero_count} / {N}"
    )

    if non_zero_count < 10000:
        # throw error
        raise ValueError(
            f"Warning: Only {non_zero_count} candidates have non-zero probabilities. "
            f"This may lead to poor subset selection."
        )

    sorted_indices = np.lexsort(
        (np.arange(N), -global_probs)
    )  # desc prob, then asc index

    return sorted_indices


def run_tsds_knn_uniform(cossim_matrix: np.ndarray) -> List[int]:
    # same hyperparameters as in the paper
    L = 5000
    ALPHA = 0.01
    C = 5.0

    M, N = cossim_matrix.shape

    dist_matrix = 1.0 - cossim_matrix  # (M, N)
    L = min(L, N)

    # ---- GetKNN(Q, D, L): for each query i, find indices of L nearest candidates ----
    # Use argpartition to get L smallest distances, then sort those L.
    top_idx_unsorted = np.argpartition(dist_matrix, L - 1, axis=1)[:, :L]  # (M, L)
    top_dist_unsorted = np.take_along_axis(
        dist_matrix, top_idx_unsorted, axis=1
    )  # (M, L)

    order = np.argsort(top_dist_unsorted, axis=1)  # ascending by distance
    top_indices = np.take_along_axis(top_idx_unsorted, order, axis=1)  # (M, L)
    d = np.take_along_axis(top_dist_unsorted, order, axis=1)  # sorted distances, (M, L)

    # ---- Choose K using Algorithm 1's while-condition ----
    # Paper uses 1-indexing: d_{i,k}. Here d[:, 0] is k=1, d[:, K] is k=K+1.
    K = 1
    rhs = (1.0 - ALPHA) * M
    while K < L:
        d_kplus1 = d[:, K]  # (M,) = d_{i,K+1}
        gaps_sum = (
            d_kplus1[:, None] - d[:, :K]
        ).sum()  # sum_i sum_{k<=K} (d_{i,K+1}-d_{i,k})
        if (ALPHA / C) * gaps_sum < rhs:
            K += 1
        else:
            break

    # ---- Build p_j: uniform over the first K neighbors for each query ----
    probs = np.zeros(N, dtype=np.float64)
    mass = 1.0 / (K * M)
    for i in range(M):
        probs[top_indices[i, :K]] += mass

    # if non zero count is small, log a warning
    non_zero_count = np.sum(probs > 0)
    logging.info(
        f"Number of candidates with non-zero probabilities: {non_zero_count} / {N}"
    )
    if non_zero_count < 10000:
        logging.warning(
            f"Warning: Only {non_zero_count} candidates have non-zero probabilities. "
            f"This may lead to poor subset selection."
        )

    # ---- Return sorted candidate indices (desc prob, tie -> smaller index) ----
    sorted_indices = np.lexsort((np.arange(N), -probs))
    return sorted_indices.tolist()


def get_train_grads(
    train_dataset, output_dir, gradient_type, ckpt_step, proj_dim, step_size
):
    # check train_grads_{args.gradient_type}_{args.ckpt_step}_dim{args.proj_dim}.pt if present in the output_dir
    full_train_grads_path = os.path.join(
        output_dir,
        f"train_grads_{gradient_type}_ckpt{ckpt_step}_dim{proj_dim}_normalized.pt",
    )
    if os.path.exists(full_train_grads_path):
        logging.info(f"Loading train gradients from {full_train_grads_path}")
        train_grads = torch.load(full_train_grads_path, map_location="cpu")
        return train_grads
    else:
        logging.info(f"Train gradients file {full_train_grads_path} not found.")
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
                logging.info(f"Loading train gradients from {train_grads_path}")
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
        logging.info(f"Saved full train gradients to {full_train_grads_path}")

        # delete the part files ONLY after successful save
        for fpath in all_files:
            try:
                os.remove(fpath)
                logging.info(f"Deleted shard file {fpath}")
            except OSError as e:
                logging.info(f"Warning: could not delete {fpath}: {e}")

        return train_grads


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev_dataset_name", type=str, required=True)
    parser.add_argument("--embed_dir", type=str, required=True)
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
    parser.add_argument(
        "--subset_dataset_dir",
        type=str,
        required=True,
        help="Output directory for TSDS results.",
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[500, 1000, 2500, 5000, 10000],
        help="List of subset sizes to create",
    )
    parser.add_argument(
        "--selection_method",
        type=str,
        choices=["knn_kde", "knn_uniform"],
        default="knn_kde",
        help='Subset selection method to use: "knn_kde" for TSDS-KDE, "knn_uniform" for TSDS-KNN-Uniform.',
    )

    args = parser.parse_args()

    # load the cosine similarity matrix between dev and train embeddings
    sim_matrix_path = os.path.join(
        args.embed_dir, f"{args.dev_dataset_name}_cossim.npy"
    )
    cossim_matrix = np.load(sim_matrix_path)

    # load the train dataset
    train_dataset = load_dataset("Harvard-DCML/tulu-v2-197K-processed", split="train")

    if args.selection_method == "knn_uniform_cos":
        logging.info("Running TSDS-KNN-Uniform subset selection...")
        ordered_indices = run_tsds_knn_uniform(cossim_matrix)
    else:
        # load the different train embeddings for epochs
        train_embeds = []
        for ckpt_step in args.checkpoint_steps:
            train_grads = get_train_grads(
                train_dataset,
                output_dir=args.embed_dir,
                gradient_type=args.train_gradient_type,
                ckpt_step=ckpt_step,
                proj_dim=args.proj_dim,
                step_size=args.step_size,
            )
            train_embeds.append(train_grads)

        # load the avg learning rates from trainer state
        args.checkpoint_steps = sorted(args.checkpoint_steps)
        trainer_state_path = os.path.join(
            args.ckpt_dir,
            f"checkpoint-{args.checkpoint_steps[-1]}",
            "trainer_state.json",
        )
        with open(trainer_state_path, "r") as f:
            trainer_state = json.load(f)
        avg_lrs = compute_avg_lr(trainer_state)

        logging.info("Running TSDS-KDE subset selection...")
        ordered_indices = run_tsds_kde(
            cossim_matrix=cossim_matrix,
            train_embeddings_list=train_embeds,
            avg_lrs=avg_lrs,
        )

    for k in args.sizes:
        logging.info(f"Creating subset dataset for top {k} samples...")
        topk_indices = ordered_indices[:k]
        subset = train_dataset.select(topk_indices)
        subset_dataset_path = os.path.join(
            args.subset_dataset_dir, f"{args.dev_dataset_name}_subset_top{k}.jsonl"
        )
        subset.to_json(subset_dataset_path)
        logging.info(
            f"Saved subsampled dataset for method KNN KDE for top {k} samples to {subset_dataset_path}"
        )
