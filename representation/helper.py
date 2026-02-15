# include code for cossim multiplication with batches since it is used acroos all the representation methods

from typing import Optional

import torch
from tqdm import tqdm


@torch.no_grad()
def _l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / x.norm(dim=1, keepdim=True).clamp_min(eps)


@torch.no_grad()
def batch_cosine_similarity(
    dev_reps: torch.Tensor,  # (N_dev, D)
    train_reps: torch.Tensor,  # (N_train, D)
    chunk_size: int = 256,
    device: Optional[torch.device] = None,
    normalize: bool = True,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Returns (N_dev, N_train). If normalize=False, returns dot products.
    """
    if dev_reps.ndim != 2 or train_reps.ndim != 2:
        raise ValueError("Inputs must be 2D: dev (N_dev, D), train (N_train, D).")
    if dev_reps.shape[1] != train_reps.shape[1]:
        raise ValueError("Feature dims must match.")

    device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if normalize:
        dev_reps = _l2_normalize(dev_reps, eps)
        train_reps = _l2_normalize(train_reps, eps)

    N_dev = dev_reps.shape[0]
    N_train = train_reps.shape[0]
    out = torch.empty((N_dev, N_train), dtype=train_reps.dtype, device="cpu")

    dev = dev_reps.to(device=device, dtype=train_reps.dtype, non_blocking=True)

    for j in tqdm(
        range(0, N_train, chunk_size), desc="Cosine similarity", unit="chunk"
    ):
        j_end = min(j + chunk_size, N_train)
        train = train_reps[j:j_end].to(
            device=device, dtype=train_reps.dtype, non_blocking=True
        )
        out[:, j:j_end] = (dev @ train.T).cpu()

    return out
