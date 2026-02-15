import numpy as np
from ot.unbalanced import sinkhorn_unbalanced
from tqdm import tqdm


def uot_selection(
    cossim_clipped: np.ndarray,
    eps: float = 0.01,
    reg_m1: float = float("inf"),
    reg_m2: float = 0.0001,
    num_samples: int = None,
):
    if cossim_clipped.ndim != 2:
        raise ValueError("cossim_clipped must be 2D")

    n_sources, n_targets = cossim_clipped.shape
    C = (1.0 - cossim_clipped) / 2.0

    a = np.ones(n_sources, dtype=np.float64) / n_sources
    b = np.ones(n_targets, dtype=np.float64) / n_targets

    T = sinkhorn_unbalanced(
        a,
        b,
        C,
        reg=eps,
        reg_m=(reg_m1, reg_m2),
        reg_type="kl",
    )

    influence_sum = T.sum(axis=0)

    target_indices = np.argsort(-influence_sum)

    if num_samples is not None:
        target_indices = target_indices[:num_samples]

    return target_indices
