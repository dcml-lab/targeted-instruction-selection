from itertools import cycle

import numpy as np


def round_robin_selection(sim_matrix: np.ndarray, num_samples: int) -> list[int]:
    if num_samples > sim_matrix.shape[1]:
        raise ValueError("num_samples > number of columns")
    scores = sim_matrix.copy()
    used = np.zeros(scores.shape[1], bool)
    picked = []
    for i in tqdm(cycle(range(scores.shape[0])), desc="Round Robin Selection"):
        if len(picked) >= num_samples:
            break
        row = np.where(~used, scores[i], -np.inf)
        j = int(np.argmax(row))
        used[j] = True
        picked.append(j)
    return picked
