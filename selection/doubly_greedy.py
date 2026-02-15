import numpy as np


def doubly_greedy_selection(sim_matrix: np.ndarray, num_samples: int) -> list[int]:
    # get the max similarity score for each candidate sample
    max_sim_scores = np.max(sim_matrix, axis=0)
    # order the indices by descending max similarity score
    ordered_indices = np.argsort(-max_sim_scores)

    return ordered_indices.tolist()[:num_samples]
