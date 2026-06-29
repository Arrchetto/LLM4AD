from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist


class GetData:
    """Generate reproducible Euclidean TSP training instances."""

    def __init__(
        self,
        n_instance: int = 50,
        problem_size: int = 100,
        seeds: list[int] | tuple[int, ...] | range | None = None,
    ):
        self.n_instance = int(n_instance)
        self.problem_size = int(problem_size)
        if seeds is None:
            seeds = range(self.n_instance)
        self.seeds = tuple(seeds)
        if len(self.seeds) != self.n_instance:
            raise ValueError(
                f"seed count ({len(self.seeds)}) must equal n_instance ({self.n_instance})"
            )

    def generate_instances(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """Return a list of (coordinates, distance_matrix) tuples."""
        instances: list[tuple[np.ndarray, np.ndarray]] = []
        for seed in self.seeds:
            rng = np.random.default_rng(seed)
            coordinates = rng.uniform(0.0, 1.0, size=(self.problem_size, 2))
            distance_matrix = cdist(coordinates, coordinates, metric="euclidean")
            instances.append((coordinates, distance_matrix))
        return instances
