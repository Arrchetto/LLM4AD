from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy.spatial.distance import cdist

from llm4ad.base import Evaluation

from .template import task_description, template_program

__all__ = [
    "TSPEoHMatrixEvaluation",
    "normalize_tour",
    "tour_cost",
    "validate_distance_matrix",
]


def validate_distance_matrix(distance_matrix: Any) -> np.ndarray:
    """Return a finite, non-empty square distance matrix as float64."""
    try:
        matrix = np.asarray(distance_matrix, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("distance matrix must contain numeric values") from error
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("distance matrix must be a non-empty two-dimensional square matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("distance matrix must contain only finite values")
    return matrix


def normalize_tour(tour: Any, city_count: int) -> list[int]:
    """Validate and normalize a candidate permutation without coercing values."""
    if isinstance(tour, np.ndarray):
        if tour.ndim != 1:
            raise ValueError("tour NumPy array must be one-dimensional")
        values: Sequence[Any] = tour.tolist()
    elif isinstance(tour, list):
        values = tour
    else:
        raise ValueError("tour must be a Python list or one-dimensional NumPy array")

    if len(values) != city_count:
        raise ValueError(f"tour length must be {city_count}, got {len(values)}")

    normalized: list[int] = []
    for position, city in enumerate(values):
        if isinstance(city, (bool, np.bool_)) or not isinstance(city, (int, np.integer)):
            raise ValueError(f"tour city at position {position} must be an integer")
        city_index = int(city)
        if city_index < 0 or city_index >= city_count:
            raise ValueError(f"tour city {city_index} is outside [0, {city_count})")
        normalized.append(city_index)

    if len(set(normalized)) != city_count:
        raise ValueError("tour must contain every city exactly once")
    return normalized


def tour_cost(distance_matrix: Any, tour: Any) -> float:
    """Calculate closed-tour cost using float64 accumulation."""
    matrix = validate_distance_matrix(distance_matrix)
    normalized = normalize_tour(tour, matrix.shape[0])
    indices = np.asarray(normalized, dtype=np.intp)
    successors = np.roll(indices, -1)
    with np.errstate(over="ignore", invalid="ignore"):
        cost = float(np.sum(matrix[indices, successors], dtype=np.float64))
    if not math_is_finite(cost):
        raise ValueError("tour cost must be finite")
    return cost


def math_is_finite(value: Any) -> bool:
    """Avoid accepting NumPy scalar truthiness as a finite numeric result."""
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(
        value, (bool, np.bool_)
    ) and bool(np.isfinite(value))


class TSPEoHMatrixEvaluation(Evaluation):
    """EoH evaluator for complete TSP tours over distance matrices."""

    supported_methods = ("EoH",)

    def __init__(self, timeout_seconds: int | float = 30, **kwargs):
        super().__init__(
            template_program=template_program,
            task_description=task_description,
            use_numba_accelerate=False,
            timeout_seconds=timeout_seconds,
            **kwargs,
        )
        self._datasets = []
        for seed in range(50):
            coords = np.random.default_rng(seed).random((100, 2))
            self._datasets.append(cdist(coords, coords, metric="euclidean"))

    @property
    def distance_matrices(self) -> tuple[np.ndarray, ...]:
        """Expose training matrices for inspection without allowing list replacement."""
        return tuple(self._datasets)

    def evaluate_program(self, program_str: str, callable_func: callable, **kwargs) -> float | None:
        del program_str, kwargs
        costs: list[float] = []
        try:
            for stored_matrix in self._datasets:
                matrix = validate_distance_matrix(stored_matrix)
                candidate_tour = callable_func(matrix.copy())
                normalized = normalize_tour(candidate_tour, matrix.shape[0])
                costs.append(tour_cost(matrix, normalized))
            fitness = -float(np.mean(np.asarray(costs, dtype=np.float64)))
            if not math_is_finite(fitness):
                return None
            return fitness
        except Exception:
            return None
