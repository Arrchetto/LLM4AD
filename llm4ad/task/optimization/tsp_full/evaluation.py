from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from llm4ad.base import Evaluation

from .get_instance import GetData
from .template import task_description, template_program

__all__ = [
    "TSPFullLLaMEAEvaluation",
    "calculate_tour_length",
    "normalize_tour",
    "validate_distance_matrix",
]


def validate_distance_matrix(distance_matrix: Any) -> np.ndarray:
    """Return a finite, non-empty square distance matrix as float64."""
    try:
        matrix = np.asarray(distance_matrix, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("distance matrix must contain numeric values") from error
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(
            "distance matrix must be a non-empty two-dimensional square matrix"
        )
    if not np.isfinite(matrix).all():
        raise ValueError("distance matrix must contain only finite values")
    return matrix


def normalize_tour(tour: Any, city_count: int) -> list[int]:
    """Validate and normalize a candidate permutation.

    The tour must be a one-dimensional sequence of exactly ``city_count``
    integer city indices in ``[0, city_count)`` with no duplicates.
    """
    if isinstance(tour, np.ndarray):
        if tour.ndim != 1:
            raise ValueError("tour NumPy array must be one-dimensional")
        values: Sequence[Any] = tour.tolist()
    elif isinstance(tour, list):
        values = tour
    else:
        raise ValueError("tour must be a Python list or one-dimensional NumPy array")

    if len(values) != city_count:
        raise ValueError(
            f"tour length must be {city_count}, got {len(values)}"
        )

    normalized: list[int] = []
    for position, city in enumerate(values):
        if isinstance(city, (bool, np.bool_)) or not isinstance(
            city, (int, np.integer)
        ):
            raise ValueError(
                f"tour city at position {position} must be an integer"
            )
        city_index = int(city)
        if city_index < 0 or city_index >= city_count:
            raise ValueError(
                f"tour city {city_index} is outside [0, {city_count})"
            )
        normalized.append(city_index)

    if len(set(normalized)) != city_count:
        raise ValueError("tour must contain every city exactly once")
    return normalized


def calculate_tour_length(distance_matrix: Any, tour: Any) -> float:
    """Calculate the closed tour length using float64 accumulation.

    The returned cost includes the edge from the last city back to the first.
    """
    matrix = validate_distance_matrix(distance_matrix)
    normalized = normalize_tour(tour, matrix.shape[0])
    indices = np.asarray(normalized, dtype=np.intp)
    successors = np.roll(indices, -1)
    cost = float(np.sum(matrix[indices, successors], dtype=np.float64))
    if not _is_finite_scalar(cost):
        raise ValueError("tour cost must be a finite scalar")
    return cost


def _is_finite_scalar(value: Any) -> bool:
    """Return True if value is a finite real scalar (excluding bool)."""
    return (
        isinstance(value, (int, float, np.integer, np.floating))
        and not isinstance(value, (bool, np.bool_))
        and bool(np.isfinite(value))
    )


class TSPFullLLaMEAEvaluation(Evaluation):
    """LLaMEA evaluator for complete TSP optimizer classes.

    The candidate must define a top-level ``TSPOptimizer`` class with a
    no-argument constructor and ``__call__(self, distance_matrix)`` that
    returns a permutation of city indices. Fitness is the negative mean tour
    length, so shorter tours yield higher (less negative) fitness values.
    """

    candidate_type = "class"
    candidate_name = "TSPOptimizer"
    candidate_call_signature = ("distance_matrix",)
    supported_methods = ("LLaMEA",)

    def __init__(
        self,
        timeout_seconds: int | float = 60,
        n_instance: int = 50,
        problem_size: int = 100,
        seeds: list[int] | tuple[int, ...] | range | None = None,
        **kwargs: Any,
    ):
        super().__init__(
            template_program=template_program,
            task_description=task_description,
            use_numba_accelerate=False,
            timeout_seconds=timeout_seconds,
            **kwargs,
        )
        self.n_instance = int(n_instance)
        self.problem_size = int(problem_size)
        self._datasets = GetData(
            n_instance=self.n_instance,
            problem_size=self.problem_size,
            seeds=seeds,
        ).generate_instances()

    def evaluate_program(
        self,
        program_str: str,
        callable_func: callable,
        **kwargs: Any,
    ) -> float | None:
        return self.evaluate(callable_func)

    def evaluate(self, optimizer_class: type) -> float | None:
        """Evaluate an optimizer class on all training instances.

        Returns the negative mean tour length, or ``None`` if any instance
        produces an invalid tour or raises an exception.
        """
        costs: list[float] = []
        for _, distance_matrix in self._datasets:
            try:
                optimizer = optimizer_class()
                route = optimizer(distance_matrix.copy())
                cost = calculate_tour_length(distance_matrix, route)
            except Exception:
                return None
            costs.append(cost)

        if not costs:
            return None

        fitness = -float(np.mean(np.asarray(costs, dtype=np.float64)))
        if not _is_finite_scalar(fitness):
            return None
        return fitness
