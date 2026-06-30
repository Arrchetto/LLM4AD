from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable

import numpy as np

from llm4ad.base import Evaluation

from .cvrplib import CVRPInstance, load_cvrplib_sets, route_set_cost
from .template import task_description, template_program


class CVRPFEvaluation(Evaluation):
    """EoH evaluator for complete CVRP algorithms on CVRPLIB A+B."""

    supported_methods = ("EoH",)
    training_sets = ("A", "B")
    expected_training_counts = {"A": 27, "B": 23}

    def __init__(
        self,
        data_root: str | Path | None = None,
        timeout_seconds: int | float = 30,
        **kwargs: Any,
    ) -> None:
        kwargs.pop("random_seed", None)
        super().__init__(
            template_program=template_program,
            task_description=task_description,
            use_numba_accelerate=False,
            random_seed=0,
            timeout_seconds=timeout_seconds,
            **kwargs,
        )
        self.data_root = (
            Path(data_root).expanduser()
            if data_root is not None
            else Path(__file__).resolve().parents[5]
            / "data"
            / "benchmarks"
            / "cvrp"
            / "extracted"
        )
        self._training_instances = load_cvrplib_sets(
            self.data_root,
            self.training_sets,
            self.expected_training_counts,
        )

    @property
    def training_instances(self) -> tuple[CVRPInstance, ...]:
        return self._training_instances

    def evaluate_with_artifact(self, candidate: Callable[..., Any]) -> dict[str, Any]:
        instance_costs: list[dict[str, Any]] = []
        for instance in self._training_instances:
            routes = candidate(
                instance.distance_matrix.copy(),
                instance.demands.copy(),
                int(instance.capacity),
                int(instance.max_vehicles),
            )
            cost = route_set_cost(instance, routes)
            if not math.isfinite(cost):
                raise ValueError(f"{instance.name}: candidate cost must be finite")
            instance_costs.append(
                {"instance_name": instance.name, "candidate_cost": float(cost)}
            )
        costs = np.asarray(
            [row["candidate_cost"] for row in instance_costs], dtype=np.float64
        )
        mean_distance = float(np.mean(costs, dtype=np.float64))
        fitness = -mean_distance
        if not math.isfinite(mean_distance) or not math.isfinite(fitness):
            raise ValueError("mean distance and fitness must be finite")
        return {
            "fitness": fitness,
            "mean_distance": mean_distance,
            "instance_costs": instance_costs,
        }

    def evaluate_program(
        self, program_str: str, callable_func: Callable[..., Any], **kwargs: Any
    ) -> float | None:
        del program_str, kwargs
        try:
            return float(self.evaluate_with_artifact(callable_func)["fitness"])
        except Exception:
            return None

    def evaluate(self, candidate: Callable[..., Any]) -> float | None:
        return self.evaluate_program("", candidate)


__all__ = ["CVRPFEvaluation"]
