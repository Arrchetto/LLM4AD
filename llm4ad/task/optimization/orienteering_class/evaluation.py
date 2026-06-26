from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from llm4ad.base import Evaluation
from llm4ad.task.optimization.orienteering_construct.get_instance import (
    GetData,
)
from llm4ad.task.optimization.orienteering_class.template import (
    task_description,
    template_program,
)

__all__ = ["OrienteeringClassEvaluation"]


class OrienteeringClassEvaluation(Evaluation):
    """Evaluate complete optimizer classes for the Orienteering Problem."""

    candidate_type = "class"
    candidate_name = "OrienteeringOptimizer"
    candidate_call_signature = ("instance",)
    supported_methods = ("LLaMEA",)

    def __init__(
            self,
            timeout_seconds=30,
            n_instance=16,
            problem_size=50,
            max_length_ratio=0.35,
            seed=2024,
            **kwargs):
        super().__init__(
            template_program=template_program,
            task_description=task_description,
            use_numba_accelerate=False,
            timeout_seconds=timeout_seconds,
        )
        self.n_instance = int(n_instance)
        self.problem_size = int(problem_size)
        self.max_length_ratio = float(max_length_ratio)
        self.seed = int(seed)
        self._datasets = GetData(
            n_instance=self.n_instance,
            problem_size=self.problem_size,
            max_length_ratio=self.max_length_ratio,
            seed=self.seed,
        ).generate_instances()

    def evaluate_program(
            self,
            program_str: str,
            callable_func: callable,
    ) -> Any | None:
        return self.evaluate(callable_func)

    @staticmethod
    def _normalize_route(route) -> list[int] | None:
        if isinstance(route, np.ndarray):
            if route.ndim != 1:
                return None
            values = route.tolist()
        elif isinstance(route, Sequence) and not isinstance(
                route,
                (str, bytes, bytearray),
        ):
            values = list(route)
            if any(
                    isinstance(value, Sequence)
                    and not isinstance(value, (str, bytes, bytearray))
                    for value in values):
                return None
        else:
            return None

        normalized = []
        for value in values:
            if isinstance(value, (bool, np.bool_)):
                return None
            if not isinstance(value, (int, np.integer)):
                return None
            normalized.append(int(value))
        return normalized

    def _score_route(
            self,
            instance: dict,
            route,
    ) -> tuple[float, float] | None:
        route = self._normalize_route(route)
        if not route:
            return None

        start_node = int(instance["start_node"])
        end_node = int(instance["end_node"])
        if route[0] != start_node or route[-1] != end_node:
            return None

        node_count = len(instance["prizes"])
        if any(node < 0 or node >= node_count for node in route):
            return None

        customer_nodes = [
            node for node in route
            if node not in (start_node, end_node)
        ]
        if len(customer_nodes) != len(set(customer_nodes)):
            return None

        distance_matrix = instance["distance_matrix"]
        travel_length = sum(
            float(distance_matrix[left][right])
            for left, right in zip(route, route[1:])
        )
        if travel_length > float(instance["max_length"]) + 1e-8:
            return None

        collected_prize = float(
            np.sum(instance["prizes"][customer_nodes])
        ) if customer_nodes else 0.0
        return collected_prize, travel_length

    def evaluate(self, optimizer_class: type) -> float | None:
        collected_prizes = []
        for instance in self._datasets:
            try:
                optimizer = optimizer_class()
                route = optimizer(instance)
            except Exception:
                return None

            result = self._score_route(instance, route)
            if result is None:
                return None
            collected_prize, _ = result
            collected_prizes.append(collected_prize)

        if not collected_prizes:
            return None
        return float(np.mean(collected_prizes))
