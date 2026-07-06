from __future__ import annotations

from typing import Any

import numpy as np

from llm4ad.base import Evaluation
from llm4ad.task.experiment.bbob_common import (
    compute_aocc,
    get_problem,
    make_bbob_training_set,
    parse_int_list,
    run_once_class,
)

from .template import task_description, template_program

__all__ = ["BBOBLLaMEAClassEvaluation"]


class BBOBLLaMEAClassEvaluation(Evaluation):
    """Evaluate complete LLaMEA optimizer classes on the BBOB 5D training suite."""

    candidate_type = "class"
    candidate_name = "BBOBOptimizer"
    candidate_call_signature = ("f", "budget", "dim")
    supported_methods = ("LLaMEA",)

    def __init__(
        self,
        timeout_seconds: int | float = 60,
        dim: int = 5,
        instances: list[int] | None = None,
        seeds: list[int] | None = None,
        budget: int = 10_000,
        lb: float = 1e-8,
        ub: float = 1e2,
        **kwargs: Any,
    ):
        super().__init__(
            template_program=template_program,
            task_description=task_description,
            use_numba_accelerate=False,
            timeout_seconds=timeout_seconds,
            **kwargs,
        )
        self.dim = int(dim)
        self.instances = parse_int_list(instances) or [1, 2, 3]
        self.seeds = parse_int_list(seeds) or [0, 1, 2]
        self.budget = int(budget)
        self.lb = float(lb)
        self.ub = float(ub)
        self.configs = make_bbob_training_set(
            dim=self.dim,
            instances=self.instances,
            seeds=self.seeds,
            budget=self.budget,
        )

    def evaluate_program(
        self,
        program_str: str,
        callable_func: callable,
        **kwargs: Any,
    ) -> float | None:
        return self.evaluate(callable_func)

    def evaluate(self, optimizer_class: type) -> float | None:
        aoccs = []
        for config in self.configs:
            try:
                history = run_once_class(optimizer_class, config)
                problem = get_problem(config)
                aocc = compute_aocc(
                    history,
                    problem.optimum.y,
                    config.budget,
                    self.lb,
                    self.ub,
                )
            except Exception:
                return None
            if not np.isfinite(aocc):
                return None
            aoccs.append(aocc)
        return float(np.mean(aoccs))
