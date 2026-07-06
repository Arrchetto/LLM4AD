from __future__ import annotations

from dataclasses import dataclass
from typing import List

import ioh


@dataclass(frozen=True)
class RunConfig:
    """Configuration for a single BBOB evaluation run."""

    problem_id: int
    instance: int
    seed: int
    dim: int
    budget: int


def make_bbob_training_set(
    dim: int = 5,
    instances: List[int] | None = None,
    seeds: List[int] | None = None,
    budget: int = 10_000,
) -> List[RunConfig]:
    """Build the default BBOB training protocol.

    Defaults match the LLaMEA paper: 24 functions, 3 instances, 3 seeds,
    dimension 5, budget 10_000.
    """
    instances = instances or [1, 2, 3]
    seeds = seeds or [0, 1, 2]
    return [
        RunConfig(problem_id, instance, seed, dim, budget)
        for problem_id in range(1, 25)
        for instance in instances
        for seed in seeds
    ]


def get_problem(config: RunConfig):
    """Instantiate an IOH BBOB problem for the given run config."""
    return ioh.get_problem(
        config.problem_id,
        instance=config.instance,
        dimension=config.dim,
    )
