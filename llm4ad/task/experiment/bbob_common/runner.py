from __future__ import annotations

import numpy as np

from .suite import RunConfig, get_problem


def _validate_history(history, budget: int) -> np.ndarray:
    """Normalize candidate history to a 1D finite array of length budget."""
    if history is None:
        raise ValueError("candidate returned None")
    history = np.asarray(history, dtype=float)
    if history.ndim != 1:
        raise ValueError("candidate history must be a 1D array")
    if len(history) == 0:
        raise ValueError("candidate history is empty")
    if len(history) < budget:
        history = np.pad(history, (0, budget - len(history)), mode="edge")
    elif len(history) > budget:
        history = history[:budget]
    if not np.all(np.isfinite(history)):
        raise ValueError("candidate history contains non-finite values")
    return history


def run_once_function(optimize_func: callable, config: RunConfig) -> np.ndarray:
    """Run a function-style candidate on one BBOB config and return history."""
    problem = get_problem(config)
    problem.reset()
    np.random.seed(config.seed)
    history = optimize_func(problem, config.budget, config.dim)
    return _validate_history(history, config.budget)


def run_once_class(optimizer_class: type, config: RunConfig) -> np.ndarray:
    """Run a class-style candidate on one BBOB config and return history."""
    problem = get_problem(config)
    problem.reset()
    np.random.seed(config.seed)
    optimizer = optimizer_class()
    history = optimizer(problem, config.budget, config.dim)
    return _validate_history(history, config.budget)
