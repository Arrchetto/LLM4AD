from __future__ import annotations

import numpy as np


def compute_aocc(
    history: np.ndarray,
    f_opt: float,
    budget: int,
    lb: float = 1e-8,
    ub: float = 1e2,
) -> float:
    """Compute normalized Area Over the Convergence Curve.

    Args:
        history: Best-so-far objective values, length <= budget.
        f_opt: Known optimum of the function.
        budget: Fixed evaluation budget. The history is padded/truncated to
            this length before computing AOCC.
        lb: Lower bound of the precision range of interest.
        ub: Upper bound of the precision range of interest.

    Returns:
        Normalized AOCC in [0, 1]. Higher is better.
    """
    history = np.asarray(history, dtype=float)
    if history.ndim != 1:
        raise ValueError("history must be a 1D array")

    if len(history) < budget:
        history = np.pad(
            history,
            (0, budget - len(history)),
            mode="edge",
        )
    elif len(history) > budget:
        history = history[:budget]

    precision = np.maximum(history - f_opt, lb)
    log_precision = np.log10(precision)
    log_lb = np.log10(lb)
    log_ub = np.log10(ub)
    clipped = np.clip(log_precision, log_lb, log_ub)
    normalized = 1.0 - (clipped - log_lb) / (log_ub - log_lb)
    return float(np.mean(normalized))
