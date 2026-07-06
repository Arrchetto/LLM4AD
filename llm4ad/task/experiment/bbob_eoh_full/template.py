template_program = '''
import numpy as np


def optimize(f, budget, dim):
    """
    Optimize a black-box function f within a fixed evaluation budget.

    Args:
        f:      A callable f(x) where x is a 1D numpy array of length dim.
        budget: Maximum number of times f may be called.
        dim:    Dimensionality of the search space.

    Returns:
        history: A list or 1D array of length <= budget, where history[t]
                 is the best function value seen after evaluation t+1.
    """
    best_value = float("inf")
    best_x = np.random.uniform(-5.0, 5.0, size=dim)
    history = []
    for _ in range(budget):
        x = np.random.uniform(-5.0, 5.0, size=dim)
        y = f(x)
        if y < best_value:
            best_value = y
            best_x = x.copy()
        history.append(best_value)
    return history
'''


task_description = (
    "Design a complete black-box optimization algorithm for the BBOB noiseless "
    "benchmark suite. The code MUST define a single function `optimize(f, budget, dim)` "
    "that takes a callable objective f, a maximum number of function evaluations "
    "(budget), and the problem dimension (dim). Within the budget, repeatedly evaluate "
    "f(x) for x in [-5, 5]^d and return a list or array of best-so-far objective values "
    "after each evaluation. The evaluator uses this history to compute the Area Over "
    "the Convergence Curve (AOCC)."
    "\n\nIMPORTANT: maintain TWO separate variables: "
    "(1) `best_value` / `best_f` as a scalar float holding the best objective value found so far, and "
    "(2) `best_x` as a 1D numpy array of length `dim` holding the solution location where that best value was found. "
    "If you implement local search or adaptive sampling around the current best, "
    "perturb `best_x` (the position vector), NOT `best_value` (the scalar). "
    "For example: `candidate = best_x + np.random.normal(0, 0.1, dim)` followed by clipping to [-5, 5]."
)
