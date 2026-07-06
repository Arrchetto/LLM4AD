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
    best = float("inf")
    history = []
    for _ in range(budget):
        x = np.random.uniform(-5.0, 5.0, size=dim)
        y = f(x)
        if y < best:
            best = y
        history.append(best)
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
)
