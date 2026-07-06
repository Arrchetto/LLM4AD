template_program = '''
import numpy as np


class BBOBOptimizer:
    """Complete optimizer for BBOB continuous black-box problems."""

    def __init__(self):
        pass

    def __call__(self, f, budget, dim):
        """
        Optimize f within a fixed evaluation budget.

        Args:
            f:      A callable f(x) where x is a 1D numpy array.
            budget: Maximum number of function evaluations.
            dim:    Dimensionality of the problem.

        Returns:
            history: A list or 1D array of best-so-far objective values.
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
    "Design a complete optimizer class named BBOBOptimizer for the BBOB noiseless "
    "benchmark suite. The class MUST have a no-argument constructor and a "
    "`__call__(self, f, budget, dim)` method. Within the budget, repeatedly evaluate "
    "f(x) for x in [-5, 5]^d and return a list or array of best-so-far objective values. "
    "The evaluator uses this history to compute the Area Over the Convergence Curve (AOCC)."
)
