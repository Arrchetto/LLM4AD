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

    for t in range(budget):
        # Balance global exploration and local exploitation.
        if t % 20 == 0:
            # Global exploration: sample uniformly in the search space.
            candidate = np.random.uniform(-5.0, 5.0, size=dim)
        else:
            # Local search: small perturbation around the best known solution.
            candidate = best_x + np.random.normal(0.0, 0.5, size=dim)
            candidate = np.clip(candidate, -5.0, 5.0)

        value = f(candidate)
        if value < best_value:
            best_value = value
            best_x = candidate.copy()

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
    "\n\nCRITICAL IMPLEMENTATION RULES:"
    "\n1. The loop body must evaluate exactly ONE new point per iteration and call "
    "`history.append(best_value)` exactly ONCE per iteration, so that the returned "
    "history has length equal to `budget`."
    "\n2. Maintain TWO separate variables: `best_value` (scalar float, best objective "
    "found) and `best_x` (1D numpy array of length `dim`, the location where it was found). "
    "When doing local search, perturb `best_x`, not `best_value`."
    "\n3. Do NOT use a pure grid search: in 5 dimensions a grid quickly exceeds the "
    "budget and performs poorly. Prefer population-based or random-perturbation strategies."
    "\n4. Do NOT evaluate the same point repeatedly; every call to f(x) should use a "
    "fresh candidate solution."
    "\n5. A simple but effective strategy is: maintain a small population, mutate the "
    "best individual(s) with Gaussian noise of adaptive or fixed scale, evaluate the "
    "offspring, and keep the best solution found so far."
)
