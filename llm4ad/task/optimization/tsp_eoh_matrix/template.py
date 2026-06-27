template_program = '''import numpy as np


def solve(distance_matrix: np.ndarray) -> list[int]:
    """Return a permutation of all city indices representing a cyclic TSP tour."""
    n = distance_matrix.shape[0]
    return list(range(n))
'''


task_description = """
Design a complete deterministic algorithm for the symmetric traveling salesman
problem. Each candidate is evaluated on 50 training instances with 100 cities
per instance, and the complete evaluation must finish within a strict timeout.
The function receives only a finite square distance matrix and must always return
a Python list containing every city index exactly once. Return exactly n city indices;
do not append the starting city. The evaluator closes the cycle implicitly by
connecting the final city back to the first.

The implementation must use a bounded polynomial-time amount of work appropriate
for the stated instance size and evaluation budget. Factorial or exponential
search over city orderings or subsets is invalid because it cannot satisfy the
runtime requirement. No particular construction or improvement algorithm is
prescribed; choose the algorithmic approach without relying on examples from
this prompt.

NumPy is already available as np. Prefer NumPy and Python built-ins, never refer
to an unimported name, and keep all algorithm logic inside the solve function.
Do not depend on top-level helper functions, classes, external files, network
access, or random global state.

Minimize the total length of the closed tour. EoH maximizes a scalar fitness, so
the evaluator reports the negative mean tour length: shorter tours have larger
(less negative) fitness values.
""".strip()
