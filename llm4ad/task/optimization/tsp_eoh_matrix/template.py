template_program = '''import numpy as np


def solve(distance_matrix: np.ndarray) -> list[int]:
    """Return a permutation of all city indices representing a cyclic TSP tour."""
    n = distance_matrix.shape[0]
    return list(range(n))
'''


task_description = """
Design a complete deterministic algorithm for the symmetric traveling salesman
problem. The function receives only a finite square distance matrix and must
return a Python list containing every city index exactly once. The returned
permutation represents a cyclic tour, so its last city is connected back to its
first city. Minimize the total length of that closed tour. EoH maximizes a scalar
fitness, therefore the evaluator reports the negative mean tour length: shorter
tours have larger (less negative) fitness values. Keep all algorithm logic inside
the solve function; do not depend on top-level helper functions, classes, files,
network access, or random global state.
""".strip()

