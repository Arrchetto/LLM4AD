template_program = '''import numpy as np


def solve(
    distance_matrix: np.ndarray,
    demands: np.ndarray,
    vehicle_capacity: int,
    max_vehicles: int,
) -> list[int]:
    """Return a customer permutation within the 30-second evaluation.

    Include every customer exactly once and exclude depot 0. The evaluator's
    capacity decoder enforces vehicle_capacity and max_vehicles. EoH maximizes
    negative mean distance, so shorter decoded routes are better.
    """
    remaining = set(range(1, len(demands)))
    permutation = []
    current = 0
    while remaining:
        customer = min(
            remaining,
            key=lambda node: (float(distance_matrix[current, node]), node),
        )
        permutation.append(customer)
        remaining.remove(customer)
        current = customer
    return permutation
'''


task_description = (
    "Design a deterministic CVRP customer-priority algorithm by implementing only solve"
    "(distance_matrix, demands, vehicle_capacity, max_vehicles). Node 0 is the depot. "
    "Return a one-dimensional Python customer permutation containing every customer "
    "1..n-1 exactly once; never include node 0. The evaluator strictly rejects malformed "
    "permutations, then uses a deterministic capacity decoder to enforce vehicle_capacity "
    "and max_vehicles and to produce complete routes. It evaluates 50 training instances: "
    "27 CVRPLIB A instances and 23 CVRPLIB B instances, under a strict timeout of 30 seconds. "
    "Any invalid output, exception, decoder failure, non-finite value, or timeout invalidates "
    "the whole candidate. Use bounded polynomial-time work. Keep all algorithm logic inside "
    "solve; NumPy is available as np. Do not depend on top-level helpers, external files, "
    "network access, random global state, or BKS values, and do not return fitness. EoH "
    "maximizes fitness, which is the negative mean decoded route distance, so shorter is better."
)


__all__ = ["task_description", "template_program"]
