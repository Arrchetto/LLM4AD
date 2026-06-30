template_program = '''import numpy as np


def solve(
    distance_matrix: np.ndarray,
    demands: np.ndarray,
    vehicle_capacity: int,
    max_vehicles: int,
) -> list[list[int]]:
    """Return complete deterministic CVRP routes within the 30-second evaluation.

    Node 0 is the depot. Every customer 1..n-1 must occur exactly once. Each
    used route must start and end at depot 0, respect vehicle_capacity, and the
    number of used routes must not exceed max_vehicles. Return routes only.
    The evaluator computes negative mean route distance; EoH maximizes it, so
    shorter routes receive larger fitness. Keep all algorithm logic here.
    """
    customer_order = sorted(
        range(1, len(demands)),
        key=lambda node: (-int(demands[node]), node),
    )
    assigned: list[list[int]] = []
    loads: list[int] = []
    for customer in customer_order:
        demand = int(demands[customer])
        feasible_bins = [
            index
            for index, load in enumerate(loads)
            if load + demand <= vehicle_capacity
        ]
        if feasible_bins:
            bin_index = min(
                feasible_bins,
                key=lambda index: (
                    vehicle_capacity - loads[index] - demand,
                    index,
                ),
            )
            assigned[bin_index].append(customer)
            loads[bin_index] += demand
        elif len(assigned) < max_vehicles and demand <= vehicle_capacity:
            assigned.append([customer])
            loads.append(demand)
        else:
            return []

    routes: list[list[int]] = []
    for customers in assigned:
        route = [0]
        remaining = set(customers)
        while remaining:
            current = route[-1]
            next_customer = min(
                remaining,
                key=lambda node: (float(distance_matrix[current, node]), node),
            )
            route.append(int(next_customer))
            remaining.remove(next_customer)
        route.append(0)
        routes.append(route)
    return routes
'''


task_description = """
Design a complete deterministic algorithm for the capacitated vehicle routing
problem (CVRP). The evaluator uses exactly 50 training instances from CVRPLIB:
27 fixed instances from set A and 23 fixed instances from set B. The complete
evaluation runs under a strict timeout of 30 seconds, so use bounded computation
suitable for all 50 instances.

Implement only the provided solve function. Its inputs are distance_matrix,
demands, vehicle_capacity, and max_vehicles. Node 0 is the depot; customer nodes
are 1 through n-1. Return a two-dimensional Python list of complete routes, for
example [[0, 4, 1, 0], [0, 3, 2, 5, 0]]. Every used route must start and end at
node 0, every customer must appear exactly once across all routes, every route
load must be at most vehicle_capacity, and the number of used routes must be at
most max_vehicles. Return routes only, never fitness or metadata.

The evaluator applies strict validation and performs no repair: it does not add
depots, remove invalid or duplicate nodes, insert missing customers, split routes,
or replace empty output. Any exception, timeout, invalid structure, infeasible
solution, or non-finite result invalidates the complete candidate.

NumPy is available as np. Keep all algorithm logic inside solve. Do not add or
depend on top-level helper functions or classes, external files, network access,
random global state, BKS values, or evaluator internals.

Minimize mean total route distance. EoH maximizes fitness, and the evaluator uses
the negative mean distance over all 50 instances. Shorter solutions therefore
have larger (less negative) fitness values.
""".strip()


__all__ = ["task_description", "template_program"]
