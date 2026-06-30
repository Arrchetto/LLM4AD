template_program = '''import numpy as np


def solve(
    distance_matrix: np.ndarray,
    demands: np.ndarray,
    vehicle_capacity: int,
    max_vehicles: int,
) -> list[list[int]]:
    """Return complete CVRP routes within the 30-second evaluation.

    Every customer must occur once. Routes start and end at depot 0, respect
    vehicle_capacity, and use at most max_vehicles. The evaluator reports
    negative mean distance; EoH maximizes it. Keep all algorithm logic here.
    """
    routes = [[] for _ in range(max_vehicles)]
    loads = [0] * max_vehicles
    for customer in sorted(range(1, len(demands)), key=lambda i: (-int(demands[i]), i)):
        demand = int(demands[customer])
        feasible = [i for i in range(max_vehicles) if loads[i] + demand <= vehicle_capacity]
        if not feasible:
            return []
        vehicle = min(feasible, key=lambda i: (vehicle_capacity - loads[i] - demand, i))
        routes[vehicle].append(customer)
        loads[vehicle] += demand
    return [[0, *route, 0] for route in routes if route]
'''


task_description = (
    "Design a complete deterministic CVRP algorithm by implementing only solve"
    "(distance_matrix, demands, vehicle_capacity, max_vehicles). Node 0 is the depot. "
    "Return a two-dimensional Python list of routes; every used route must start and "
    "end at node 0, every customer 1..n-1 must appear exactly once, every route load "
    "must not exceed vehicle_capacity, and at most max_vehicles routes may be used. "
    "The evaluator performs strict validation without repair; invalid output, exceptions, "
    "or timeout invalidate the candidate. It evaluates 50 training instances: 27 CVRPLIB "
    "A instances and 23 CVRPLIB B instances, under a strict timeout of 30 seconds. Keep "
    "all algorithm logic inside solve; use NumPy as np and do not use top-level helpers, "
    "external files, network access, BKS values, or return fitness. EoH maximizes fitness, "
    "which is the negative mean route distance, so shorter routes are better."
)


__all__ = ["task_description", "template_program"]
