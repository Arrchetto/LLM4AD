template_program = '''
import numpy as np


class OrienteeringOptimizer:
    """Construct a complete feasible route for one Orienteering instance."""

    def __init__(self):
        pass

    def __call__(self, instance: dict) -> list[int]:
        """
        Build and return a complete route.

        The instance dictionary contains coordinates, distance_matrix, prizes,
        start_node, end_node, and max_length. The route must start and end at
        the depot, must not repeat customer nodes, and must respect max_length.
        It is valid to visit only a profitable subset of customer nodes.
        """
        distance_matrix = instance["distance_matrix"]
        prizes = instance["prizes"]
        depot = int(instance["start_node"])
        remaining_budget = float(instance["max_length"])
        current_node = depot
        unvisited_nodes = np.arange(1, len(prizes), dtype=int)
        route = [depot]

        while len(unvisited_nodes) > 0:
            travel_cost = distance_matrix[current_node][unvisited_nodes]
            return_cost = distance_matrix[unvisited_nodes, depot]
            feasible_nodes = unvisited_nodes[
                travel_cost + return_cost <= remaining_budget + 1e-12
            ]
            if len(feasible_nodes) == 0:
                break

            scores = prizes[feasible_nodes] / (
                distance_matrix[current_node][feasible_nodes]
                + distance_matrix[feasible_nodes, depot]
                + 1e-12
            )
            next_node = int(feasible_nodes[np.argmax(scores)])
            remaining_budget -= float(
                distance_matrix[current_node][next_node]
            )
            route.append(next_node)
            current_node = next_node
            unvisited_nodes = unvisited_nodes[unvisited_nodes != next_node]

        route.append(depot)
        return route
'''


task_description = (
    "Design a complete Python optimizer class for the Orienteering Problem. "
    "The code MUST define class OrienteeringOptimizer with a no-argument "
    "constructor and exactly this callable interface: "
    "__call__(self, instance). The instance dictionary provides coordinates, "
    "distance_matrix, prizes, start_node, end_node, and max_length. The class "
    "must construct and return the complete route as a list of integer node "
    "IDs. The route must start and end at the depot, may visit a profitable "
    "subset of customers, must not repeat customers, and must respect the "
    "travel budget. Do not return fitness; the evaluator calculates it."
)

