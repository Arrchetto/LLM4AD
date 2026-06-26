template_program = '''
import numpy as np
def select_next_node(current_node: int, destination_node: int, unvisited_nodes: np.ndarray, distance_matrix: np.ndarray, prizes: np.ndarray, remaining_budget: float) -> int:
    """
    Design a novel constructive heuristic for the Orienteering Problem.

    Args:
    current_node: ID of the current node.
    destination_node: ID of the route destination node.
    unvisited_nodes: Array of feasible unvisited node IDs. Visiting one of these nodes still leaves enough budget to return to the destination.
    distance_matrix: Pairwise Euclidean distance matrix of all nodes.
    prizes: Prize values of all nodes. The depot prize is 0.
    remaining_budget: Remaining travel budget before selecting the next node.

    Return:
    ID of the next node to visit.
    """
    next_node = unvisited_nodes[0]

    return next_node

# IMPORTANT CONSTRAINTS:
# 1. You MUST keep the exact function signature shown above:
#    def select_next_node(current_node: int, destination_node: int, unvisited_nodes: np.ndarray,
#                         distance_matrix: np.ndarray, prizes: np.ndarray, remaining_budget: float) -> int
# 2. Do NOT add, remove, or rename any parameters.
# 3. Do NOT use default argument values.
# 4. The evaluator will call this function with exactly these 6 arguments.
#    Parameters such as cumulative_prize, cumulative_distance, or depot_node are NOT available.
#    Changing the signature will cause the code to fail evaluation.
'''

task_description = ("Given a depot and a set of optional customer nodes with coordinates and prizes, "
                    "the Orienteering Problem asks for a route that starts and ends at the depot, "
                    "keeps total travel distance within a fixed budget, and maximizes collected prize. "
                    "The task can be solved step-by-step by repeatedly selecting one feasible unvisited node. "
                    "Help me design a novel algorithm to select the next node in each step. "
                    "CRITICAL: The select_next_node function MUST keep exactly this signature: "
                    "select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix, prizes, remaining_budget). "
                    "Do NOT add, remove, or rename any parameters. Do NOT use default arguments.")

