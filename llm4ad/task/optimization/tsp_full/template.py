template_program = '''
import numpy as np


class TSPOptimizer:
    """Construct a complete TSP tour from a distance matrix."""

    def __init__(self):
        pass

    def __call__(self, distance_matrix: np.ndarray) -> list[int]:
        """Return a tour that visits every city exactly once.

        Args:
            distance_matrix: A finite, non-empty n x n symmetric distance
                matrix where entry (i, j) is the distance from city i to city j.

        Returns:
            A list of exactly n integer city indices in [0, n), each appearing
            once. The evaluator appends the closing edge from the last city back
            to the first, so do not duplicate the starting city at the end.
        """
        n = distance_matrix.shape[0]
        return list(range(n))
'''


task_description = (
    "Design a complete Python optimizer class for the Traveling Salesman Problem. "
    "The code MUST define a top-level class named TSPOptimizer with a no-argument "
    "constructor and exactly this callable interface: "
    "__call__(self, distance_matrix). The distance_matrix input is a finite, "
    "non-empty n x n symmetric NumPy array. The class must construct and return "
    "a complete tour as a Python list (or one-dimensional NumPy array) of integer "
    "city indices. The tour must contain every city in [0, n) exactly once and "
    "must NOT repeat the starting city at the end. The evaluator computes the "
    "closed tour length (including the return edge) and returns fitness as the "
    "negative mean length, so shorter tours are better. "
    "CRITICAL runtime constraint: the optimizer is evaluated on 50 instances with "
    "100 cities each. The total evaluation budget for one candidate is limited, "
    "so the algorithm must run in low-order polynomial time. Do NOT use factorial, "
    "exponential, or high-order polynomial search (e.g. repeated full 2-opt scans "
    "inside nested loops, or many random restarts). Keep nested loops shallow and "
    "avoid Python-level per-city list comprehensions inside inner loops. "
    "Do not return fitness; the evaluator calculates it. Do not write a main block or test code."
)
