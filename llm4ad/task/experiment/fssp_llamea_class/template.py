"""LLaMEA template and baseline optimizer class for the FSSP."""

from __future__ import annotations


class FlowShopOptimizer:
    """Baseline complete optimizer for the permutation flow shop problem."""

    def __init__(self):
        pass

    def __call__(self, instance: dict) -> list[int]:
        """Return a 1-indexed job permutation for the given instance."""
        n = instance["n"]
        matrix = instance["matrix"]  # n x m, each row is one job across all machines

        # Sort jobs by total processing time descending; tie-break by original index.
        order = sorted(
            range(n),
            key=lambda j: (sum(matrix[j]), -j),
            reverse=True,
        )
        return [j + 1 for j in order]


# The string used as the LLaMEA seed / prompt example. It contains both the
# required class and a thin ``solve`` wrapper so that ``SecureEvaluator`` can
# execute the template (its machinery needs exactly one top-level function).
_baseline_class_code = '''
class FlowShopOptimizer:
    """Complete flow-shop optimizer returning a 1-indexed job permutation."""

    def __init__(self):
        pass

    def __call__(self, instance: dict) -> list[int]:
        """
        Solve one FSSP instance.

        The instance dictionary contains:
            instance_id: stable identifier
            n: number of jobs
            m: number of machines
            matrix: n x m list of lists; matrix[j][k] is the processing time
                    of job j on machine k
            family: instance family name
            subseries: subseries name (e.g. "tai20_5")

        Returns:
            A permutation of job indices using 1-based numbering. Every job
            from 1 to n must appear exactly once.
        """
        n = instance["n"]
        matrix = instance["matrix"]

        # Sort jobs by total processing time descending; tie-break by index.
        order = sorted(
            range(n),
            key=lambda j: (sum(matrix[j]), -j),
            reverse=True,
        )
        return [j + 1 for j in order]


def solve(instance: dict) -> list[int]:
    """Infrastructure wrapper used by SecureEvaluator; delegates to the class."""
    return FlowShopOptimizer()(instance)
'''


template_program = _baseline_class_code


task_description = (
    "Design a complete Python optimizer class for the permutation flow shop "
    "scheduling problem (FSSP). Your code MUST define a class named "
    "FlowShopOptimizer with a no-argument constructor and exactly this "
    "callable interface: __call__(self, instance). The instance dictionary "
    "provides instance_id, n, m, matrix (n x m list of lists where "
    "matrix[j][k] is the processing time of job j on machine k), family, "
    "and subseries. The class must construct and return a complete "
    "1-indexed permutation of all jobs as a list[int]; every job from 1 to n "
    "must appear exactly once. Implement the whole algorithm inside the class; "
    "do not return a priority score, partial schedule, or raw makespan value. "
    "The evaluator recomputes the makespan and compares it to a best-known "
    "value, then returns the negative macro-average relative gap. The "
    "training set contains 30 Taillard instances with 20 jobs each, so keep "
    "the optimizer deterministic and computationally bounded."
)
