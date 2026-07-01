"""EoH template and baseline solver for the FSSP complete-algorithm task."""

from __future__ import annotations


template_program = '''
def solve(instance_id: str, n: int, m: int, matrix: list[list[int]]) -> list[int]:
    """
    Solves the flow shop scheduling problem for one instance.

    Args:
        instance_id: stable instance identifier.
        n: number of jobs.
        m: number of machines.
        matrix: n x m list where matrix[j][k] is the processing time of job j
                on machine k.

    Returns:
        A permutation of the job indices using 1-based numbering. Every job
        from 1 to n must appear exactly once.

    Evaluation:
        The evaluator recomputes the makespan from this sequence and compares
        it to a best-known makespan. Minimize the makespan.
    """
    # Placeholder: return the identity permutation.
    return list(range(1, n + 1))
'''


task_description = (
    "Design a complete algorithm for the permutation flow shop scheduling problem. "
    "The solve function receives one full instance (instance_id, number of jobs n, "
    "number of machines m, and an n x m processing-time matrix) and must return a "
    "single permutation of the job indices using 1-based numbering. Every job from 1 "
    "to n must appear exactly once. The evaluator recomputes the makespan from the "
    "returned sequence and compares it to a best-known makespan. Minimize the makespan. "
    "Implement the whole solver inside solve, including construction and any improvement "
    "phases; do not return a priority score, partial rule, or raw makespan value. "
    "The training set contains 30 Taillard instances with 20 jobs each, so keep the "
    "algorithm deterministic and computationally bounded."
)


def _makespan(m: int, matrix: list[list[int]], sequence: list[int]) -> int:
    """Compute makespan for a 1-indexed job sequence of any length."""
    k = len(sequence)
    completion = [[0] * m for _ in range(k)]
    for i in range(k):
        job = sequence[i] - 1
        for j in range(m):
            proc = matrix[job][j]
            if i == 0 and j == 0:
                completion[i][j] = proc
            elif i == 0:
                completion[i][j] = completion[i][j - 1] + proc
            elif j == 0:
                completion[i][j] = completion[i - 1][j] + proc
            else:
                completion[i][j] = max(completion[i - 1][j], completion[i][j - 1]) + proc
    return completion[-1][-1] if k else 0


def solve(instance_id: str, n: int, m: int, matrix: list[list[int]]) -> list[int]:
    """Baseline NEH-style heuristic used by tests and documentation."""
    # Total processing time per job, descending.
    order = sorted(range(n), key=lambda j: sum(matrix[j]), reverse=True)
    sequence: list[int] = [order[0] + 1]
    for job in order[1:]:
        best_pos = 0
        best_makespan = float("inf")
        for pos in range(len(sequence) + 1):
            candidate = sequence[:pos] + [job + 1] + sequence[pos:]
            ms = _makespan(m, matrix, candidate)
            if ms < best_makespan:
                best_makespan = ms
                best_pos = pos
        sequence.insert(best_pos, job + 1)
    return sequence
