"""Shared FSSP feasibility checking, makespan computation, and gap aggregation."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from numbers import Integral
from typing import Any

from .dataset import FSSPInstance


def _normalize_sequence(value: Any) -> list[int] | None:
    """Convert a candidate job sequence into a list of ints, rejecting invalid types."""
    if isinstance(value, str) or isinstance(value, bytes) or isinstance(value, bytearray):
        return None
    if isinstance(value, Sequence):
        values = list(value)
    else:
        return None

    normalized: list[int] = []
    for item in values:
        if isinstance(item, bool) or not isinstance(item, Integral):
            return None
        normalized.append(int(item))
    return normalized


def validate_sequence(instance: FSSPInstance, sequence: Any) -> tuple[int, ...]:
    """Validate that ``sequence`` is a 1-indexed permutation of all jobs.

    Returns the normalized sequence as a tuple of ints.
    Raises ``ValueError`` for any malformed or infeasible sequence.
    """
    normalized = _normalize_sequence(sequence)
    if normalized is None:
        raise ValueError("job sequence must be a sequence of integers")
    if len(normalized) != instance.n:
        raise ValueError(
            f"job sequence length {len(normalized)} does not match n={instance.n}"
        )

    seen = [False] * instance.n
    for job in normalized:
        if job < 1 or job > instance.n:
            raise ValueError(f"job index {job} is out of range [1, {instance.n}]")
        if seen[job - 1]:
            raise ValueError(f"job {job} appears more than once")
        seen[job - 1] = True

    if not all(seen):
        raise ValueError("not every job appears in the sequence")

    return tuple(normalized)


def compute_makespan(instance: FSSPInstance, sequence: Sequence[int]) -> int:
    """Compute the makespan for ``sequence`` using the classical flow-shop recurrence.

    ``sequence`` must already be validated and use 1-based job indices.
    """
    n = instance.n
    m = instance.m
    matrix = instance.matrix

    # completion[i][j] = completion time of i-th job in sequence on machine j.
    completion = [[0] * m for _ in range(n)]
    for i in range(n):
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
    return completion[-1][-1]


def relative_gap(candidate_value: float, reference_value: float) -> float:
    """Return (candidate - reference) / reference, requiring finite positive reference."""
    if reference_value <= 0:
        raise ValueError("reference value must be positive")
    if candidate_value < 0:
        raise ValueError("candidate value must be non-negative")
    gap = (candidate_value - reference_value) / reference_value
    if not math.isfinite(gap):
        raise ValueError("relative gap must be finite")
    return float(gap)


def macro_average_fitness(series_gaps: Sequence[tuple[str, float]]) -> float:
    """Average gaps within each subseries, then average subseries means equally.

    Returns the negative macro-average because EoH maximizes fitness.
    """
    grouped: dict[str, list[float]] = defaultdict(list)
    for series, gap in series_gaps:
        if not series or not math.isfinite(gap):
            raise ValueError("series and finite gap are required")
        grouped[series].append(float(gap))
    if not grouped:
        raise ValueError("at least one gap is required")
    series_means = [sum(values) / len(values) for values in grouped.values()]
    return -float(sum(series_means) / len(series_means))
