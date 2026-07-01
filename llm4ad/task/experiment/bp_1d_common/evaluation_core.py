from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import math
from numbers import Integral
from typing import Any

from .dataset import BPInstance


def validate_packing(instance: BPInstance, solution: Any) -> tuple[tuple[int, ...], ...]:
    if not isinstance(solution, Mapping):
        raise ValueError("solution must be a mapping")
    num_bins = solution.get("num_bins")
    bins = solution.get("bins")
    if isinstance(num_bins, bool) or not isinstance(num_bins, Integral):
        raise ValueError("num_bins must be an integer")
    if not isinstance(bins, Sequence) or isinstance(bins, (str, bytes, bytearray)):
        raise ValueError("bins must be a sequence")
    if int(num_bins) != len(bins) or int(num_bins) <= 0:
        raise ValueError("num_bins must equal the positive number of bins")

    normalized: list[tuple[int, ...]] = []
    counts = [0] * (instance.num_items + 1)
    for bin_items in bins:
        if not isinstance(bin_items, Sequence) or isinstance(
            bin_items, (str, bytes, bytearray)
        ):
            raise ValueError("each bin must be a sequence")
        if not bin_items:
            raise ValueError("empty bins are invalid")
        current: list[int] = []
        load = 0
        for item_index in bin_items:
            if isinstance(item_index, bool) or not isinstance(item_index, Integral):
                raise ValueError("item indices must be integers")
            index = int(item_index)
            if index < 1 or index > instance.num_items:
                raise ValueError("item index is out of range")
            counts[index] += 1
            load += instance.items[index - 1]
            current.append(index)
        if load > instance.bin_capacity:
            raise ValueError("bin capacity exceeded")
        normalized.append(tuple(current))
    if any(count != 1 for count in counts[1:]):
        raise ValueError("every item must appear exactly once")
    return tuple(normalized)


def relative_gap(num_bins: int, reference_bins: int) -> float:
    if reference_bins <= 0 or num_bins <= 0:
        raise ValueError("bin counts must be positive")
    gap = (num_bins - reference_bins) / reference_bins
    if not math.isfinite(gap):
        raise ValueError("relative gap must be finite")
    return float(gap)


def macro_average_fitness(series_gaps: Sequence[tuple[str, float]]) -> float:
    grouped: dict[str, list[float]] = defaultdict(list)
    for series, gap in series_gaps:
        if not series or not math.isfinite(gap):
            raise ValueError("series and finite gap are required")
        grouped[series].append(float(gap))
    if not grouped:
        raise ValueError("at least one gap is required")
    series_means = [sum(values) / len(values) for values in grouped.values()]
    return -float(sum(series_means) / len(series_means))

