from __future__ import annotations

from typing import List


from .metrics import compute_aocc
from .runner import run_once_class, run_once_function
from .suite import RunConfig, get_problem, make_bbob_training_set


def parse_int_list(value) -> List[int] | None:
    """Normalize GUI/YAML list parameters into a list of ints.

    The GUI may pass YAML lists as strings (e.g. '[1, 2, 3]'). This helper
    converts such values into Python lists of integers. If the input is
    already a list or tuple, its elements are cast to int. If parsing fails,
    None is returned so the caller can fall back to defaults.
    """
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [int(x) for x in value]
    if isinstance(value, str):
        cleaned = value.strip()
        if cleaned.startswith("[") and cleaned.endswith("]"):
            cleaned = cleaned[1:-1]
        if not cleaned:
            return None
        return [int(x.strip()) for x in cleaned.split(",")]
    return None


__all__ = [
    "RunConfig",
    "compute_aocc",
    "get_problem",
    "make_bbob_training_set",
    "parse_int_list",
    "run_once_class",
    "run_once_function",
]
