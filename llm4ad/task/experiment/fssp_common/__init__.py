"""Shared FSSP data, validation, and scoring core for EoH and LLaMEA tasks."""

from .dataset import FSSPInstance, load_manifest, load_split, parse_instance
from .evaluation_core import compute_makespan, macro_average_fitness, relative_gap, validate_sequence

__all__ = [
    "FSSPInstance",
    "load_manifest",
    "load_split",
    "parse_instance",
    "compute_makespan",
    "macro_average_fitness",
    "relative_gap",
    "validate_sequence",
]
