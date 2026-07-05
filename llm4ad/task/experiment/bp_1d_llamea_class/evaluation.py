from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from llm4ad.base import Evaluation
from llm4ad.task.experiment.bp_1d_common.dataset import (
    DEFAULT_MANIFEST_PATH,
    load_split,
)
from llm4ad.task.experiment.bp_1d_common.evaluation_core import (
    macro_average_fitness,
    relative_gap,
    validate_packing,
)
from llm4ad.task.experiment.bp_1d_common.references import (
    load_references,
    validate_reference_coverage,
)

from .template import task_description, template_program


class BP1DLLaMEAClassEvaluation(Evaluation):
    """Evaluate complete LLaMEA optimizer classes on a frozen BPPLIB split."""

    candidate_type = "class"
    candidate_name = "BinPackingOptimizer"
    candidate_call_signature = ("instance",)
    supported_methods = ("LLaMEA",)

    def __init__(
        self,
        timeout_seconds: int | float = 30,
        data_root: str | Path | None = None,
        split: str = "train",
        manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
        best_known_path: str | Path | None = None,
        **kwargs: Any,
    ):
        super().__init__(
            template_program=template_program,
            task_description=task_description,
            use_numba_accelerate=False,
            timeout_seconds=timeout_seconds,
            **kwargs,
        )
        self.split = split
        self.instances = load_split(split, data_root, manifest_path)
        reference_path = (
            Path(best_known_path)
            if best_known_path
            else Path(__file__).resolve().parents[1] / "bp_1d_common/best_known.csv"
        )
        self.references = load_references(reference_path)
        validate_reference_coverage(self.instances, self.references, self.split)

    def evaluate_program(
        self,
        program_str: str,
        callable_func: callable,
        **kwargs: Any,
    ) -> float | None:
        return self.evaluate(callable_func)

    def evaluate(self, optimizer_class: type) -> float | None:
        series_gaps: list[tuple[str, float]] = []
        try:
            for instance in self.instances:
                optimizer = optimizer_class()
                instance_dict = {
                    "instance_id": instance.instance_id,
                    "bin_capacity": instance.bin_capacity,
                    "num_items": instance.num_items,
                    "items": list(instance.items),
                }
                solution = optimizer(instance_dict)
                if not isinstance(solution, dict):
                    return None
                candidate_bins = solution.get("bins")
                if not isinstance(candidate_bins, list) or any(
                    not isinstance(bin_items, list) for bin_items in candidate_bins
                ):
                    return None
                bins = validate_packing(instance, solution)
                reference = self.references[instance.instance_id]
                if reference.status not in {"OPTIMAL", "BEST_KNOWN"}:
                    return None
                series_gaps.append(
                    (instance.subseries, relative_gap(len(bins), reference.bins))
                )
            fitness = macro_average_fitness(series_gaps)
            return fitness if math.isfinite(fitness) else None
        except Exception:
            return None


__all__ = ["BP1DLLaMEAClassEvaluation"]
