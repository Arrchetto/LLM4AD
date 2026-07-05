"""Evaluator for LLaMEA optimizer classes on the FSSP protocol."""

from __future__ import annotations

import csv
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llm4ad.base import Evaluation
from llm4ad.task.experiment.fssp_common.dataset import (
    DEFAULT_MANIFEST_PATH,
    FSSPInstance,
    load_split,
)
from llm4ad.task.experiment.fssp_common.evaluation_core import (
    compute_makespan,
    macro_average_fitness,
    relative_gap,
    validate_sequence,
)

from .template import task_description, template_program

__all__ = ["FSSPLLAMEAClassEvaluation"]


@dataclass(frozen=True)
class ReferenceValue:
    makespan: int
    status: str
    source: str


def _load_references(path: str | Path) -> dict[str, ReferenceValue]:
    """Parse the best-known metadata CSV into a lookup table."""
    path = Path(path)
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as error:
        raise FileNotFoundError(f"cannot read best-known metadata: {path}") from error

    references: dict[str, ReferenceValue] = {}
    for row in rows:
        instance_id = (row.get("instance_id") or "").strip()
        value = (row.get("best_known_value") or "").strip()
        status = (row.get("status") or "").strip().upper()
        source = (row.get("source") or "").strip()
        if not instance_id:
            raise ValueError("best-known metadata contains an empty instance_id")
        if instance_id in references:
            raise ValueError(f"duplicate best-known row: {instance_id}")
        if status not in {"OPTIMAL", "BEST_KNOWN", "LOWER_BOUND", "UNVERIFIED"}:
            raise ValueError(f"invalid reference status for {instance_id}: {status}")
        if status in {"OPTIMAL", "BEST_KNOWN", "LOWER_BOUND"}:
            try:
                makespan = int(value)
            except ValueError as error:
                raise ValueError(f"invalid reference value for {instance_id}") from error
            if makespan <= 0:
                raise ValueError(f"reference value must be positive: {instance_id}")
        else:
            makespan = 0
        references[instance_id] = ReferenceValue(makespan, status, source)
    return references


class FSSPLLAMEAClassEvaluation(Evaluation):
    """Evaluate complete LLaMEA optimizer classes on a frozen Taillard split."""

    candidate_type = "class"
    candidate_name = "FlowShopOptimizer"
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
        # Blank or empty data_root from the GUI should resolve to the default path.
        if data_root is not None and not str(data_root).strip():
            data_root = None
        self.split = split
        self.instances = load_split(split, data_root, manifest_path)
        reference_path = (
            Path(best_known_path)
            if best_known_path
            else Path(__file__).resolve().parents[1] / "fssp_common/best_known.csv"
        )
        self.references = _load_references(reference_path)
        self._validate_reference_coverage(self.instances)

    def _validate_reference_coverage(self, instances: tuple[FSSPInstance, ...]) -> None:
        missing = [item.instance_id for item in instances if item.instance_id not in self.references]
        if missing:
            raise ValueError(f"missing best-known metadata for: {', '.join(missing[:5])}")
        if self.split == "train":
            invalid = [
                item.instance_id
                for item in instances
                if self.references[item.instance_id].status not in {"OPTIMAL", "BEST_KNOWN"}
                or not self.references[item.instance_id].source
            ]
            if invalid:
                raise ValueError(
                    "training references must be sourced OPTIMAL or BEST_KNOWN values: "
                    + ", ".join(invalid[:5])
                )

    @staticmethod
    def _instance_to_dict(instance: FSSPInstance) -> dict[str, Any]:
        """Convert an FSSPInstance into the dict expected by candidate classes."""
        return {
            "instance_id": instance.instance_id,
            "n": instance.n,
            "m": instance.m,
            "matrix": [list(row) for row in instance.matrix],
            "family": instance.family,
            "subseries": instance.subseries,
        }

    def evaluate_program(
        self,
        program_str: str,
        callable_func: callable,
        **kwargs: Any,
    ) -> float | None:
        return self.evaluate(callable_func)

    def evaluate(self, optimizer: Any) -> float | None:
        """Evaluate a candidate optimizer class or callable on the loaded split."""
        series_gaps: list[tuple[str, float]] = []
        try:
            for instance in self.instances:
                instance_dict = self._instance_to_dict(instance)
                if inspect.isclass(optimizer):
                    sequence = optimizer()(instance_dict)
                else:
                    sequence = optimizer(instance_dict)
                validated = validate_sequence(instance, sequence)
                reference = self.references[instance.instance_id]
                if reference.status not in {"OPTIMAL", "BEST_KNOWN"}:
                    return None
                makespan = compute_makespan(instance, validated)
                series_gaps.append(
                    (instance.subseries, relative_gap(makespan, reference.makespan))
                )
            return macro_average_fitness(series_gaps)
        except Exception:
            return None
