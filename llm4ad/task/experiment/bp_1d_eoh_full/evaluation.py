from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llm4ad.base import Evaluation
from llm4ad.task.experiment.bp_1d_common.dataset import (
    DEFAULT_MANIFEST_PATH,
    BPInstance,
    load_split,
)
from llm4ad.task.experiment.bp_1d_common.evaluation_core import (
    macro_average_fitness,
    relative_gap,
    validate_packing,
)

from .template import task_description, template_program

__all__ = ["BP1DEoHFullEvaluation"]


@dataclass(frozen=True)
class ReferenceValue:
    bins: int
    status: str
    source: str


def _load_references(path: str | Path) -> dict[str, ReferenceValue]:
    path = Path(path)
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as error:
        raise FileNotFoundError(f"cannot read best-known metadata: {path}") from error
    references: dict[str, ReferenceValue] = {}
    for row in rows:
        instance_id = (row.get("instance_id") or "").strip()
        value = (row.get("best_known_bins") or "").strip()
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
                bins = int(value)
            except ValueError as error:
                raise ValueError(f"invalid reference value for {instance_id}") from error
            if bins <= 0:
                raise ValueError(f"reference value must be positive: {instance_id}")
        else:
            bins = 0
        references[instance_id] = ReferenceValue(bins, status, source)
    return references


class BP1DEoHFullEvaluation(Evaluation):
    """Evaluate complete EoH bin-packing solvers on a frozen BPPLIB split."""

    supported_methods = ("EoH",)

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
        self.references = _load_references(reference_path)
        self._validate_reference_coverage(self.instances)

    def _validate_reference_coverage(self, instances: tuple[BPInstance, ...]) -> None:
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

    def evaluate_program(
        self,
        program_str: str,
        callable_func: callable,
        **kwargs: Any,
    ) -> float | None:
        return self.evaluate(callable_func)

    def evaluate(self, solver: callable) -> float | None:
        series_gaps: list[tuple[str, float]] = []
        try:
            for instance in self.instances:
                solution = solver(
                    instance.instance_id,
                    instance.bin_capacity,
                    instance.num_items,
                    list(instance.items),
                )
                bins = validate_packing(instance, solution)
                reference = self.references[instance.instance_id]
                if reference.status not in {"OPTIMAL", "BEST_KNOWN"}:
                    return None
                series_gaps.append(
                    (instance.subseries, relative_gap(len(bins), reference.bins))
                )
            return macro_average_fitness(series_gaps)
        except Exception:
            return None
