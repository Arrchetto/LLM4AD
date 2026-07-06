from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from llm4ad.base import Evaluation
from llm4ad.task.experiment.bp_1d_common.dataset import (
    DEFAULT_MANIFEST_PATH,
    load_split,
)
from llm4ad.task.experiment.bp_1d_common.evaluation_core import (
    macro_average_fitness,
    macro_average_metric,
    packing_concentration,
    relative_gap,
    validate_packing,
)
from llm4ad.task.experiment.bp_1d_common.references import (
    load_references,
    validate_reference_coverage,
)

from .template import task_description, template_program

__all__ = ["BP1DEoHFullEvaluation", "SelectionFitness"]


_BANNED_IMPORTS = {
    "cvxpy",
    "mip",
    "ortools",
    "pulp",
    "requests",
    "scipy",
    "socket",
    "subprocess",
}


def _uses_external_solver_or_io(program_str: str) -> bool:
    """Reject optimizer delegation and external I/O before candidate execution."""
    try:
        tree = ast.parse(program_str)
    except SyntaxError:
        return True
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            modules = [node.module or ""]
        else:
            modules = []
        for module in modules:
            root = module.split(".", 1)[0]
            if root in _BANNED_IMPORTS:
                return True
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in {"__import__", "eval", "exec", "open"}:
                return True
    return False


class SelectionFitness(float):
    """Primary fitness with a selection-only secondary tie-break value."""

    def __new__(
        cls,
        primary: float,
        tie_break: float = 0.0,
    ) -> "SelectionFitness":
        value = super().__new__(cls, primary)
        value.tie_break = float(tie_break)
        return value

    def __reduce__(self):
        return type(self), (float(self), self.tie_break)


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
        self.references = load_references(reference_path)
        validate_reference_coverage(self.instances, self.references, self.split)

    def evaluate_program(
        self,
        program_str: str,
        callable_func: callable,
        **kwargs: Any,
    ) -> float | None:
        if _uses_external_solver_or_io(program_str):
            return None
        return self.evaluate(callable_func)

    def evaluate(self, solver: callable) -> float | None:
        series_gaps: list[tuple[str, float]] = []
        series_concentration: list[tuple[str, float]] = []
        try:
            for instance in self.instances:
                solution = solver(
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
                series_concentration.append(
                    (instance.subseries, packing_concentration(instance, bins))
                )
            return SelectionFitness(
                macro_average_fitness(series_gaps),
                tie_break=macro_average_metric(series_concentration),
            )
        except Exception:
            return None
