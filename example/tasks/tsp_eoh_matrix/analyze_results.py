from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import re
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.figure import Figure

from llm4ad.task.optimization.tsp_eoh_matrix.evaluation import (
    normalize_tour,
    tour_cost,
    validate_distance_matrix,
)


SAMPLE_FILE_PATTERN = re.compile(r"^samples_(\d+)~(\d+)\.json$")


def _sample_file_key(path: Path) -> tuple[int, int, str]:
    match = SAMPLE_FILE_PATTERN.fullmatch(path.name)
    if match is None:
        return (sys.maxsize, sys.maxsize, path.name)
    return (int(match.group(1)), int(match.group(2)), path.name)


def _normalized_score(score: Any) -> tuple[float | None, str | None]:
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        return None, f"score is not a numeric scalar: {score!r}"
    fitness = float(score)
    if not math.isfinite(fitness):
        return None, f"score is not finite: {score!r}"
    return fitness, None


def read_sample_history(run_dir: str | Path) -> list[dict[str, Any]]:
    """Read complete profiler history and explicitly annotate invalid scores."""
    samples_dir = Path(run_dir) / "samples"
    if not samples_dir.is_dir():
        raise ValueError(f"Sample directory does not exist: {samples_dir}")
    paths = sorted(
        (path for path in samples_dir.iterdir() if SAMPLE_FILE_PATTERN.fullmatch(path.name)),
        key=_sample_file_key,
    )
    if not paths:
        raise ValueError(f"No sample history files found in {samples_dir}")

    samples: list[dict[str, Any]] = []
    seen_orders: set[int] = set()
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8")
            if not text.strip():
                raise ValueError("file is empty")
            records = json.loads(text)
        except (OSError, json.JSONDecodeError, ValueError) as error:
            raise ValueError(f"Invalid sample log {path.name}: {error}") from error
        if not isinstance(records, list) or not records:
            raise ValueError(f"Invalid sample log {path.name}: expected a non-empty JSON list")

        for record_index, record in enumerate(records):
            if not isinstance(record, dict):
                raise ValueError(
                    f"Invalid sample log {path.name}: record {record_index} is not an object"
                )
            sample_order = record.get("sample_order")
            if isinstance(sample_order, bool) or not isinstance(sample_order, int) or sample_order < 1:
                raise ValueError(
                    f"Invalid sample log {path.name}: record {record_index} has invalid sample_order"
                )
            if sample_order in seen_orders:
                raise ValueError(
                    f"Invalid sample log {path.name}: duplicate sample_order {sample_order}"
                )
            seen_orders.add(sample_order)
            normalized = dict(record)
            normalized["fitness"], normalized["invalid_reason"] = _normalized_score(
                record.get("score")
            )
            samples.append(normalized)

    samples.sort(key=lambda sample: sample["sample_order"])
    return samples


def select_best_sample(samples: list[dict[str, Any]]) -> dict[str, Any]:
    valid_samples = [sample for sample in samples if sample.get("fitness") is not None]
    if not valid_samples:
        raise ValueError("No sample has a valid fitness")
    return max(valid_samples, key=lambda sample: (sample["fitness"], -sample["sample_order"]))


def compile_solve(program: Any):
    """Compile a profiler program after confirming its sole top-level function is solve."""
    if not isinstance(program, str) or not program.strip():
        raise ValueError("Best sample is missing a non-empty program")
    try:
        tree = ast.parse(program)
    except SyntaxError as error:
        raise ValueError(f"Best sample program is invalid Python: {error}") from error
    top_level_functions = [
        node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    if len(top_level_functions) != 1 or top_level_functions[0].name != "solve":
        raise ValueError("Candidate program must export exactly one top-level function named solve")
    if isinstance(top_level_functions[0], ast.AsyncFunctionDef):
        raise ValueError("Candidate solve function must be synchronous")
    disallowed = [
        node
        for node in tree.body
        if not isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef))
    ]
    if disallowed:
        raise ValueError("Candidate program contains unsupported top-level statements")

    namespace: dict[str, Any] = {}
    try:
        exec(compile(tree, "<best-tsp-sample>", "exec"), namespace)
    except Exception as error:
        raise ValueError(f"Candidate program could not be compiled: {error}") from error
    solve = namespace.get("solve")
    if not callable(solve):
        raise ValueError("Candidate program did not export callable solve")
    return solve


def write_convergence_artifacts(
    run_dir: str | Path, samples: list[dict[str, Any]]
) -> tuple[Path, Path]:
    if not any(sample.get("fitness") is not None for sample in samples):
        raise ValueError("Cannot generate convergence artifacts without a valid fitness")

    rows: list[dict[str, Any]] = []
    best_fitness: float | None = None
    for sample in sorted(samples, key=lambda item: item["sample_order"]):
        fitness = sample.get("fitness")
        if fitness is not None and (best_fitness is None or fitness > best_fitness):
            best_fitness = fitness
        rows.append(
            {
                "sample_order": sample["sample_order"],
                "fitness": fitness,
                "best_fitness": best_fitness,
            }
        )

    output_dir = Path(run_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "convergence_data.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("sample_order", "fitness", "best_fitness")
        )
        writer.writeheader()
        writer.writerows(rows)

    valid_rows = [row for row in rows if row["best_fitness"] is not None]
    figure = Figure(figsize=(8, 5), dpi=150)
    axis = figure.add_subplot(111)
    axis.plot(
        [row["sample_order"] for row in valid_rows],
        [row["best_fitness"] for row in valid_rows],
        color="tab:blue",
        linewidth=1.8,
    )
    axis.set_title("TSP EoH Convergence Curve")
    axis.set_xlabel("Sample Order")
    axis.set_ylabel("Best Fitness")
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    png_path = output_dir / "convergence_curve.png"
    figure.savefig(png_path)
    return csv_path, png_path


def load_benchmark_instances(path: str | Path) -> dict[Any, tuple[np.ndarray, float]]:
    npz_path = Path(path)
    try:
        with np.load(npz_path, allow_pickle=True) as data:
            if "distance_matrix_dict" not in data:
                raise ValueError("NPZ is missing distance_matrix_dict")
            try:
                raw_mapping = data["distance_matrix_dict"].item()
            except (ValueError, AttributeError) as error:
                raise ValueError("distance_matrix_dict must contain one serialized mapping") from error
    except OSError as error:
        raise ValueError(f"Could not read benchmark NPZ {npz_path}: {error}") from error

    if not isinstance(raw_mapping, Mapping) or not raw_mapping:
        raise ValueError("distance_matrix_dict must be a non-empty mapping")

    validated: dict[Any, tuple[np.ndarray, float]] = {}
    for instance, value in raw_mapping.items():
        if not isinstance(value, tuple) or len(value) != 2:
            raise ValueError(f"Benchmark instance {instance!r} must be a (distance_matrix, optimum) tuple")
        try:
            matrix = validate_distance_matrix(value[0])
        except ValueError as error:
            raise ValueError(f"Benchmark instance {instance!r} has invalid distance matrix: {error}") from error
        optimum = value[1]
        if (
            isinstance(optimum, (bool, np.bool_))
            or not isinstance(optimum, (int, float, np.integer, np.floating))
            or not math.isfinite(float(optimum))
            or float(optimum) <= 0.0
        ):
            raise ValueError(f"Benchmark instance {instance!r} has invalid optimum {optimum!r}")
        validated[instance] = (matrix, float(optimum))
    return validated


def evaluate_benchmark(
    solve,
    instances: Mapping[Any, tuple[np.ndarray, float]],
    output_path: str | Path,
) -> list[dict[str, Any]]:
    """Evaluate every benchmark instance, writing CSV only after all succeed."""
    rows: list[dict[str, Any]] = []
    for instance, (distance_matrix, optimum) in instances.items():
        try:
            matrix = validate_distance_matrix(distance_matrix)
            result = solve(matrix.copy())
            normalized = normalize_tour(result, matrix.shape[0])
            cost = tour_cost(matrix, normalized)
            gap = (cost - float(optimum)) / float(optimum) * 100.0
            if not math.isfinite(gap):
                raise ValueError("gap is not finite")
        except Exception as error:
            raise ValueError(f"Benchmark instance {instance!r} failed: {error}") from error
        row = {
            "instance": instance,
            "cost": cost,
            "optimum": float(optimum),
            "gap": gap,
        }
        rows.append(row)
        print(f"{instance}: cost={cost:.12g}, optimum={float(optimum):.12g}, gap={gap:.6f}%")

    if not rows:
        raise ValueError("No benchmark instances were evaluated")
    csv_path = Path(output_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("instance", "cost", "optimum", "gap"))
        writer.writeheader()
        writer.writerows(rows)
    average_gap = float(np.mean([row["gap"] for row in rows]))
    print(f"Successfully evaluated {len(rows)} benchmark instances.")
    print(f"Average gap: {average_gap:.6f}%")
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze one EoH TSP matrix run and evaluate its best sample."
    )
    parser.add_argument("run_dir", type=Path, help="Concrete EoH run directory")
    parser.add_argument("--instances", required=True, type=Path, help="Benchmark NPZ path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        samples = read_sample_history(args.run_dir)
        best = select_best_sample(samples)
        solve = compile_solve(best.get("program"))
        csv_path, png_path = write_convergence_artifacts(args.run_dir, samples)
        instances = load_benchmark_instances(args.instances)
        benchmark_path = args.run_dir / "benchmark_results.csv"
        evaluate_benchmark(solve, instances, benchmark_path)
        print(
            f"Best sample: order={best['sample_order']}, fitness={best['fitness']:.12g}"
        )
        print(f"Convergence CSV: {csv_path}")
        print(f"Convergence PNG: {png_path}")
        print(f"Benchmark CSV: {benchmark_path}")
        return 0
    except Exception as error:
        print(f"Analysis failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
