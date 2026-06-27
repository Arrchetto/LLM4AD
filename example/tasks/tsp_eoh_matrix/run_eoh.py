from __future__ import annotations

import argparse
import os
from pathlib import Path

from llm4ad.method.eoh import EoH, EoHProfiler
from llm4ad.task.optimization.tsp_eoh_matrix import TSPEoHMatrixEvaluation
from llm4ad.tools.llm.llm_api_https import HttpsApi


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LOG_DIR = REPO_ROOT / "GUI" / "logs" / "eoh" / "tsp_eoh_matrix"
API_KEY_ENV = "LLM4AD_LLM_API_KEY"


class _ProfilerSafeHttpsApi(HttpsApi):
    """Expose a redacted attribute mapping to the reflection-based profiler."""

    def __getattribute__(self, name):
        if name == "__dict__":
            attributes = dict(object.__getattribute__(self, "__dict__"))
            if "_key" in attributes:
                attributes["_key"] = "<redacted>"
            return attributes
        return super().__getattribute__(name)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed


def _optional_positive_int(value: str) -> int | None:
    if value.strip().lower() == "none":
        return None
    return _positive_int(value)


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be greater than 0")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run EoH on the complete-tour TSP distance-matrix task.",
        epilog=(
            "Formal 1000-sample configuration (documented only; do not use for offline validation): "
            "python example/tasks/tsp_eoh_matrix/run_eoh.py --max-sample-nums 1000 "
            "--max-generations 1000 --pop-size none --num-samplers 1 --num-evaluators 1. "
            f"Set {API_KEY_ENV}; never put an API key on the command line."
        ),
    )
    parser.add_argument("--max-sample-nums", type=_positive_int, default=20)
    parser.add_argument("--max-generations", type=_optional_positive_int, default=10)
    parser.add_argument(
        "--pop-size",
        type=_optional_positive_int,
        default=2,
        help="Positive integer, or 'none' to let EoH auto-adjust (20 at 1000 samples).",
    )
    parser.add_argument("--num-samplers", type=_positive_int, default=1)
    parser.add_argument("--num-evaluators", type=_positive_int, default=1)
    parser.add_argument("--timeout-seconds", type=_positive_float, default=60.0)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument(
        "--llm-host",
        default=os.environ.get("LLM4AD_LLM_HOST"),
        help="HTTPS host/base URL; defaults to LLM4AD_LLM_HOST.",
    )
    parser.add_argument(
        "--llm-model",
        default=os.environ.get("LLM4AD_LLM_MODEL"),
        help="Model name; defaults to LLM4AD_LLM_MODEL.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    api_key = os.environ.get(API_KEY_ENV)
    if not args.llm_host:
        parser.error("--llm-host or LLM4AD_LLM_HOST is required")
    if not args.llm_model:
        parser.error("--llm-model or LLM4AD_LLM_MODEL is required")
    if not api_key:
        parser.error(f"{API_KEY_ENV} must be set in the environment")

    llm = _ProfilerSafeHttpsApi(
        host=args.llm_host,
        key=api_key,
        model=args.llm_model,
        timeout=args.timeout_seconds,
    )
    task = TSPEoHMatrixEvaluation(timeout_seconds=args.timeout_seconds)
    profiler = EoHProfiler(
        log_dir=str(args.log_dir),
        log_style="complex",
        create_random_path=True,
    )
    method = EoH(
        llm=llm,
        evaluation=task,
        profiler=profiler,
        max_sample_nums=args.max_sample_nums,
        max_generations=args.max_generations,
        pop_size=args.pop_size,
        num_samplers=args.num_samplers,
        num_evaluators=args.num_evaluators,
    )
    print(f"Run directory: {profiler._log_dir}")
    print("Prompt/completion token counts and API cost: unavailable from HttpsApi.")
    method.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
