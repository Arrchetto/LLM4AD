from decimal import Decimal
from fractions import Fraction
import math
import os
import shutil
from numbers import Real

from llamea import LLaMEA as LLaMEA_Algorithm
from ...base import LLM

from .evaluation import generate_evaluator
from .sampler import LLaMEASampler

from llm4ad.base import Evaluation


def _is_finite_scalar_fitness(value) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, Decimal):
        return value.is_finite()
    if isinstance(value, Fraction):
        return True
    if not isinstance(value, Real):
        return False
    try:
        return math.isfinite(value)
    except (TypeError, ValueError, OverflowError):
        return False


def _relocate_llamea_log_dir(logger, profiler) -> None:
    """Move the current LLaMEA experiment directory under GUI/logs/llamea."""
    if logger is None or profiler is None:
        return

    profiler_log_dir = getattr(profiler, "_log_dir", None)
    source_dir = getattr(logger, "dirname", None)
    if not profiler_log_dir or not source_dir:
        return

    profiler_parent = os.path.dirname(os.path.abspath(profiler_log_dir))
    if (
        os.path.basename(profiler_parent).lower() == "llamea"
        or (
            os.path.basename(profiler_parent).lower() == "test"
            and os.path.basename(os.path.dirname(profiler_parent)).lower()
            == "llamea"
        )
    ):
        target_parent = profiler_parent
    else:
        target_parent = os.path.join(profiler_parent, "llamea")
    target_dir = os.path.join(target_parent, os.path.basename(source_dir))

    if os.path.abspath(source_dir) == os.path.abspath(target_dir):
        return

    os.makedirs(target_parent, exist_ok=True)
    shutil.move(source_dir, target_dir)
    logger.dirname = target_dir


def _write_llamea_log_dir_marker(logger, profiler) -> None:
    """Link the GUI profiler run directory to its LLaMEA experiment directory."""
    if logger is None or profiler is None:
        return

    profiler_log_dir = getattr(profiler, "_log_dir", None)
    llamea_log_dir = getattr(logger, "dirname", None)
    if not profiler_log_dir or not llamea_log_dir:
        return

    os.makedirs(profiler_log_dir, exist_ok=True)
    marker_path = os.path.join(profiler_log_dir, "llamea_output_dir.txt")
    with open(marker_path, "w", encoding="utf-8") as file:
        file.write(os.path.abspath(llamea_log_dir))


class LLaMEA(LLaMEA_Algorithm):
    def __init__(
            self,
            llm : LLM,
            evaluation: Evaluation,
            profiler=None,
            name: str | None = None,
            iterations : int = 50,
            n_parents: int = 5,
            n_offsprings: int=5,
            role_prompt : str = "",
            task_prompt: str = "",
            example_prompt :str | None = None,
            minimization: bool = False,
            elitism: bool = True,
            max_sample_nums: int | None = None,
            samples_per_prompt: int | None = None,
            num_samplers: int | None = None,
            num_evaluators: int = 4,
            eval_timeout: int = 30,
            parallel_backend: str = "threading",
            **kwargs
    ):
        """
        Args:
            evaluation_function: A function for scoring the fitness of the llm generated heuristic.
            llm: An instance of llamea.LLM one of the llms in Ollama, OpenAI, Gemini, DeepSeek,
            evaluation: An instance of llm4ad.base.Evaluation, which defines the way to calculate the score of a generated function.
            profiler: An instance of llm4ad.base.ProfilerBase for logging, or None.
            name: Method name passed by GUI, accepted for compatibility and ignored.
            iterations: Iteration Count for evolution process,
            n_parents: Number of individuals in parent population (λ),
            n_offsprings: Number of individuals in offspring population (µ),
            role_prompt: LLM role prompt like: "You are an excellent scientific programmer tasked to solve the problem of GVRP.",
            task_prompt: Task prompt is llm4ad.tasks.*.template.task_description for solving a problem,
            example_prompt: Example propmt is llm4ad.tasks.*.template.template_program for solving a problem,
            minimisation: Flag to define direction of optimality.
            elitism: A bool flag to run algorithm in (λ + µ) if set True, else (λ , µ).
            max_sample_nums: Total LLaMEA candidate budget supplied by the GUI.
            samples_per_prompt: Not used by LLaMEA, accepted for GUI compatibility.
            num_samplers: Not used by LLaMEA, accepted for GUI compatibility.
            num_evaluators: Number of parallel evaluation workers (mapped to max_workers).
            eval_timeout: Maximum seconds allowed for one candidate evaluation.
            parallel_backend: Joblib backend. Threading avoids pickling the LLaMEA logger
                and is appropriate for concurrent LLM API requests.
        """
        positive_parameters = {
            "n_parents": n_parents,
            "n_offsprings": n_offsprings,
            "num_evaluators": num_evaluators,
            "eval_timeout": eval_timeout,
        }
        if max_sample_nums is not None:
            positive_parameters["max_sample_nums"] = max_sample_nums
        for parameter_name, parameter_value in positive_parameters.items():
            if parameter_value is None or parameter_value <= 0:
                raise ValueError(f"{parameter_name} must be a positive integer.")
        if max_sample_nums is not None and max_sample_nums < n_parents:
            raise ValueError("max_sample_nums must be greater than or equal to n_parents.")

        if not task_prompt:
            task_prompt = evaluation.task_description
        if example_prompt is None:
            example_prompt = str(evaluation.template_program)

        budget = max_sample_nums if max_sample_nums is not None else iterations
        evaluation_function = generate_evaluator(evaluation, profiler=profiler)
        super().__init__(
            f=evaluation_function,
            llm=llm,
            budget=budget,
            n_offspring=n_offsprings,
            n_parents=n_parents,
            role_prompt=role_prompt,
            task_prompt=task_prompt,
            example_prompt=example_prompt,
            minimization=minimization,
            elitism=elitism,
            max_workers=num_evaluators or 10,
            eval_timeout=eval_timeout,
            parallel_backend=parallel_backend,
            **kwargs
        )

        _relocate_llamea_log_dir(getattr(self, "logger", None), profiler)
        self.evaluator = evaluation
        self.profiler = profiler
        self.sampler = LLaMEASampler(llm)
        if profiler is not None:
            profiler.record_parameters(llm, evaluation, self)
            _write_llamea_log_dir_marker(
                getattr(self, "logger", None),
                profiler,
            )

    def selection(self, parents, offspring):
        if not (
            self.n_parents == 1
            and self.n_offspring == 1
            and self.elitism is False
            and self.multi_objective is False
            and self.niching is None
            and len(parents) == 1
            and len(offspring) == 1
        ):
            return super().selection(parents, offspring)

        return (
            offspring
            if _is_finite_scalar_fitness(offspring[0].fitness)
            else parents
        )
