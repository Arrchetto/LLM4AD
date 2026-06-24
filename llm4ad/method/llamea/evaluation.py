import inspect
import math

from llamea import Solution, prepare_namespace
from llm4ad.base import Function, TextFunctionProgramConverter
from llm4ad.base.evaluate import Evaluation


def _normalized_parameters(callable_object) -> tuple[tuple[object, ...], ...]:
    """Return parameters relevant to call compatibility, ignoring annotations."""
    signature = inspect.signature(callable_object)
    normalized = []
    for parameter in signature.parameters.values():
        default_is_empty = parameter.default is inspect.Parameter.empty
        normalized.append(
            (
                parameter.name,
                parameter.kind,
                default_is_empty,
                None if default_is_empty else type(parameter.default),
                None if default_is_empty else repr(parameter.default),
            )
        )
    return tuple(normalized)


def _load_template_signatures(
        for_instance: Evaluation,
) -> tuple[
    dict[str, tuple[tuple[object, ...], ...]],
    dict[str, str],
    str | None,
]:
    """Parse, execute, and inspect the task template once per evaluator."""
    template_program = getattr(for_instance, "template_program", None)
    if template_program is None:
        return {}, {}, None

    template_code = str(template_program)
    parsed_program = TextFunctionProgramConverter.text_to_program(template_code)
    if parsed_program is None or not parsed_program.functions:
        return {}, {}, None

    local_ns = {}
    try:
        global_ns, _ = prepare_namespace(
            template_code,
            allowed=["pandas", "numpy", "numbas"],
        )
        exec(template_code, global_ns, local_ns)
    except Exception:
        # Some existing Evaluation implementations do not provide an executable
        # template. Preserve their previous behavior by skipping validation.
        return {}, {}, None

    signatures = {}
    signature_errors = {}
    function_names = [function.name for function in parsed_program.functions]
    for function_name in function_names:
        template_callable = local_ns.get(
            function_name,
            global_ns.get(function_name),
        )
        if not callable(template_callable):
            signature_errors[function_name] = (
                f"Template signature unavailable for {function_name}."
            )
            continue
        try:
            signatures[function_name] = _normalized_parameters(
                template_callable
            )
        except Exception as error:
            signature_errors[function_name] = (
                f"Template signature inspection failed: {error}"
            )

    sole_function_name = function_names[0] if len(function_names) == 1 else None
    return signatures, signature_errors, sole_function_name


def _register_with_profiler(solution: Solution, profiler, gui_score) -> None:
    """Publish a LLaMEA candidate through LLM4AD's GUI profiler pipeline."""
    if profiler is None:
        return

    program = TextFunctionProgramConverter.text_to_program(solution.code)
    function = None
    if program is not None:
        try:
            function = program.get_function(solution.name)
        except ValueError:
            function = None

    if function is None:
        function = Function(
            name=solution.name or "invalid_candidate",
            args="",
            body="    pass",
        )

    function.score = gui_score
    function.operator = solution.operator or "LLaMEA"
    profiler.register_function(function, program=solution.code)


def generate_evaluator(for_instance: Evaluation, profiler=None):
    """A LLaMEA instance works on llamea.Solution object, this generator
    takes the instance of evaluation, that have evaluate member mapping 
    Callable -> float, and returns a function that takes that `float` value 
    to update the `Solution` with appropriate fitness.
    """

    (
        template_signatures,
        template_signature_errors,
        sole_template_function_name,
    ) = _load_template_signatures(for_instance)

    def evaluator(solution: Solution, explogger=None) -> Solution:
        """
            LLaMEA anad llm4ad evaluate functions differently, this function 
            serves as an wrapper to help evaluate the functions properly.

        Args:
            `solution: llamea.Solution`: LLaMEA comes with a `Solution` object that have all
            the arguements necessary for LLaMEA to track it as an individual in population.

            `evaulator: Callable` here is a CVRPEvaluation.evaluate, that takes in a 
            callable function, and returns its score as float.

        Returns:
            `Solution` object with updated score.
        """
        code = solution.code
        possible_issue = None
        local_ns = {}
        try:
            global_ns, possible_issue = prepare_namespace(code, allowed=['pandas', 'numpy', 'numbas'])
            exec(code, global_ns, local_ns)

        except Exception as e:
            solution.set_scores(
                float("-inf"),  # Always maximisation problem in llm4ad.
                (possible_issue if possible_issue else "") + f". Exec block failed to execute.",
                e
            )
            _register_with_profiler(solution, profiler, None)
            return solution
        executable = local_ns[solution.name]

        template_function_name = None
        if (
            solution.name in template_signatures
            or solution.name in template_signature_errors
        ):
            template_function_name = solution.name
        elif sole_template_function_name is not None:
            template_function_name = sole_template_function_name

        if template_function_name in template_signature_errors:
            feedback = template_signature_errors[template_function_name]
            solution.set_scores(float("-inf"), feedback, None)
            _register_with_profiler(solution, profiler, None)
            return solution

        if template_function_name in template_signatures:
            expected_parameters = template_signatures[template_function_name]
            try:
                actual_parameters = _normalized_parameters(executable)
            except Exception as error:
                solution.set_scores(
                    float("-inf"),
                    f"Candidate signature inspection failed: {error}",
                    None,
                )
                _register_with_profiler(solution, profiler, None)
                return solution

            if actual_parameters != expected_parameters:
                error = ValueError(
                    "Function signature mismatch: "
                    f"expected {expected_parameters}, got {actual_parameters}"
                )
                solution.set_scores(
                    float("-inf"),
                    str(error),
                    None,
                )
                _register_with_profiler(solution, profiler, None)
                return solution

        try:
            score = for_instance.evaluate(executable)
            if score is None:
                raise ValueError("Evaluation returned an invalid score: None")
            score = float(score)
            if not math.isfinite(score):
                raise ValueError(
                    f"Evaluation returned an invalid score: {score}"
                )
            solution.set_scores(
                score,
                f"The average distance of this heursitic is {score}.",
                None
            )
            _register_with_profiler(solution, profiler, score)
            return solution
        except Exception as e:
            solution.set_scores(
                float("-inf"),
                f"Code failed to execute {e}.",
                e
            )
            _register_with_profiler(solution, profiler, None)
            return solution
    return evaluator
