import inspect
import math
import multiprocessing
import sys

from llamea import Solution, prepare_namespace
from llm4ad.base import Function, TextFunctionProgramConverter
from llm4ad.base.evaluate import Evaluation


class CandidateEvaluationTimeoutError(TimeoutError):
    """Raised when a LLaMEA candidate exceeds its evaluation budget."""


def _candidate_evaluation_timeout(for_instance: Evaluation) -> float | None:
    """Return the active hard timeout for safe evaluation, if configured."""
    if getattr(for_instance, "safe_evaluate", None) is not True:
        return None
    timeout = getattr(for_instance, "timeout_seconds", None)
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        return None
    timeout = float(timeout)
    return timeout if math.isfinite(timeout) and timeout > 0 else None


def _candidate_evaluation_worker(
    result_connection,
    for_instance: Evaluation,
    code: str,
    candidate_type: str,
    executable_name: str,
) -> None:
    """Execute and evaluate candidate source inside an isolated process."""
    try:
        global_ns, _ = prepare_namespace(
            code,
            allowed=["pandas", "numpy", "numbas"],
        )
        local_ns = {}
        if candidate_type == "class":
            exec(code, global_ns)
            executable = global_ns[executable_name]
        else:
            exec(code, global_ns, local_ns)
            executable = local_ns[executable_name]
        result_connection.send(("success", for_instance.evaluate(executable)))
    except BaseException as error:
        result_connection.send(
            ("error", f"{type(error).__name__}: {error}")
        )
    finally:
        result_connection.close()


def _terminate_process(process: multiprocessing.Process) -> None:
    """Stop a candidate process and wait until no worker remains alive."""
    if process.is_alive():
        process.terminate()
        process.join(timeout=1)
    if process.is_alive():
        process.kill()
        process.join()


def _evaluate_with_hard_timeout(
    for_instance: Evaluation,
    code: str,
    executable,
    candidate_type: str,
    executable_name: str,
):
    """Evaluate directly or in a killable subprocess when safe mode is on."""
    timeout = _candidate_evaluation_timeout(for_instance)
    if timeout is None:
        return for_instance.evaluate(executable)

    fork_proc = getattr(for_instance, "fork_proc", "auto")
    use_fork = fork_proc is True or (
        fork_proc == "auto"
        and (
            sys.platform.startswith("darwin")
            or sys.platform.startswith("linux")
        )
    )
    context = multiprocessing.get_context("fork" if use_fork else "spawn")
    parent_connection, child_connection = context.Pipe(duplex=False)
    process = context.Process(
        target=_candidate_evaluation_worker,
        args=(
            child_connection,
            for_instance,
            code,
            candidate_type,
            executable_name,
        ),
        daemon=getattr(for_instance, "daemon_eval_process", False),
    )
    process.start()
    child_connection.close()
    try:
        if not parent_connection.poll(timeout):
            _terminate_process(process)
            raise CandidateEvaluationTimeoutError(
                f"Candidate evaluation timed out after {timeout:g} seconds"
            )
        try:
            status, payload = parent_connection.recv()
        except EOFError as error:
            raise RuntimeError(
                "Candidate evaluation process exited without a result"
            ) from error
        process.join(timeout=1)
        if status == "error":
            raise RuntimeError(payload)
        return payload
    finally:
        parent_connection.close()
        _terminate_process(process)
        process.close()


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


def _register_with_profiler(
        solution: Solution,
        profiler,
        gui_score,
        candidate_name: str | None = None,
) -> None:
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
            name=candidate_name or solution.name or "invalid_candidate",
            args="",
            body="    pass",
        )

    function.score = gui_score
    function.operator = solution.operator or "LLaMEA"
    profiler.register_function(function, program=solution.code)


def _required_constructor_parameters(candidate_class: type) -> list[str]:
    """Return constructor parameters that prevent no-argument construction."""
    signature = inspect.signature(candidate_class)
    return [
        parameter.name
        for parameter in signature.parameters.values()
        if parameter.default is inspect.Parameter.empty
        and parameter.kind not in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        )
    ]


def _validate_class_call_signature(
        candidate_class: type,
        expected_names: tuple[str, ...],
) -> str | None:
    """Return an error message when __call__ does not match the contract."""
    try:
        signature = inspect.signature(candidate_class.__call__)
    except Exception as error:
        return f"Candidate __call__ signature inspection failed: {error}"

    parameters = list(signature.parameters.values())
    if parameters and parameters[0].name == "self":
        parameters = parameters[1:]

    actual_names = tuple(parameter.name for parameter in parameters)
    exact_positional_parameters = all(
        parameter.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
        and parameter.default is inspect.Parameter.empty
        for parameter in parameters
    )
    if actual_names != expected_names or not exact_positional_parameters:
        return (
            "Candidate __call__ signature mismatch: "
            f"expected {expected_names}, got {actual_names}."
        )
    return None


def generate_evaluator(for_instance: Evaluation, profiler=None):
    """A LLaMEA instance works on llamea.Solution object, this generator
    takes the instance of evaluation, that have evaluate member mapping 
    Callable -> float, and returns a function that takes that `float` value 
    to update the `Solution` with appropriate fitness.
    """

    candidate_type = (
        "class"
        if getattr(for_instance, "candidate_type", None) == "class"
        else "function"
    )
    if candidate_type == "class":
        candidate_name = getattr(for_instance, "candidate_name", None)
        candidate_call_signature = tuple(
            getattr(for_instance, "candidate_call_signature", ())
        )
    else:
        candidate_name = None
        candidate_call_signature = ()

    if candidate_type == "class":
        template_signatures = {}
        template_signature_errors = {}
        sole_template_function_name = None
    else:
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
            if candidate_type == "class":
                exec(code, global_ns)
            else:
                exec(code, global_ns, local_ns)

        except Exception as e:
            solution.set_scores(
                float("-inf"),  # Always maximisation problem in llm4ad.
                (possible_issue if possible_issue else "") + f". Exec block failed to execute.",
                e
            )
            _register_with_profiler(
                solution,
                profiler,
                None,
                candidate_name=candidate_name,
            )
            return solution

        if candidate_type == "class":
            executable = global_ns.get(candidate_name)
            if not inspect.isclass(executable):
                error = ValueError(
                    f"Candidate must define class {candidate_name}."
                )
                solution.set_scores(float("-inf"), str(error), None)
                _register_with_profiler(
                    solution,
                    profiler,
                    None,
                    candidate_name=candidate_name,
                )
                return solution

            try:
                required_parameters = _required_constructor_parameters(
                    executable
                )
            except Exception as error:
                solution.set_scores(
                    float("-inf"),
                    f"Candidate constructor inspection failed: {error}",
                    error,
                )
                _register_with_profiler(
                    solution,
                    profiler,
                    None,
                    candidate_name=candidate_name,
                )
                return solution

            if required_parameters:
                error = ValueError(
                    "Candidate constructor must accept no arguments; "
                    f"required parameters: {required_parameters}."
                )
                solution.set_scores(float("-inf"), str(error), None)
                _register_with_profiler(
                    solution,
                    profiler,
                    None,
                    candidate_name=candidate_name,
                )
                return solution

            signature_error = _validate_class_call_signature(
                executable,
                candidate_call_signature,
            )
            if signature_error is not None:
                solution.set_scores(float("-inf"), signature_error, None)
                _register_with_profiler(
                    solution,
                    profiler,
                    None,
                    candidate_name=candidate_name,
                )
                return solution
        else:
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
            score = _evaluate_with_hard_timeout(
                for_instance=for_instance,
                code=code,
                executable=executable,
                candidate_type=candidate_type,
                executable_name=(
                    candidate_name
                    if candidate_type == "class"
                    else solution.name
                ),
            )
            if score is None:
                raise ValueError("Evaluation returned an invalid score: None")
            score = float(score)
            if not math.isfinite(score):
                raise ValueError(
                    f"Evaluation returned an invalid score: {score}"
                )
            if candidate_type == "class":
                feedback = f"The optimizer class fitness is {score}."
            else:
                feedback = (
                    f"The average distance of this heursitic is {score}."
                )
            solution.set_scores(
                score,
                feedback,
                None
            )
            _register_with_profiler(
                solution,
                profiler,
                score,
                candidate_name=candidate_name,
            )
            return solution
        except Exception as e:
            solution.set_scores(
                float("-inf"),
                f"Code failed to execute {e}.",
                e
            )
            _register_with_profiler(
                solution,
                profiler,
                None,
                candidate_name=candidate_name,
            )
            return solution
    return evaluator
