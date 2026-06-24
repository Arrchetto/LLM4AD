import inspect
import math
import unittest
from unittest.mock import Mock, patch

from llamea import Solution, prepare_namespace

from llm4ad.method.llamea.evaluation import generate_evaluator


class _ValidEvaluation:
    def evaluate(self, _callable):
        return 12.5


class _InvalidEvaluation:
    def evaluate(self, _callable):
        return None


class LlameaEvaluationAdapterTest(unittest.TestCase):
    def test_numpy_array_default_is_treated_as_same_signature(self):
        evaluation = Mock()
        evaluation.template_program = (
            "import numpy as np\n\n"
            "def candidate(value=np.array([1, 2])):\n"
            "    return value\n"
        )
        evaluation.evaluate.return_value = 12.5
        solution = Solution(
            name="candidate",
            code=(
                "import numpy as np\n\n"
                "def candidate(value=np.array([1, 2])):\n"
                "    return value\n"
            ),
        )

        evaluated = generate_evaluator(evaluation)(solution)

        evaluation.evaluate.assert_called_once()
        self.assertEqual(evaluated.fitness, 12.5)

    def test_nan_default_is_treated_as_same_signature(self):
        evaluation = Mock()
        evaluation.template_program = (
            "def candidate(value=float('nan')):\n"
            "    return value\n"
        )
        evaluation.evaluate.return_value = 12.5
        solution = Solution(
            name="candidate",
            code=(
                "def candidate(value=float('nan')):\n"
                "    return value\n"
            ),
        )

        evaluated = generate_evaluator(evaluation)(solution)

        evaluation.evaluate.assert_called_once()
        self.assertEqual(evaluated.fitness, 12.5)

    def test_parameter_kind_mismatch_is_rejected(self):
        profiler = Mock()
        evaluation = Mock()
        evaluation.template_program = (
            "def candidate(value, *, limit=1):\n"
            "    return value\n"
        )
        evaluation.evaluate.return_value = 99.0
        solution = Solution(
            name="candidate",
            code=(
                "def candidate(value, limit=1):\n"
                "    return value\n"
            ),
        )

        evaluated = generate_evaluator(
            evaluation,
            profiler=profiler,
        )(solution)

        evaluation.evaluate.assert_not_called()
        self.assertEqual(evaluated.fitness, float("-inf"))
        self.assertIn("signature", evaluated.feedback.lower())
        profiler.register_function.assert_called_once()

    def test_uninspectable_candidate_is_rejected_without_escaping(self):
        profiler = Mock()
        evaluation = Mock()
        evaluation.template_program = (
            "def candidate(value):\n"
            "    return value\n"
        )
        evaluation.evaluate.return_value = 99.0
        solution = Solution(
            name="candidate",
            code=(
                "def candidate(value):\n"
                "    return value\n"
            ),
        )
        expected_signature = inspect.signature(lambda value: value)

        with patch(
            "llm4ad.method.llamea.evaluation.inspect.signature",
            side_effect=[expected_signature, TypeError("not inspectable")],
        ):
            evaluator = generate_evaluator(
                evaluation,
                profiler=profiler,
            )
            evaluated = evaluator(solution)

        evaluation.evaluate.assert_not_called()
        self.assertEqual(evaluated.fitness, float("-inf"))
        self.assertIn("signature", evaluated.feedback.lower())
        profiler.register_function.assert_called_once()
        registered_function = profiler.register_function.call_args.args[0]
        self.assertIsNone(registered_function.score)

    def test_uninspectable_template_is_rejected_without_escaping(self):
        profiler = Mock()
        evaluation = Mock()
        evaluation.template_program = (
            "def candidate(value):\n"
            "    return value\n"
        )
        solution = Solution(
            name="candidate",
            code=(
                "def candidate(value):\n"
                "    return value\n"
            ),
        )

        with patch(
            "llm4ad.method.llamea.evaluation.inspect.signature",
            side_effect=ValueError("template is not inspectable"),
        ):
            evaluator = generate_evaluator(
                evaluation,
                profiler=profiler,
            )
            evaluated = evaluator(solution)

        evaluation.evaluate.assert_not_called()
        self.assertEqual(evaluated.fitness, float("-inf"))
        self.assertIn("signature", evaluated.feedback.lower())
        profiler.register_function.assert_called_once()

    def test_template_execution_is_cached_when_evaluator_is_created(self):
        evaluation = Mock()
        evaluation.template_program = (
            "def candidate(value):\n"
            "    return value\n"
        )
        evaluation.evaluate.return_value = 12.5

        with patch(
            "llm4ad.method.llamea.evaluation.prepare_namespace",
            wraps=prepare_namespace,
        ) as namespace_preparer:
            evaluator = generate_evaluator(evaluation)
            evaluator(
                Solution(
                    name="candidate",
                    code="def candidate(value):\n    return value + 0\n",
                )
            )
            evaluator(
                Solution(
                    name="candidate",
                    code="def candidate(value):\n    return value + 1\n",
                )
            )

        template_calls = [
            call
            for call in namespace_preparer.call_args_list
            if call.args[0] == evaluation.template_program
        ]
        self.assertEqual(len(template_calls), 1)

    def test_annotation_differences_do_not_reject_candidate(self):
        evaluation = Mock()
        evaluation.template_program = (
            "def candidate(value: int) -> int:\n"
            "    return value\n"
        )
        evaluation.evaluate.return_value = 12.5
        solution = Solution(
            name="candidate",
            code=(
                "def candidate(value: str) -> str:\n"
                "    return value\n"
            ),
        )

        evaluated = generate_evaluator(evaluation)(solution)

        evaluation.evaluate.assert_called_once()
        self.assertEqual(evaluated.fitness, 12.5)

    def test_signature_mismatch_is_rejected_before_expensive_evaluation(self):
        profiler = Mock()
        evaluation = Mock()
        evaluation.template_program = (
            "import numpy as np\n\n"
            "def select_next_node(\n"
            "    current_node: int,\n"
            "    destination_node: int,\n"
            "    unvisited_nodes: np.ndarray,\n"
            "    distance_matrix: np.ndarray,\n"
            "    prizes: np.ndarray,\n"
            "    remaining_budget: float,\n"
            ") -> int:\n"
            "    return int(unvisited_nodes[0])\n"
        )
        evaluation.evaluate.return_value = 99.0
        code = (
            "import numpy as np\n\n"
            "def select_next_node(\n"
            "    current_node: int,\n"
            "    destination_node: int,\n"
            "    unvisited_nodes: np.ndarray,\n"
            "    distance_matrix: np.ndarray,\n"
            "    prizes: np.ndarray,\n"
            "    remaining_budget: float,\n"
            "    cumulative_distance: float,\n"
            ") -> int:\n"
            "    return int(unvisited_nodes[0])\n"
        )
        solution = Solution(name="select_next_node", code=code)

        evaluated = generate_evaluator(
            evaluation,
            profiler=profiler,
        )(solution)

        evaluation.evaluate.assert_not_called()
        self.assertEqual(evaluated.fitness, float("-inf"))
        self.assertIn("signature", evaluated.feedback.lower())
        self.assertEqual(evaluated.code, code)
        profiler.register_function.assert_called_once()
        registered_function = profiler.register_function.call_args.args[0]
        self.assertIsNone(registered_function.score)
        self.assertEqual(
            profiler.register_function.call_args.kwargs["program"],
            code,
        )

    def test_valid_candidate_is_registered_with_gui_profiler(self):
        profiler = Mock()
        solution = Solution(
            name="candidate",
            code="def candidate():\n    return 0\n",
        )

        evaluated = generate_evaluator(
            _ValidEvaluation(),
            profiler=profiler,
        )(solution)

        registered_function = profiler.register_function.call_args.args[0]
        self.assertEqual(evaluated.fitness, 12.5)
        self.assertEqual(registered_function.score, 12.5)
        self.assertEqual(registered_function.name, "candidate")
        self.assertEqual(
            profiler.register_function.call_args.kwargs["program"],
            solution.code,
        )

    def test_profiler_uses_named_candidate_when_code_has_helper_functions(self):
        profiler = Mock()
        solution = Solution(
            name="candidate",
            code=(
                "def helper():\n"
                "    return 1\n\n"
                "def candidate():\n"
                "    return helper()\n"
            ),
        )

        generate_evaluator(
            _ValidEvaluation(),
            profiler=profiler,
        )(solution)

        registered_function = profiler.register_function.call_args.args[0]
        self.assertEqual(registered_function.name, "candidate")
        self.assertIn("return helper()", str(registered_function))

    def test_none_score_marks_candidate_as_failed(self):
        profiler = Mock()
        solution = Solution(
            name="candidate",
            code="def candidate():\n    return 0\n",
        )

        evaluated = generate_evaluator(
            _InvalidEvaluation(),
            profiler=profiler,
        )(solution)

        self.assertEqual(evaluated.fitness, float("-inf"))
        self.assertTrue(math.isinf(evaluated.fitness))
        self.assertIn("invalid score", evaluated.feedback.lower())
        registered_function = profiler.register_function.call_args.args[0]
        self.assertIsNone(registered_function.score)

    def test_non_finite_scores_mark_candidate_as_failed(self):
        for score in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(score=score):
                profiler = Mock()
                evaluation = Mock()
                evaluation.template_program = (
                    "def candidate():\n"
                    "    return 0\n"
                )
                evaluation.evaluate.return_value = score
                solution = Solution(
                    name="candidate",
                    code="def candidate():\n    return 0\n",
                )

                evaluated = generate_evaluator(
                    evaluation,
                    profiler=profiler,
                )(solution)

                self.assertEqual(evaluated.fitness, float("-inf"))
                self.assertIn("invalid score", evaluated.feedback.lower())
                registered_function = profiler.register_function.call_args.args[0]
                self.assertIsNone(registered_function.score)


if __name__ == "__main__":
    unittest.main()
