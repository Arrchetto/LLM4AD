import math
import unittest
from unittest.mock import Mock

from llamea import Solution

from llm4ad.method.llamea.evaluation import generate_evaluator


class _ValidEvaluation:
    def evaluate(self, _callable):
        return 12.5


class _InvalidEvaluation:
    def evaluate(self, _callable):
        return None


class LlameaEvaluationAdapterTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
