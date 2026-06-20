import math
import unittest

from llamea import Solution

from llm4ad.method.llamea.evaluation import generate_evaluator


class _InvalidEvaluation:
    def evaluate(self, _callable):
        return None


class LlameaEvaluationAdapterTest(unittest.TestCase):
    def test_none_score_marks_candidate_as_failed(self):
        solution = Solution(
            name="candidate",
            code="def candidate():\n    return 0\n",
        )

        evaluated = generate_evaluator(_InvalidEvaluation())(solution)

        self.assertEqual(evaluated.fitness, float("-inf"))
        self.assertTrue(math.isinf(evaluated.fitness))
        self.assertIn("invalid score", evaluated.feedback.lower())


if __name__ == "__main__":
    unittest.main()
