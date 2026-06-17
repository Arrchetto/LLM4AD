import unittest
import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

import numpy as np

from llm4ad.task import OrienteeringEvaluation as ExportedOrienteeringEvaluation
from llm4ad.task.optimization.orienteering_construct import OrienteeringEvaluation
from llm4ad.task.optimization.orienteering_construct.get_instance import GetData


def prize_density_priority(
    current_node: int,
    destination_node: int,
    candidate_nodes: np.ndarray,
    distance_matrix: np.ndarray,
    prizes: np.ndarray,
    remaining_budget: float,
) -> int:
    travel_cost = distance_matrix[current_node][candidate_nodes]
    return_cost = distance_matrix[candidate_nodes, destination_node]
    scores = prizes[candidate_nodes] / (travel_cost + return_cost + 1e-12)
    return int(candidate_nodes[np.argmax(scores)])


class OrienteeringConstructTest(unittest.TestCase):
    def test_task_package_exports_evaluation_for_dynamic_discovery(self):
        self.assertIs(ExportedOrienteeringEvaluation, OrienteeringEvaluation)

    def test_generate_instances_are_reproducible_and_well_formed(self):
        data_a = GetData(n_instance=3, problem_size=12, seed=7).generate_instances()
        data_b = GetData(n_instance=3, problem_size=12, seed=7).generate_instances()

        self.assertEqual(len(data_a), 3)
        self.assertEqual(len(data_b), 3)

        first_a = data_a[0]
        first_b = data_b[0]
        np.testing.assert_allclose(first_a["coordinates"], first_b["coordinates"])
        np.testing.assert_allclose(first_a["distance_matrix"], first_b["distance_matrix"])
        np.testing.assert_allclose(first_a["prizes"], first_b["prizes"])

        self.assertEqual(first_a["coordinates"].shape, (12, 2))
        self.assertEqual(first_a["distance_matrix"].shape, (12, 12))
        self.assertEqual(first_a["prizes"].shape, (12,))
        self.assertEqual(first_a["start_node"], 0)
        self.assertEqual(first_a["end_node"], 0)
        self.assertEqual(first_a["prizes"][0], 0.0)
        self.assertGreater(first_a["max_length"], 0.0)

    def test_evaluation_scores_a_constructive_priority_function(self):
        evaluation = OrienteeringEvaluation(
            timeout_seconds=5,
            n_instance=4,
            problem_size=15,
            max_length_ratio=0.45,
            seed=11,
        )

        score = evaluation.evaluate(prize_density_priority)

        self.assertIsInstance(score, float)
        self.assertGreater(score, 0.0)
        self.assertLessEqual(score, 14.0)


if __name__ == "__main__":
    unittest.main()
