from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path


class FSSPLLAMEAClassTest(unittest.TestCase):
    def test_task_is_exported_and_llamea_only(self):
        from llm4ad.task import FSSPLLAMEAClassEvaluation

        self.assertEqual(FSSPLLAMEAClassEvaluation.supported_methods, ("LLaMEA",))
        self.assertEqual(FSSPLLAMEAClassEvaluation.candidate_type, "class")
        self.assertEqual(FSSPLLAMEAClassEvaluation.candidate_name, "FlowShopOptimizer")
        self.assertEqual(FSSPLLAMEAClassEvaluation.candidate_call_signature, ("instance",))

    def test_baseline_class_returns_complete_solution(self):
        from llm4ad.task.experiment.fssp_common.dataset import FSSPInstance
        from llm4ad.task.experiment.fssp_common.evaluation_core import validate_sequence
        from llm4ad.task.experiment.fssp_llamea_class.template import FlowShopOptimizer

        instance = FSSPInstance("sample", 3, 2, ((1, 4), (2, 5), (3, 6)), "Taillard", "tai3_2")
        optimizer = FlowShopOptimizer()
        sequence = optimizer(
            {
                "instance_id": instance.instance_id,
                "n": instance.n,
                "m": instance.m,
                "matrix": [list(row) for row in instance.matrix],
                "family": instance.family,
                "subseries": instance.subseries,
            }
        )
        validated = validate_sequence(instance, sequence)
        self.assertEqual(len(validated), instance.n)
        self.assertEqual(sorted(validated), [1, 2, 3])

    def _make_evaluator(self, root: Path):
        from llm4ad.task.experiment.fssp_llamea_class.evaluation import (
            FSSPLLAMEAClassEvaluation,
        )

        return FSSPLLAMEAClassEvaluation(
            data_root=root,
            manifest_path=root / "split_manifest.json",
            best_known_path=root / "best_known.csv",
            safe_evaluate=False,
        )

    def _build_custom_protocol(self, directory: str):
        root = Path(directory)
        extracted = root / "extracted/taillard/tai3_2"
        extracted.mkdir(parents=True)
        instance_text = (
            "number of jobs, number of machines, initial seed, upper bound and lower bound :\n"
            "3 2 0 16 16\n"
            "processing times :\n"
            "1 2 3\n"
            "4 5 6\n"
        )
        (extracted / "tai3_2_00.txt").write_text(instance_text, encoding="utf-8")
        (extracted / "tai3_2_01.txt").write_text(instance_text, encoding="utf-8")
        manifest = root / "split_manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "instances": [
                        {
                            "instance_id": "tai3_2_00",
                            "relative_path": "taillard/tai3_2/tai3_2_00.txt",
                            "family": "Taillard",
                            "subseries": "tai3_2_a",
                            "size": 3,
                            "split": "train",
                        },
                        {
                            "instance_id": "tai3_2_01",
                            "relative_path": "taillard/tai3_2/tai3_2_01.txt",
                            "family": "Taillard",
                            "subseries": "tai3_2_b",
                            "size": 3,
                            "split": "train",
                        },
                    ]
                }
            ),
            encoding="utf-8",
        )
        references = root / "best_known.csv"
        references.write_text(
            "instance_id,best_known_value,status,source,checked_date\n"
            "tai3_2_00,16,OPTIMAL,test,2026-07-01\n"
            "tai3_2_01,20,OPTIMAL,test,2026-07-01\n",
            encoding="utf-8",
        )

    def test_evaluator_rejects_missing_job(self):
        from llm4ad.task.experiment.fssp_llamea_class.evaluation import (
            FSSPLLAMEAClassEvaluation,
        )

        with tempfile.TemporaryDirectory() as directory:
            self._build_custom_protocol(directory)
            evaluation = self._make_evaluator(Path(directory))

            class BadOptimizer:
                def __call__(self, instance: dict) -> list[int]:
                    return [1, 2]

            self.assertIsNone(evaluation.evaluate(BadOptimizer))

    def test_evaluator_rejects_duplicate_job(self):
        from llm4ad.task.experiment.fssp_llamea_class.evaluation import (
            FSSPLLAMEAClassEvaluation,
        )

        with tempfile.TemporaryDirectory() as directory:
            self._build_custom_protocol(directory)
            evaluation = self._make_evaluator(Path(directory))

            class BadOptimizer:
                def __call__(self, instance: dict) -> list[int]:
                    return [1, 2, 2]

            self.assertIsNone(evaluation.evaluate(BadOptimizer))

    def test_evaluator_rejects_out_of_range(self):
        from llm4ad.task.experiment.fssp_llamea_class.evaluation import (
            FSSPLLAMEAClassEvaluation,
        )

        with tempfile.TemporaryDirectory() as directory:
            self._build_custom_protocol(directory)
            evaluation = self._make_evaluator(Path(directory))

            class BadOptimizer:
                def __call__(self, instance: dict) -> list[int]:
                    return [0, 1, 2]

            self.assertIsNone(evaluation.evaluate(BadOptimizer))

    def test_evaluator_rejects_non_integer(self):
        from llm4ad.task.experiment.fssp_llamea_class.evaluation import (
            FSSPLLAMEAClassEvaluation,
        )

        with tempfile.TemporaryDirectory() as directory:
            self._build_custom_protocol(directory)
            evaluation = self._make_evaluator(Path(directory))

            class BadOptimizer:
                def __call__(self, instance: dict) -> list[int]:
                    return [1, 2.5, 3]

            self.assertIsNone(evaluation.evaluate(BadOptimizer))

    def test_evaluator_rejects_bool(self):
        from llm4ad.task.experiment.fssp_llamea_class.evaluation import (
            FSSPLLAMEAClassEvaluation,
        )

        with tempfile.TemporaryDirectory() as directory:
            self._build_custom_protocol(directory)
            evaluation = self._make_evaluator(Path(directory))

            class BadOptimizer:
                def __call__(self, instance: dict) -> list[int]:
                    return [True, 2, 3]

            self.assertIsNone(evaluation.evaluate(BadOptimizer))

    def test_evaluator_scores_complete_solver_on_custom_protocol(self):
        from llm4ad.task.experiment.fssp_common.evaluation_core import (
            compute_makespan,
            macro_average_fitness,
            relative_gap,
            validate_sequence,
        )
        from llm4ad.task.experiment.fssp_llamea_class.evaluation import (
            FSSPLLAMEAClassEvaluation,
        )

        with tempfile.TemporaryDirectory() as directory:
            self._build_custom_protocol(directory)
            evaluation = self._make_evaluator(Path(directory))

            class FixedSequence:
                def __call__(self, instance: dict) -> list[int]:
                    return [1, 2, 3]

            score = evaluation.evaluate(FixedSequence)

            # Recompute expected score independently to verify evaluator logic.
            expected_gaps = []
            for instance in evaluation.instances:
                validated = validate_sequence(instance, [1, 2, 3])
                makespan = compute_makespan(instance, validated)
                reference = evaluation.references[instance.instance_id].makespan
                expected_gaps.append(
                    (instance.subseries, relative_gap(makespan, reference))
                )
            expected = macro_average_fitness(expected_gaps)

            self.assertIsNotNone(score)
            self.assertAlmostEqual(score, expected)
            # Instance 0: gap = (16 - 16) / 16 = 0
            # Instance 1: gap = (16 - 20) / 20 = -0.2
            # Macro-average gap = -0.1, fitness = 0.1
            self.assertAlmostEqual(score, 0.1)

    def test_blank_data_root_uses_default_path(self):
        from llm4ad.task.experiment.fssp_llamea_class.evaluation import (
            FSSPLLAMEAClassEvaluation,
        )

        evaluation = FSSPLLAMEAClassEvaluation(data_root="", safe_evaluate=False)
        self.assertEqual(len(evaluation.instances), 30)

    def test_baseline_smoke_test_on_all_training_instances(self):
        import time

        from llm4ad.task.experiment.fssp_llamea_class.evaluation import (
            FSSPLLAMEAClassEvaluation,
        )
        from llm4ad.task.experiment.fssp_llamea_class.template import (
            FlowShopOptimizer,
        )

        evaluation = FSSPLLAMEAClassEvaluation(safe_evaluate=False)
        self.assertEqual(len(evaluation.instances), 30)
        start = time.time()
        score = evaluation.evaluate(FlowShopOptimizer)
        elapsed = time.time() - start
        self.assertTrue(math.isfinite(score))
        print(f"baseline fitness={score:.6f} elapsed={elapsed:.3f}s")

    def test_secure_evaluator_executes_template(self):
        from llm4ad.base.evaluate import SecureEvaluator
        from llm4ad.task.experiment.fssp_llamea_class.evaluation import (
            FSSPLLAMEAClassEvaluation,
        )
        from llm4ad.task.experiment.fssp_llamea_class.template import template_program

        evaluator = FSSPLLAMEAClassEvaluation(safe_evaluate=True, timeout_seconds=10)
        secure = SecureEvaluator(evaluator)
        score = secure.evaluate_program(template_program)
        self.assertTrue(isinstance(score, float) and math.isfinite(score))

    def test_gui_discovery_hides_support_packages(self):
        from GUI.run_gui import discover_task_directories

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "runnable").mkdir()
            (root / "runnable/paras.yaml").write_text("name: X\n", encoding="utf-8")
            (root / "common").mkdir()
            self.assertEqual(discover_task_directories(root), ["runnable"])

    def test_gui_blank_optional_path_becomes_none(self):
        from GUI.run_gui import _convert_parameter_value

        self.assertIsNone(_convert_parameter_value("", "<class 'NoneType'>"))


if __name__ == "__main__":
    unittest.main()
