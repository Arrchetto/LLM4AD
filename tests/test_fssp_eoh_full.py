from __future__ import annotations

import csv
import json
import math
import tempfile
import unittest
from pathlib import Path


class FSSPCommonTest(unittest.TestCase):
    def test_parse_taillard_instance(self):
        from llm4ad.task.experiment.fssp_common.dataset import parse_instance

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.txt"
            path.write_text(
                "number of jobs, number of machines, initial seed, upper bound and lower bound :\n"
                "3 2 0 10 8\n"
                "processing times :\n"
                "1 2 3\n"
                "4 5 6\n",
                encoding="utf-8",
            )
            instance = parse_instance(path, "sample", "Taillard", "tai3_2")

        self.assertEqual(instance.instance_id, "sample")
        self.assertEqual(instance.n, 3)
        self.assertEqual(instance.m, 2)
        self.assertEqual(instance.matrix, ((1, 4), (2, 5), (3, 6)))

    def test_parse_rejects_wrong_row_length(self):
        from llm4ad.task.experiment.fssp_common.dataset import parse_instance

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.txt"
            path.write_text(
                "number of jobs, number of machines, initial seed, upper bound and lower bound :\n"
                "3 2 0 10 8\n"
                "processing times :\n"
                "1 2\n"
                "4 5 6\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "expected 3 processing times"):
                parse_instance(path, "bad")

    def test_parse_rejects_non_integer(self):
        from llm4ad.task.experiment.fssp_common.dataset import parse_instance

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.txt"
            path.write_text(
                "number of jobs, number of machines, initial seed, upper bound and lower bound :\n"
                "3 2 0 10 8\n"
                "processing times :\n"
                "1 2 x\n"
                "4 5 6\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "processing times must be integers"):
                parse_instance(path, "bad")

    def test_validate_complete_sequence(self):
        from llm4ad.task.experiment.fssp_common.dataset import FSSPInstance
        from llm4ad.task.experiment.fssp_common.evaluation_core import validate_sequence

        instance = FSSPInstance("sample", 3, 2, ((1, 4), (2, 5), (3, 6)), "Taillard", "tai3_2")
        normalized = validate_sequence(instance, [3, 1, 2])
        self.assertEqual(normalized, (3, 1, 2))

    def test_validate_rejects_duplicate_job(self):
        from llm4ad.task.experiment.fssp_common.dataset import FSSPInstance
        from llm4ad.task.experiment.fssp_common.evaluation_core import validate_sequence

        instance = FSSPInstance("sample", 3, 2, ((1, 4), (2, 5), (3, 6)), "Taillard", "tai3_2")
        with self.assertRaisesRegex(ValueError, "appears more than once"):
            validate_sequence(instance, [1, 2, 2])

    def test_validate_rejects_missing_job(self):
        from llm4ad.task.experiment.fssp_common.dataset import FSSPInstance
        from llm4ad.task.experiment.fssp_common.evaluation_core import validate_sequence

        instance = FSSPInstance("sample", 3, 2, ((1, 4), (2, 5), (3, 6)), "Taillard", "tai3_2")
        with self.assertRaisesRegex(ValueError, "does not match"):
            validate_sequence(instance, [1, 2])

    def test_validate_rejects_out_of_range(self):
        from llm4ad.task.experiment.fssp_common.dataset import FSSPInstance
        from llm4ad.task.experiment.fssp_common.evaluation_core import validate_sequence

        instance = FSSPInstance("sample", 3, 2, ((1, 4), (2, 5), (3, 6)), "Taillard", "tai3_2")
        with self.assertRaisesRegex(ValueError, "out of range"):
            validate_sequence(instance, [0, 1, 2])

    def test_validate_rejects_non_integer(self):
        from llm4ad.task.experiment.fssp_common.dataset import FSSPInstance
        from llm4ad.task.experiment.fssp_common.evaluation_core import validate_sequence

        instance = FSSPInstance("sample", 3, 2, ((1, 4), (2, 5), (3, 6)), "Taillard", "tai3_2")
        with self.assertRaisesRegex(ValueError, "sequence of integers"):
            validate_sequence(instance, [1, 2.5, 3])

    def test_validate_rejects_bool(self):
        from llm4ad.task.experiment.fssp_common.dataset import FSSPInstance
        from llm4ad.task.experiment.fssp_common.evaluation_core import validate_sequence

        instance = FSSPInstance("sample", 3, 2, ((1, 4), (2, 5), (3, 6)), "Taillard", "tai3_2")
        with self.assertRaisesRegex(ValueError, "sequence of integers"):
            validate_sequence(instance, [True, 2, 3])

    def test_compute_makespan_independent_of_candidate(self):
        from llm4ad.task.experiment.fssp_common.dataset import FSSPInstance
        from llm4ad.task.experiment.fssp_common.evaluation_core import (
            compute_makespan,
            validate_sequence,
        )

        instance = FSSPInstance("sample", 3, 2, ((1, 4), (2, 5), (3, 6)), "Taillard", "tai3_2")
        sequence = validate_sequence(instance, [1, 2, 3])
        # Manually compute makespan: job1 (1,4), job2 (2,5), job3 (3,6)
        # C[0] = [1,5]; C[1]=[3,10]; C[2]=[6,16]
        self.assertEqual(compute_makespan(instance, sequence), 16)

    def test_relative_gap_direction(self):
        from llm4ad.task.experiment.fssp_common.evaluation_core import relative_gap

        self.assertAlmostEqual(relative_gap(11, 10), 0.1)
        self.assertAlmostEqual(relative_gap(9, 10), -0.1)

    def test_macro_average_weights_subseries_equally(self):
        from llm4ad.task.experiment.fssp_common.evaluation_core import (
            macro_average_fitness,
        )

        fitness = macro_average_fitness(
            [("a", 0.0), ("a", 0.2), ("b", 0.4)]
        )
        self.assertAlmostEqual(fitness, -0.25)

    def test_committed_manifest_has_expected_split_counts(self):
        root = (
            Path(__file__).resolve().parents[1]
            / "llm4ad/task/experiment/fssp_common"
        )
        manifest = json.loads(
            (root / "split_manifest.json").read_text(encoding="utf-8")
        )
        entries = manifest["instances"]
        counts = {}
        paths = set()
        subseries_by_split: dict[str, set[str]] = {}
        for entry in entries:
            counts[entry["split"]] = counts.get(entry["split"], 0) + 1
            self.assertNotIn(entry["relative_path"], paths)
            paths.add(entry["relative_path"])
            subseries_by_split.setdefault(entry["split"], set()).add(entry["subseries"])

        self.assertEqual(
            counts,
            {
                "train": 30,
                "validation": 30,
                "test_cross_scale": 60,
            },
        )
        # Splits must not share any subseries.
        for split_a, set_a in subseries_by_split.items():
            for split_b, set_b in subseries_by_split.items():
                if split_a != split_b:
                    self.assertEqual(set_a & set_b, set())

    def test_every_manifest_instance_has_a_reference(self):
        root = (
            Path(__file__).resolve().parents[1]
            / "llm4ad/task/experiment/fssp_common"
        )
        manifest = json.loads(
            (root / "split_manifest.json").read_text(encoding="utf-8")
        )["instances"]
        with (root / "best_known.csv").open(encoding="utf-8", newline="") as handle:
            references = list(csv.DictReader(handle))
        self.assertEqual(
            {entry["instance_id"] for entry in manifest},
            {entry["instance_id"] for entry in references},
        )
        self.assertTrue(all(entry["source"] for entry in references))

    def test_training_references_are_best_known_or_optimal(self):
        root = (
            Path(__file__).resolve().parents[1]
            / "llm4ad/task/experiment/fssp_common"
        )
        manifest = json.loads(
            (root / "split_manifest.json").read_text(encoding="utf-8")
        )["instances"]
        with (root / "best_known.csv").open(encoding="utf-8", newline="") as handle:
            references = {
                row["instance_id"]: row["status"]
                for row in csv.DictReader(handle)
            }
        train_ids = {e["instance_id"] for e in manifest if e["split"] == "train"}
        self.assertTrue(
            all(references[iid] in {"OPTIMAL", "BEST_KNOWN"} for iid in train_ids)
        )


class FSSPEoHFullTest(unittest.TestCase):
    def test_task_is_exported_and_eoh_only(self):
        from llm4ad.task import FSSPEoHFullEvaluation

        self.assertEqual(FSSPEoHFullEvaluation.supported_methods, ("EoH",))

    def test_baseline_solver_returns_complete_solution(self):
        from llm4ad.task.experiment.fssp_common.dataset import FSSPInstance
        from llm4ad.task.experiment.fssp_common.evaluation_core import validate_sequence
        from llm4ad.task.experiment.fssp_eoh_full.template import solve

        instance = FSSPInstance("sample", 3, 2, ((1, 4), (2, 5), (3, 6)), "Taillard", "tai3_2")
        sequence = solve(
            instance.instance_id,
            instance.n,
            instance.m,
            [list(row) for row in instance.matrix],
        )
        validated = validate_sequence(instance, sequence)
        self.assertEqual(len(validated), instance.n)

    def test_evaluator_scores_complete_solver_on_custom_protocol(self):
        from llm4ad.task.experiment.fssp_eoh_full.evaluation import (
            FSSPEoHFullEvaluation,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            extracted = root / "extracted/taillard/tai3_2"
            extracted.mkdir(parents=True)
            (extracted / "tai3_2_00.txt").write_text(
                "number of jobs, number of machines, initial seed, upper bound and lower bound :\n"
                "3 2 0 16 16\n"
                "processing times :\n"
                "1 2 3\n"
                "4 5 6\n",
                encoding="utf-8",
            )
            manifest = root / "split_manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "instances": [
                            {
                                "instance_id": "tai3_2_00",
                                "relative_path": "taillard/tai3_2/tai3_2_00.txt",
                                "family": "Taillard",
                                "subseries": "tai3_2",
                                "size": 3,
                                "split": "train",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            references = root / "best_known.csv"
            references.write_text(
                "instance_id,best_known_value,status,source,checked_date\n"
                "tai3_2_00,16,OPTIMAL,test,2026-07-01\n",
                encoding="utf-8",
            )
            evaluation = FSSPEoHFullEvaluation(
                data_root=root,
                manifest_path=manifest,
                best_known_path=references,
                safe_evaluate=False,
            )

            def solver(instance_id, n, m, matrix):
                return [1, 2, 3]

            score = evaluation.evaluate(solver)
            self.assertEqual(score, 0.0)

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

    def test_baseline_smoke_test_on_all_training_instances(self):
        import time
        from llm4ad.task.experiment.fssp_eoh_full.evaluation import (
            FSSPEoHFullEvaluation,
        )
        from llm4ad.task.experiment.fssp_eoh_full.template import solve

        evaluation = FSSPEoHFullEvaluation(safe_evaluate=False)
        self.assertEqual(len(evaluation.instances), 30)
        start = time.time()
        score = evaluation.evaluate(solve)
        elapsed = time.time() - start
        self.assertTrue(math.isfinite(score))
        print(f"baseline fitness={score:.6f} elapsed={elapsed:.3f}s")

    def test_secure_evaluator_executes_template(self):
        from llm4ad.base.evaluate import SecureEvaluator
        from llm4ad.task.experiment.fssp_eoh_full.evaluation import (
            FSSPEoHFullEvaluation,
        )
        from llm4ad.task.experiment.fssp_eoh_full.template import template_program

        evaluator = FSSPEoHFullEvaluation(safe_evaluate=True, timeout_seconds=10)
        secure = SecureEvaluator(evaluator)
        score = secure.evaluate_program(template_program)
        self.assertTrue(isinstance(score, float) and math.isfinite(score))


if __name__ == "__main__":
    unittest.main()
