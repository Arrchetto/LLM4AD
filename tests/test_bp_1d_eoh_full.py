from __future__ import annotations

import json
import csv
import math
import ast
import tempfile
import unittest
from pathlib import Path


class BP1DCommonTest(unittest.TestCase):
    def test_common_reference_loader_parses_valid_rows(self):
        from llm4ad.task.experiment.bp_1d_common.references import (
            ReferenceValue,
            load_references,
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "best_known.csv"
            path.write_text(
                "instance_id,best_known_bins,status,source,checked_date\n"
                "a,2,optimal,test,2026-07-01\n",
                encoding="utf-8",
            )
            references = load_references(path)

        self.assertEqual(
            references,
            {"a": ReferenceValue(bins=2, status="OPTIMAL", source="test")},
        )

    def test_common_reference_loader_rejects_duplicate_and_invalid_status(self):
        from llm4ad.task.experiment.bp_1d_common.references import load_references

        cases = {
            "duplicate": (
                "a,2,OPTIMAL,test,2026-07-01\n"
                "a,2,OPTIMAL,test,2026-07-01\n",
                "duplicate best-known row",
            ),
            "status": (
                "a,2,UNKNOWN,test,2026-07-01\n",
                "invalid reference status",
            ),
        }
        for label, (rows, message) in cases.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "best_known.csv"
                path.write_text(
                    "instance_id,best_known_bins,status,source,checked_date\n" + rows,
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(ValueError, message):
                    load_references(path)

    def test_common_reference_coverage_enforces_training_protocol(self):
        from llm4ad.task.experiment.bp_1d_common.dataset import BPInstance
        from llm4ad.task.experiment.bp_1d_common.references import (
            ReferenceValue,
            validate_reference_coverage,
        )

        instances = (BPInstance("a", 2, 10, (5, 5), "f", "s"),)
        with self.assertRaisesRegex(ValueError, "missing best-known metadata"):
            validate_reference_coverage(instances, {}, "train")
        with self.assertRaisesRegex(ValueError, "sourced OPTIMAL or BEST_KNOWN"):
            validate_reference_coverage(
                instances,
                {"a": ReferenceValue(2, "UNVERIFIED", "")},
                "train",
            )

    def test_parse_bpplib_instance(self):
        from llm4ad.task.experiment.bp_1d_common.dataset import parse_instance

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.txt"
            path.write_text("4\n10\n2\n3\n4\n5\n", encoding="utf-8")
            instance = parse_instance(path, "sample")

        self.assertEqual(instance.instance_id, "sample")
        self.assertEqual(instance.num_items, 4)
        self.assertEqual(instance.bin_capacity, 10)
        self.assertEqual(instance.items, (2, 3, 4, 5))

    def test_parse_rejects_wrong_item_count(self):
        from llm4ad.task.experiment.bp_1d_common.dataset import parse_instance

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.txt"
            path.write_text("3\n10\n2\n3\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "expected 3 item sizes"):
                parse_instance(path, "bad")

    def test_validate_complete_packing(self):
        from llm4ad.task.experiment.bp_1d_common.dataset import BPInstance
        from llm4ad.task.experiment.bp_1d_common.evaluation_core import (
            validate_packing,
        )

        instance = BPInstance("sample", 4, 10, (2, 3, 4, 5), "family", "series")
        bins = validate_packing(
            instance,
            {"num_bins": 2, "bins": [[1, 2, 3], [4]]},
        )
        self.assertEqual(bins, ((1, 2, 3), (4,)))

    def test_validate_rejects_duplicate_item(self):
        from llm4ad.task.experiment.bp_1d_common.dataset import BPInstance
        from llm4ad.task.experiment.bp_1d_common.evaluation_core import (
            validate_packing,
        )

        instance = BPInstance("sample", 3, 10, (2, 3, 4), "family", "series")
        with self.assertRaisesRegex(ValueError, "exactly once"):
            validate_packing(
                instance,
                {"num_bins": 2, "bins": [[1, 2], [2, 3]]},
            )

    def test_macro_average_weights_series_equally(self):
        from llm4ad.task.experiment.bp_1d_common.evaluation_core import (
            macro_average_fitness,
        )

        fitness = macro_average_fitness(
            [("a", 0.0), ("a", 0.2), ("b", 0.4)]
        )
        self.assertAlmostEqual(fitness, -0.25)

    def test_committed_manifest_has_expected_split_counts(self):
        root = (
            Path(__file__).resolve().parents[1]
            / "llm4ad/task/experiment/bp_1d_common"
        )
        manifest = json.loads(
            (root / "split_manifest.json").read_text(encoding="utf-8")
        )
        entries = manifest["instances"]
        counts = {}
        paths = set()
        for entry in entries:
            counts[entry["split"]] = counts.get(entry["split"], 0) + 1
            self.assertNotIn(entry["relative_path"], paths)
            paths.add(entry["relative_path"])

        self.assertEqual(
            counts,
            {
                "train": 30,
                "validation": 30,
                "test_cross_family": 245,
                "test_cross_scale": 36,
            },
        )

    def test_every_manifest_instance_has_an_optimal_reference(self):
        root = (
            Path(__file__).resolve().parents[1]
            / "llm4ad/task/experiment/bp_1d_common"
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
        self.assertTrue(all(entry["status"] == "OPTIMAL" for entry in references))
        self.assertTrue(all(entry["source"] for entry in references))


class BP1DEoHFullTest(unittest.TestCase):
    def test_candidate_contract_prevents_common_indexing_failures(self):
        from llm4ad.task.experiment.bp_1d_eoh_full.template import (
            task_description,
            template_program,
        )

        module = ast.parse(template_program)
        function = next(node for node in module.body if isinstance(node, ast.FunctionDef))
        contract = ast.get_docstring(function)
        combined = f"{contract}\n{task_description}".lower()

        self.assertIn("list of lists", combined)
        self.assertIn("original 1-based", combined)
        self.assertIn("duplicate sizes", combined)
        self.assertIn("do not use items.index", combined)
        self.assertIn("(index + 1, items[index])", combined)

    def test_task_is_exported_and_eoh_only(self):
        from llm4ad.task import BP1DEoHFullEvaluation

        self.assertEqual(BP1DEoHFullEvaluation.supported_methods, ("EoH",))

    def test_baseline_solver_returns_complete_solution(self):
        from llm4ad.task.experiment.bp_1d_common.dataset import BPInstance
        from llm4ad.task.experiment.bp_1d_common.evaluation_core import (
            validate_packing,
        )
        from llm4ad.task.experiment.bp_1d_eoh_full.template import solve

        instance = BPInstance("sample", 4, 10, (2, 3, 4, 5), "family", "series")
        solution = solve(
            instance.instance_id,
            instance.bin_capacity,
            instance.num_items,
            list(instance.items),
        )
        validate_packing(instance, solution)

    def test_evaluator_scores_complete_solver_on_custom_protocol(self):
        from llm4ad.task.experiment.bp_1d_eoh_full.evaluation import (
            BP1DEoHFullEvaluation,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "extracted/family").mkdir(parents=True)
            (root / "extracted/family/a.txt").write_text(
                "4\n10\n5\n5\n5\n5\n", encoding="utf-8"
            )
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "instances": [
                            {
                                "instance_id": "a",
                                "relative_path": "family/a.txt",
                                "family": "family",
                                "subseries": "series",
                                "size": 4,
                                "split": "train",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            references = root / "best_known.csv"
            references.write_text(
                "instance_id,best_known_bins,status,source,checked_date\n"
                "a,2,OPTIMAL,test,2026-07-01\n",
                encoding="utf-8",
            )
            evaluation = BP1DEoHFullEvaluation(
                data_root=root,
                manifest_path=manifest,
                best_known_path=references,
                safe_evaluate=False,
            )

            def solver(instance_id, bin_capacity, num_items, items):
                return {"num_bins": 2, "bins": [[1, 2], [3, 4]]}

            self.assertEqual(evaluation.evaluate(solver), 0.0)

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
        from llm4ad.task.experiment.bp_1d_eoh_full.evaluation import (
            BP1DEoHFullEvaluation,
        )
        from llm4ad.task.experiment.bp_1d_eoh_full.template import solve

        evaluation = BP1DEoHFullEvaluation(safe_evaluate=False)
        score = evaluation.evaluate(solve)
        self.assertEqual(len(evaluation.instances), 30)
        self.assertTrue(math.isfinite(score))


if __name__ == "__main__":
    unittest.main()
