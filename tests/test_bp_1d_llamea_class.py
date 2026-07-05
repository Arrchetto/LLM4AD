from __future__ import annotations

import ast
import inspect
import json
import math
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch


def _write_protocol(root: Path, instance_count: int = 1) -> tuple[Path, Path]:
    extracted = root / "extracted/family"
    extracted.mkdir(parents=True)
    entries = []
    reference_rows = []
    for index in range(instance_count):
        instance_id = f"a{index}"
        relative_path = f"family/{instance_id}.txt"
        (root / "extracted" / relative_path).write_text(
            "4\n10\n5\n5\n5\n5\n",
            encoding="utf-8",
        )
        entries.append(
            {
                "instance_id": instance_id,
                "relative_path": relative_path,
                "family": "family",
                "subseries": "series_a" if index % 2 == 0 else "series_b",
                "size": 4,
                "split": "train",
            }
        )
        reference_rows.append(
            f"{instance_id},2,OPTIMAL,test,2026-07-01\n"
        )
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps({"instances": entries}), encoding="utf-8")
    references = root / "best_known.csv"
    references.write_text(
        "instance_id,best_known_bins,status,source,checked_date\n"
        + "".join(reference_rows),
        encoding="utf-8",
    )
    return manifest, references


def _make_evaluation(
    root: Path,
    instance_count: int = 1,
    *,
    safe_evaluate: bool = False,
    timeout_seconds: int | float = 30,
):
    from llm4ad.task.experiment.bp_1d_llamea_class.evaluation import (
        BP1DLLaMEAClassEvaluation,
    )

    manifest, references = _write_protocol(root, instance_count)
    return BP1DLLaMEAClassEvaluation(
        data_root=root,
        manifest_path=manifest,
        best_known_path=references,
        safe_evaluate=safe_evaluate,
        timeout_seconds=timeout_seconds,
    )


class BP1DLLaMEAClassContractTest(unittest.TestCase):
    def test_candidate_metadata_is_exact(self):
        from llm4ad.task import BP1DLLaMEAClassEvaluation

        self.assertEqual(BP1DLLaMEAClassEvaluation.candidate_type, "class")
        self.assertEqual(
            BP1DLLaMEAClassEvaluation.candidate_name,
            "BinPackingOptimizer",
        )
        self.assertEqual(
            BP1DLLaMEAClassEvaluation.candidate_call_signature,
            ("instance",),
        )
        self.assertEqual(
            BP1DLLaMEAClassEvaluation.supported_methods,
            ("LLaMEA",),
        )

    def test_template_has_exactly_one_target_top_level_class(self):
        from llm4ad.task.experiment.bp_1d_llamea_class.template import (
            template_program,
        )

        module = ast.parse(template_program)
        target_classes = [
            node
            for node in module.body
            if isinstance(node, ast.ClassDef)
            and node.name == "BinPackingOptimizer"
        ]
        self.assertEqual(len(target_classes), 1)

    def test_template_constructor_and_call_signatures_are_exact(self):
        from llm4ad.task.experiment.bp_1d_llamea_class.template import (
            BinPackingOptimizer,
        )

        self.assertEqual(
            tuple(inspect.signature(BinPackingOptimizer).parameters),
            (),
        )
        self.assertEqual(
            tuple(inspect.signature(BinPackingOptimizer.__call__).parameters),
            ("self", "instance"),
        )

    def test_baseline_preserves_duplicate_item_identity(self):
        from llm4ad.task.experiment.bp_1d_common.dataset import BPInstance
        from llm4ad.task.experiment.bp_1d_common.evaluation_core import (
            validate_packing,
        )
        from llm4ad.task.experiment.bp_1d_llamea_class.template import (
            BinPackingOptimizer,
        )

        instance = BPInstance("a", 4, 10, (5, 5, 5, 5), "f", "s")
        solution = BinPackingOptimizer()(
            {
                "instance_id": "a",
                "bin_capacity": 10,
                "num_items": 4,
                "items": [5, 5, 5, 5],
            }
        )
        self.assertEqual(len(validate_packing(instance, solution)), 2)

    def test_description_specifies_complete_optimizer_and_index_contract(self):
        from llm4ad.task.experiment.bp_1d_llamea_class.template import (
            task_description,
        )

        description = task_description.lower()
        self.assertIn("complete optimizer class", description)
        self.assertIn("original 1-based item index", description)
        self.assertIn("do not use items.index", description)
        self.assertIn("different algorithm families", description)


class BP1DLLaMEAClassEvaluationTest(unittest.TestCase):
    def test_valid_optimizer_scores_zero_gap(self):
        with tempfile.TemporaryDirectory() as directory:
            evaluation = _make_evaluation(Path(directory))

            class ValidOptimizer:
                def __call__(self, instance):
                    return {"num_bins": 2, "bins": [[1, 2], [3, 4]]}

            self.assertEqual(evaluation.evaluate(ValidOptimizer), 0.0)

    def test_creates_fresh_optimizer_for_every_instance(self):
        with tempfile.TemporaryDirectory() as directory:
            evaluation = _make_evaluation(Path(directory), instance_count=3)

            class StatefulOptimizer:
                created = 0

                def __init__(self):
                    type(self).created += 1
                    self.used = False

                def __call__(self, instance):
                    if self.used:
                        raise RuntimeError("optimizer reused")
                    self.used = True
                    return {"num_bins": 2, "bins": [[1, 2], [3, 4]]}

            self.assertEqual(evaluation.evaluate(StatefulOptimizer), 0.0)
            self.assertEqual(StatefulOptimizer.created, 3)

    def test_passes_independent_instance_and_items_copies(self):
        with tempfile.TemporaryDirectory() as directory:
            evaluation = _make_evaluation(Path(directory), instance_count=3)

            class MutatingOptimizer:
                payloads = []
                item_lists = []
                first_values = []

                def __call__(self, instance):
                    type(self).payloads.append(instance)
                    type(self).item_lists.append(instance["items"])
                    type(self).first_values.append(tuple(instance["items"]))
                    instance["items"][0] = 999
                    instance["pollution"] = True
                    return {"num_bins": 2, "bins": [[1, 2], [3, 4]]}

            self.assertEqual(evaluation.evaluate(MutatingOptimizer), 0.0)
            self.assertEqual(MutatingOptimizer.first_values, [(5, 5, 5, 5)] * 3)
            self.assertEqual(len({id(value) for value in MutatingOptimizer.payloads}), 3)
            self.assertEqual(len({id(value) for value in MutatingOptimizer.item_lists}), 3)
            self.assertTrue(all(instance.items == (5, 5, 5, 5) for instance in evaluation.instances))

    def test_invalid_candidate_results_return_none(self):
        invalid_solutions = {
            "bins_dict": {"num_bins": 2, "bins": {0: [1, 2], 1: [3, 4]}},
            "bins_tuple": {"num_bins": 2, "bins": ([1, 2], [3, 4])},
            "inner_tuple": {"num_bins": 2, "bins": [(1, 2), [3, 4]]},
            "sizes_not_indices": {"num_bins": 2, "bins": [[5, 5], [5, 5]]},
            "duplicate_index": {"num_bins": 2, "bins": [[1, 1], [3, 4]]},
            "missing_index": {"num_bins": 2, "bins": [[1, 2], [3]]},
            "items_index_duplicate_mapping": {"num_bins": 2, "bins": [[1, 1], [1, 1]]},
            "empty_bin": {"num_bins": 3, "bins": [[1, 2], [], [3, 4]]},
            "capacity_exceeded": {"num_bins": 2, "bins": [[1, 2, 3], [4]]},
            "num_bins_mismatch": {"num_bins": 3, "bins": [[1, 2], [3, 4]]},
            "non_integer_index": {"num_bins": 2, "bins": [[1.0, 2], [3, 4]]},
            "out_of_range": {"num_bins": 2, "bins": [[1, 2], [3, 5]]},
        }
        for label, solution in invalid_solutions.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                evaluation = _make_evaluation(Path(directory))

                class InvalidOptimizer:
                    def __call__(self, instance):
                        return solution

                self.assertIsNone(evaluation.evaluate(InvalidOptimizer))

    def test_constructor_and_call_exceptions_return_none(self):
        class ConstructorFailure:
            def __init__(self):
                raise RuntimeError("constructor failed")

        class CallFailure:
            def __call__(self, instance):
                raise RuntimeError("call failed")

        for optimizer_class in (ConstructorFailure, CallFailure):
            with self.subTest(optimizer=optimizer_class.__name__), tempfile.TemporaryDirectory() as directory:
                evaluation = _make_evaluation(Path(directory))
                self.assertIsNone(evaluation.evaluate(optimizer_class))

    def test_non_finite_aggregate_fitness_returns_none(self):
        with tempfile.TemporaryDirectory() as directory:
            evaluation = _make_evaluation(Path(directory))

            class ValidOptimizer:
                def __call__(self, instance):
                    return {"num_bins": 2, "bins": [[1, 2], [3, 4]]}

            with patch(
                "llm4ad.task.experiment.bp_1d_llamea_class.evaluation."
                "macro_average_fitness",
                return_value=float("nan"),
            ):
                self.assertIsNone(evaluation.evaluate(ValidOptimizer))


class BP1DLLaMEAClassIntegrationTest(unittest.TestCase):
    def test_train_protocol_remains_thirty_instances_and_four_subseries(self):
        from llm4ad.task.experiment.bp_1d_common.dataset import load_manifest

        training = [entry for entry in load_manifest() if entry.split == "train"]
        self.assertEqual(len(training), 30)
        self.assertEqual(len({entry.subseries for entry in training}), 4)

    def test_baseline_is_feasible_on_every_training_instance(self):
        from llm4ad.task.experiment.bp_1d_common.evaluation_core import (
            validate_packing,
        )
        from llm4ad.task.experiment.bp_1d_llamea_class.evaluation import (
            BP1DLLaMEAClassEvaluation,
        )
        from llm4ad.task.experiment.bp_1d_llamea_class.template import (
            BinPackingOptimizer,
        )

        evaluation = BP1DLLaMEAClassEvaluation(safe_evaluate=False)
        for instance in evaluation.instances:
            solution = BinPackingOptimizer()(
                {
                    "instance_id": instance.instance_id,
                    "bin_capacity": instance.bin_capacity,
                    "num_items": instance.num_items,
                    "items": list(instance.items),
                }
            )
            validate_packing(instance, solution)
        self.assertEqual(len(evaluation.instances), 30)

    def test_eoh_and_llamea_baselines_have_identical_fitness(self):
        from llm4ad.task.experiment.bp_1d_eoh_full.evaluation import (
            BP1DEoHFullEvaluation,
        )
        from llm4ad.task.experiment.bp_1d_llamea_class.evaluation import (
            BP1DLLaMEAClassEvaluation,
        )
        from llm4ad.task.experiment.bp_1d_llamea_class.template import (
            BinPackingOptimizer,
        )

        def eoh_solver(instance_id, bin_capacity, num_items, items):
            return BinPackingOptimizer()(
                {
                    "instance_id": instance_id,
                    "bin_capacity": bin_capacity,
                    "num_items": num_items,
                    "items": list(items),
                }
            )

        eoh_fitness = BP1DEoHFullEvaluation(safe_evaluate=False).evaluate(
            eoh_solver
        )
        llamea_fitness = BP1DLLaMEAClassEvaluation(
            safe_evaluate=False
        ).evaluate(BinPackingOptimizer)
        self.assertIsNotNone(eoh_fitness)
        self.assertIsNotNone(llamea_fitness)
        self.assertAlmostEqual(eoh_fitness, llamea_fitness, places=12)

    def test_gui_discovers_task_but_not_common_support_package(self):
        from GUI.run_gui import discover_task_directories

        experiment_root = (
            Path(__file__).resolve().parents[1]
            / "llm4ad/task/experiment"
        )
        tasks = discover_task_directories(experiment_root)
        self.assertIn("bp_1d_llamea_class", tasks)
        self.assertNotIn("bp_1d_common", tasks)

    def test_method_compatibility_accepts_llamea_and_rejects_eoh(self):
        from llm4ad.gui import _validate_method_evaluation_compatibility

        with tempfile.TemporaryDirectory() as directory:
            evaluation = _make_evaluation(Path(directory))
            _validate_method_evaluation_compatibility("LLaMEA", evaluation)
            with self.assertRaisesRegex(ValueError, "BP1DLLaMEAClassEvaluation.*EoH"):
                _validate_method_evaluation_compatibility("EoH", evaluation)

    def test_standard_llamea_adapter_executes_class_template(self):
        from llamea import Solution

        from llm4ad.method.llamea.evaluation import generate_evaluator
        from llm4ad.task.experiment.bp_1d_llamea_class.template import (
            template_program,
        )

        with tempfile.TemporaryDirectory() as directory:
            evaluation = _make_evaluation(Path(directory))
            solution = Solution(
                name="BinPackingOptimizer",
                code=template_program,
            )
            evaluated = generate_evaluator(evaluation)(solution)

        self.assertTrue(math.isfinite(evaluated.fitness))
        self.assertEqual(evaluated.fitness, 0.0)

    def test_standard_adapter_enforces_total_candidate_timeout(self):
        from llamea import Solution

        from llm4ad.method.llamea.evaluation import generate_evaluator

        code = (
            "import time\n\n"
            "class BinPackingOptimizer:\n"
            "    def __init__(self):\n"
            "        pass\n\n"
            "    def __call__(self, instance):\n"
            "        time.sleep(0.3)\n"
            "        return {'num_bins': 2, 'bins': [[1, 2], [3, 4]]}\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            evaluation = _make_evaluation(
                Path(directory),
                safe_evaluate=True,
                timeout_seconds=0.05,
            )
            started = time.monotonic()
            evaluated = generate_evaluator(evaluation)(
                Solution(name="BinPackingOptimizer", code=code)
            )
            elapsed = time.monotonic() - started

        self.assertEqual(evaluated.fitness, float("-inf"))
        self.assertIn("timed out", evaluated.feedback.lower())
        self.assertLess(elapsed, 0.25)


if __name__ == "__main__":
    unittest.main()
