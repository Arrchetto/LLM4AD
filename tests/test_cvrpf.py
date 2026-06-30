from __future__ import annotations

import ast
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import yaml

import llm4ad.task as task_module
from llm4ad.base import SecureEvaluator, TextFunctionProgramConverter
from llm4ad.gui import _validate_method_evaluation_compatibility
from llm4ad.task.optimization.cvrpf import CVRPFEvaluation
from llm4ad.task.optimization.cvrpf.cvrplib import (
    build_euc_2d_distance_matrix,
    load_cvrplib_sets,
    load_instance_pair,
    natural_key,
    parse_vrp,
    route_set_cost,
    validate_routes,
)
from llm4ad.task.optimization.cvrpf.template import task_description, template_program


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT.parent / "data" / "benchmarks" / "cvrp" / "extracted"
A_N32_VRP = DATA_ROOT / "A" / "A" / "A-n32-k5.vrp"
A_N32_SOL = DATA_ROOT / "A" / "A" / "A-n32-k5.sol"
TASK_DIR = REPO_ROOT / "llm4ad" / "task" / "optimization" / "cvrpf"
SOURCE_EN = REPO_ROOT.parent / "data" / "benchmarks" / "cvrp" / "SOURCE.md"
SOURCE_ZH = REPO_ROOT.parent / "data" / "benchmarks" / "cvrp" / "SOURCE.zh.md"


def write_minimal_vrp(path: Path, *, edge_weight_type: str = "EUC_2D", depot: int = 1) -> None:
    path.write_text(
        "\n".join(
            (
                "NAME : Tiny-n3-k2",
                "COMMENT : parser fixture",
                "TYPE : CVRP",
                "DIMENSION : 3",
                f"EDGE_WEIGHT_TYPE : {edge_weight_type}",
                "CAPACITY : 5",
                "NODE_COORD_SECTION",
                "1 0 0",
                "2 3 0",
                "3 0 4",
                "DEMAND_SECTION",
                "1 0",
                "2 2",
                "3 3",
                "DEPOT_SECTION",
                str(depot),
                "-1",
                "EOF",
            )
        )
        + "\n",
        encoding="utf-8",
    )


class CVRPLIBParserTests(unittest.TestCase):
    def test_natural_sort_orders_numeric_name_components(self):
        names = ["A-n80-k10", "A-n9-k2", "A-n32-k5", "A-n100-k10"]
        self.assertEqual(
            sorted(names, key=natural_key),
            ["A-n9-k2", "A-n32-k5", "A-n80-k10", "A-n100-k10"],
        )

    def test_euc_2d_uses_tsplib_nearest_integer_rule(self):
        coordinates = np.array([[0.0, 0.0], [1.0, 1.0], [3.0, 4.0]])
        matrix = build_euc_2d_distance_matrix(coordinates)
        np.testing.assert_array_equal(
            matrix,
            np.array([[0, 1, 5], [1, 0, 4], [5, 4, 0]], dtype=np.int64),
        )

    def test_a_and_b_sets_have_exact_expected_counts(self):
        a_instances = load_cvrplib_sets(DATA_ROOT, ("A",), {"A": 27})
        b_instances = load_cvrplib_sets(DATA_ROOT, ("B",), {"B": 23})
        self.assertEqual(len(a_instances), 27)
        self.assertEqual(len(b_instances), 23)
        self.assertTrue(all(instance.dataset == "A" for instance in a_instances))
        self.assertTrue(all(instance.dataset == "B" for instance in b_instances))

    def test_combined_loading_is_deterministic_and_naturally_sorted(self):
        first = load_cvrplib_sets(DATA_ROOT, ("A", "B"), {"A": 27, "B": 23})
        second = load_cvrplib_sets(DATA_ROOT, ("A", "B"), {"A": 27, "B": 23})
        first_names = [instance.name for instance in first]
        self.assertEqual(len(first), 50)
        self.assertEqual(first_names, sorted(first_names, key=natural_key))
        self.assertEqual(first_names, [instance.name for instance in second])

    def test_a_n32_k5_metadata_and_solution_are_parsed(self):
        instance = load_instance_pair(A_N32_VRP, A_N32_SOL, dataset="A")
        self.assertEqual(instance.name, "A-n32-k5")
        self.assertEqual(instance.dimension, 32)
        self.assertEqual(instance.capacity, 100)
        self.assertEqual(instance.max_vehicles, 5)
        self.assertEqual(instance.best_known_cost, 784.0)
        self.assertEqual(len(instance.best_known_routes), 5)
        self.assertTrue(all(route[0] == route[-1] == 0 for route in instance.best_known_routes))

    def test_loaded_training_arrays_are_read_only(self):
        instance = load_instance_pair(A_N32_VRP, A_N32_SOL, dataset="A")
        for array in (instance.coordinates, instance.demands, instance.distance_matrix):
            with self.subTest(shape=array.shape):
                self.assertFalse(array.flags.writeable)
                with self.assertRaises(ValueError):
                    array.flat[0] = 999

    def test_solution_declared_cost_must_match_recomputed_routes(self):
        with tempfile.TemporaryDirectory() as tmp:
            sol_path = Path(tmp) / A_N32_SOL.name
            sol_path.write_text(
                A_N32_SOL.read_text(encoding="utf-8").replace("Cost 784", "Cost 785"),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Cost.*recomputed"):
                load_instance_pair(A_N32_VRP, sol_path, dataset="A")

    def test_instance_and_solution_stems_must_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            sol_path = Path(tmp) / "different-name.sol"
            sol_path.write_text(A_N32_SOL.read_text(encoding="utf-8"), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not match"):
                load_instance_pair(A_N32_VRP, sol_path, dataset="A")

    def test_unknown_distance_type_fails_explicitly(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "Tiny-n3-k2.vrp"
            write_minimal_vrp(path, edge_weight_type="GEO")
            with self.assertRaisesRegex(ValueError, "EDGE_WEIGHT_TYPE"):
                parse_vrp(path, dataset="A")

    def test_invalid_depot_fails_explicitly(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "Tiny-n3-k2.vrp"
            write_minimal_vrp(path, depot=4)
            with self.assertRaisesRegex(ValueError, "depot"):
                parse_vrp(path, dataset="A")

    def test_missing_required_field_fails_explicitly(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "Tiny-n3-k2.vrp"
            write_minimal_vrp(path)
            path.write_text(
                path.read_text(encoding="utf-8").replace("CAPACITY : 5\n", ""),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "CAPACITY"):
                parse_vrp(path, dataset="A")

    def test_duplicate_node_identifier_fails_explicitly(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "Tiny-n3-k2.vrp"
            write_minimal_vrp(path)
            path.write_text(
                path.read_text(encoding="utf-8").replace("3 0 4", "2 0 4"),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicate"):
                parse_vrp(path, dataset="A")

    def test_missing_solution_pair_fails_without_skipping(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            set_dir = root / "A"
            set_dir.mkdir()
            write_minimal_vrp(set_dir / "Tiny-n3-k2.vrp")
            with self.assertRaisesRegex(ValueError, "pair"):
                load_cvrplib_sets(root, ("A",), {"A": 1})


class StrictRouteScoringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.instance = load_instance_pair(A_N32_VRP, A_N32_SOL, dataset="A")

    def test_known_solution_recomputes_exact_bks(self):
        self.assertEqual(
            route_set_cost(self.instance, self.instance.best_known_routes),
            784.0,
        )

    def test_valid_routes_are_preserved_without_repair(self):
        normalized = validate_routes(self.instance, self.instance.best_known_routes)
        self.assertEqual(normalized, self.instance.best_known_routes)

    def test_empty_output_is_invalid(self):
        with self.assertRaises(ValueError):
            validate_routes(self.instance, [])

    def test_duplicate_customer_is_invalid(self):
        routes = [list(route) for route in self.instance.best_known_routes]
        routes[0].insert(-1, routes[0][1])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_routes(self.instance, routes)

    def test_missing_customer_is_invalid(self):
        routes = [list(route) for route in self.instance.best_known_routes]
        routes[0].pop(1)
        with self.assertRaisesRegex(ValueError, "missing"):
            validate_routes(self.instance, routes)

    def test_out_of_range_customer_is_invalid(self):
        routes = [list(route) for route in self.instance.best_known_routes]
        routes[0][1] = self.instance.dimension
        with self.assertRaisesRegex(ValueError, "outside"):
            validate_routes(self.instance, routes)

    def test_missing_start_or_end_depot_is_invalid(self):
        routes = [list(route) for route in self.instance.best_known_routes]
        missing_start = [list(route) for route in routes]
        missing_start[0] = missing_start[0][1:]
        missing_end = [list(route) for route in routes]
        missing_end[0] = missing_end[0][:-1]
        for invalid in (missing_start, missing_end):
            with self.subTest(route=invalid[0]), self.assertRaisesRegex(ValueError, "depot"):
                validate_routes(self.instance, invalid)

    def test_interior_depot_is_invalid(self):
        routes = [list(route) for route in self.instance.best_known_routes]
        routes[0].insert(2, 0)
        with self.assertRaisesRegex(ValueError, "interior depot"):
            validate_routes(self.instance, routes)

    def test_over_capacity_route_is_invalid(self):
        all_customers = [node for route in self.instance.best_known_routes for node in route[1:-1]]
        with self.assertRaisesRegex(ValueError, "capacity"):
            validate_routes(self.instance, [[0, *all_customers, 0]])

    def test_excess_vehicle_count_is_invalid(self):
        routes = [[0, customer, 0] for customer in range(1, self.instance.dimension)]
        with self.assertRaisesRegex(ValueError, "vehicles"):
            validate_routes(self.instance, routes)

    def test_non_integer_nodes_are_invalid_without_coercion(self):
        for invalid_node in (True, 1.0, np.float64(1.0), "1"):
            routes = [list(route) for route in self.instance.best_known_routes]
            routes[0][1] = invalid_node
            with self.subTest(node=invalid_node), self.assertRaisesRegex(ValueError, "integer"):
                validate_routes(self.instance, routes)

    def test_non_route_objects_are_invalid(self):
        invalid_outputs = (None, "routes", b"routes", 1, [0, 1, 0], [[0, 1, object(), 0]])
        for invalid in invalid_outputs:
            with self.subTest(output=invalid), self.assertRaises(ValueError):
                validate_routes(self.instance, invalid)

    def test_nonfinite_route_cost_is_invalid(self):
        matrix = self.instance.distance_matrix.astype(np.float64)
        matrix[0, self.instance.best_known_routes[0][1]] = np.nan
        invalid_instance = replace(self.instance, distance_matrix=matrix)
        with self.assertRaisesRegex(ValueError, "finite"):
            route_set_cost(invalid_instance, invalid_instance.best_known_routes)


def compile_template_solve():
    namespace: dict[str, object] = {}
    exec(template_program, namespace)
    return namespace["solve"]


class CVRPFEvaluatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evaluation = CVRPFEvaluation(data_root=DATA_ROOT, safe_evaluate=False)
        cls.solve = staticmethod(compile_template_solve())

    def test_template_has_exactly_one_top_level_function(self):
        tree = ast.parse(template_program)
        functions = [
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        self.assertEqual([function.name for function in functions], ["solve"])
        converted = TextFunctionProgramConverter.text_to_function(template_program)
        self.assertEqual(converted.name, "solve")

    def test_template_function_documents_runtime_routes_and_fitness_direction(self):
        template_text = " ".join(template_program.lower().split())
        self.assertLessEqual(len(template_program.splitlines()), 38)
        for required in (
            "30-second",
            "every customer",
            "start and end at depot 0",
            "max_vehicles",
            "negative mean",
            "eoh maximizes",
        ):
            with self.subTest(required=required):
                self.assertIn(required, template_text)

    def test_task_description_states_complete_strict_contract(self):
        description = " ".join(task_description.lower().split())
        self.assertLessEqual(len(task_description), 1300)
        for required in (
            "50 training instances",
            "27",
            "23",
            "distance_matrix",
            "demands",
            "vehicle_capacity",
            "max_vehicles",
            "node 0",
            "exactly once",
            "start and end",
            "strict timeout",
            "30 seconds",
            "all algorithm logic",
            "negative mean",
            "eoh maximizes",
        ):
            with self.subTest(required=required):
                self.assertIn(required, description)

    def test_task_description_requires_feasible_grouping_before_route_ordering(self):
        description = " ".join(task_description.lower().split())
        for required in (
            "assign every customer",
            "capacity-feasible groups",
            "reassignment",
            "only then optimize",
            "never stop with unassigned customers",
        ):
            with self.subTest(required=required):
                self.assertIn(required, description)

    def test_evaluator_loads_exactly_a_and_b_in_deterministic_order(self):
        instances = self.evaluation.training_instances
        self.assertEqual(len(instances), 50)
        self.assertEqual({instance.dataset for instance in instances}, {"A", "B"})
        self.assertEqual(sum(instance.dataset == "A" for instance in instances), 27)
        self.assertEqual(sum(instance.dataset == "B" for instance in instances), 23)
        self.assertEqual(
            [instance.name for instance in instances],
            sorted((instance.name for instance in instances), key=natural_key),
        )
        self.assertEqual(self.evaluation.training_sets, ("A", "B"))

    def test_default_data_root_is_workspace_relative_and_configurable(self):
        default_evaluation = CVRPFEvaluation(safe_evaluate=False)
        self.assertEqual(default_evaluation.data_root.resolve(), DATA_ROOT.resolve())
        self.assertEqual(self.evaluation.data_root.resolve(), DATA_ROOT.resolve())

    def test_supported_method_is_eoh_only_without_class_metadata(self):
        self.assertEqual(self.evaluation.supported_methods, ("EoH",))
        self.assertEqual(self.evaluation.random_seed, 0)
        for forbidden in ("candidate_type", "candidate_name", "candidate_call_signature"):
            self.assertFalse(hasattr(self.evaluation, forbidden))

    def test_template_baseline_is_feasible_on_all_training_instances(self):
        artifact = self.evaluation.evaluate_with_artifact(self.solve)
        self.assertTrue(math.isfinite(artifact["fitness"]))
        self.assertTrue(math.isfinite(artifact["mean_distance"]))
        self.assertEqual(len(artifact["instance_costs"]), 50)

    def test_fitness_is_negative_mean_distance_and_artifact_is_ordered(self):
        artifact = self.evaluation.evaluate_with_artifact(self.solve)
        costs = [row["candidate_cost"] for row in artifact["instance_costs"]]
        names = [row["instance_name"] for row in artifact["instance_costs"]]
        self.assertAlmostEqual(artifact["mean_distance"], float(np.mean(costs)), places=12)
        self.assertAlmostEqual(artifact["fitness"], -float(np.mean(costs)), places=12)
        self.assertEqual(names, [instance.name for instance in self.evaluation.training_instances])
        self.assertLess(artifact["fitness"], 0.0)

    def test_evaluate_program_and_evaluate_return_same_scalar_fitness(self):
        expected = self.evaluation.evaluate_with_artifact(self.solve)["fitness"]
        self.assertAlmostEqual(self.evaluation.evaluate_program("", self.solve), expected)
        self.assertAlmostEqual(self.evaluation.evaluate(self.solve), expected)

    def test_same_candidate_and_data_have_identical_fitness(self):
        first = self.evaluation.evaluate_program("", self.solve)
        second = self.evaluation.evaluate_program("", self.solve)
        self.assertEqual(first, second)

    def test_any_candidate_failure_invalidates_complete_candidate(self):
        def raises(*_args):
            raise RuntimeError("candidate failed")

        invalid_candidates = (
            raises,
            lambda *_args: [],
            lambda matrix, *_args: [[0, *range(1, len(matrix) - 1), 0]],
        )
        for candidate in invalid_candidates:
            with self.subTest(candidate=candidate):
                self.assertIsNone(self.evaluation.evaluate_program("", candidate))

    def test_candidate_receives_copies_of_arrays(self):
        original_matrices = [
            instance.distance_matrix.copy() for instance in self.evaluation.training_instances
        ]
        original_demands = [
            instance.demands.copy() for instance in self.evaluation.training_instances
        ]

        def mutating_candidate(matrix, demands, capacity, max_vehicles):
            routes = self.solve(matrix, demands, capacity, max_vehicles)
            matrix[:] = 0
            demands[:] = 0
            return routes

        self.assertIsNotNone(self.evaluation.evaluate_program("", mutating_candidate))
        for instance, matrix, demands in zip(
            self.evaluation.training_instances, original_matrices, original_demands
        ):
            np.testing.assert_array_equal(instance.distance_matrix, matrix)
            np.testing.assert_array_equal(instance.demands, demands)

    def test_template_runs_through_standard_secure_evaluator(self):
        evaluation = CVRPFEvaluation(data_root=DATA_ROOT, timeout_seconds=30)
        score = SecureEvaluator(evaluation).evaluate_program(template_program)
        self.assertIsInstance(score, float)
        self.assertTrue(math.isfinite(score))

    def test_standard_secure_evaluator_timeout_invalidates_candidate(self):
        program = """import numpy as np

def solve(distance_matrix, demands, vehicle_capacity, max_vehicles):
    while True:
        pass
"""
        evaluation = CVRPFEvaluation(data_root=DATA_ROOT, timeout_seconds=0.1)
        self.assertIsNone(SecureEvaluator(evaluation).evaluate_program(program))


class CVRPFIntegrationTests(unittest.TestCase):
    def test_dynamic_task_export_and_gui_method_guard(self):
        self.assertIs(task_module.CVRPFEvaluation, CVRPFEvaluation)
        evaluation = CVRPFEvaluation(data_root=DATA_ROOT, safe_evaluate=False)
        _validate_method_evaluation_compatibility("EoH", evaluation)
        with self.assertRaisesRegex(ValueError, "CVRPFEvaluation.*LLaMEA"):
            _validate_method_evaluation_compatibility("LLaMEA", evaluation)

    def test_paras_yaml_has_only_function_mode_configuration(self):
        config = yaml.safe_load((TASK_DIR / "paras.yaml").read_text(encoding="utf-8"))
        self.assertEqual(config["name"], "CVRPFEvaluation")
        self.assertEqual(config["timeout_seconds"], 30)
        self.assertNotIn("data_root", config)
        for forbidden in (
            "candidate_type",
            "candidate_name",
            "candidate_call_signature",
            "use_smoke_test",
            "smoke_test_only",
        ):
            self.assertNotIn(forbidden, config)

    def test_task_has_no_random_generation_smoke_gate_or_p_evaluation(self):
        template_text = (TASK_DIR / "template.py").read_text(encoding="utf-8")
        evaluation_text = (TASK_DIR / "evaluation.py").read_text(encoding="utf-8")
        combined = template_text + evaluation_text
        for forbidden in (
            "np.random",
            "random.",
            "smoke_gate",
            "smoke_test",
            "gap_summary",
            "evaluate_cvrpf_best_on_p",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, combined)
        self.assertIn('training_sets = ("A", "B")', evaluation_text)

    def test_source_documents_have_set_specific_attribution(self):
        for path in (SOURCE_EN, SOURCE_ZH):
            text = path.read_text(encoding="utf-8")
            with self.subTest(path=path.name):
                self.assertIn("Augerat", text)
                self.assertIn("1995", text)
                self.assertIn("Computational results with a branch and cut code", text)
                self.assertIn("Uchoa", text)
                self.assertIn("2017", text)
                self.assertIn("10.1016/j.ejor.2016.08.012", text)
                self.assertIn("A", text)
                self.assertIn("B", text)
                self.assertIn("P", text)
                self.assertIn("X", text)


if __name__ == "__main__":
    unittest.main()
