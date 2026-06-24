from decimal import Decimal
from fractions import Fraction
import os
import unittest
from unittest.mock import Mock, patch

import numpy as np

from llamea import LLaMEA as LLaMEA_Algorithm
from llamea.multi_objective_fitness import Fitness
from llamea.solution import Solution

from llm4ad.method.llamea.llamea import (
    LLaMEA,
    _relocate_llamea_log_dir,
    _write_llamea_log_dir_marker,
)


class _DummyLLM:
    model = "test-model"

    def query(self, messages):
        return ""


class _DummyEvaluation:
    task_description = "task from evaluation"
    template_program = "def candidate():\n    return 0\n"


def _solution(fitness):
    solution = Solution(code="def candidate():\n    return 0\n")
    solution.fitness = fitness
    return solution


def _selection_adapter(*, n_parents=1, n_offspring=1, elitism=False):
    method = LLaMEA.__new__(LLaMEA)
    method.n_parents = n_parents
    method.n_offspring = n_offspring
    method.elitism = elitism
    method.multi_objective = False
    method.minimization = False
    method.niching = None
    return method


class LlameaAdapterTest(unittest.TestCase):
    def test_comma_one_one_replaces_parent_with_worse_finite_offspring(self):
        method = _selection_adapter()
        parent = _solution(10.0)
        offspring = _solution(1.0)

        selected = method.selection([parent], [offspring])

        self.assertEqual(selected, [offspring])

    def test_comma_one_one_keeps_parent_for_negative_infinity_offspring(self):
        method = _selection_adapter()
        parent = _solution(10.0)
        offspring = _solution(float("-inf"))

        selected = method.selection([parent], [offspring])

        self.assertEqual(selected, [parent])

    def test_comma_one_one_keeps_parent_for_other_invalid_offspring_fitness(self):
        invalid_fitnesses = (
            float("nan"),
            float("inf"),
            None,
            [1.0],
            True,
            np.bool_(True),
            1 + 2j,
            np.complex128(1 + 2j),
            Decimal("NaN"),
            Decimal("Infinity"),
        )

        for invalid_fitness in invalid_fitnesses:
            with self.subTest(fitness=invalid_fitness):
                method = _selection_adapter()
                parent = _solution(10.0)
                offspring = _solution(invalid_fitness)

                selected = method.selection([parent], [offspring])

                self.assertEqual(selected, [parent])

    def test_comma_one_one_accepts_finite_real_scalar_fitness_without_float_coercion(
        self,
    ):
        valid_fitnesses = (
            np.float64(1.25),
            Decimal("1e999999"),
            Fraction(10**10000, 3),
        )

        for valid_fitness in valid_fitnesses:
            with self.subTest(fitness_type=type(valid_fitness).__name__):
                method = _selection_adapter()
                parent = _solution(10.0)
                offspring = _solution(valid_fitness)

                selected = method.selection([parent], [offspring])

                self.assertEqual(selected, [offspring])

    def test_comma_one_one_with_niching_delegates_to_upstream_selection(self):
        method = _selection_adapter()
        method.niching = "novelty"
        parents = [_solution(10.0)]
        offspring = [_solution(1.0)]
        upstream_result = [object()]

        with patch.object(
            LLaMEA_Algorithm,
            "selection",
            return_value=upstream_result,
        ) as upstream_selection:
            selected = method.selection(parents, offspring)

        self.assertIs(selected, upstream_result)
        upstream_selection.assert_called_once_with(parents, offspring)

    def test_comma_one_one_with_empty_offspring_delegates_to_upstream_selection(
        self,
    ):
        method = _selection_adapter()
        parents = [_solution(10.0)]
        offspring = []
        upstream_result = [object()]

        with patch.object(
            LLaMEA_Algorithm,
            "selection",
            return_value=upstream_result,
        ) as upstream_selection:
            selected = method.selection(parents, offspring)

        self.assertIs(selected, upstream_result)
        upstream_selection.assert_called_once_with(parents, offspring)

    def test_comma_one_one_with_two_offspring_delegates_to_upstream_selection(
        self,
    ):
        method = _selection_adapter()
        parents = [_solution(10.0)]
        offspring = [_solution(1.0), _solution(2.0)]
        upstream_result = [object()]

        with patch.object(
            LLaMEA_Algorithm,
            "selection",
            return_value=upstream_result,
        ) as upstream_selection:
            selected = method.selection(parents, offspring)

        self.assertIs(selected, upstream_result)
        upstream_selection.assert_called_once_with(parents, offspring)

    def test_multi_objective_comma_one_one_delegates_to_upstream_selection(self):
        method = _selection_adapter()
        method.multi_objective = True
        parents = [_solution(Fitness({"quality": 10.0}))]
        offspring = [_solution(Fitness({"quality": 1.0}))]
        upstream_result = [object()]

        with patch.object(
            LLaMEA_Algorithm,
            "selection",
            return_value=upstream_result,
        ) as upstream_selection:
            selected = method.selection(parents, offspring)

        self.assertIs(selected, upstream_result)
        upstream_selection.assert_called_once_with(parents, offspring)

    def test_non_exact_comma_one_one_configurations_delegate_to_upstream_selection(
        self,
    ):
        configurations = (
            {"n_parents": 2, "n_offspring": 1, "elitism": False},
            {"n_parents": 1, "n_offspring": 2, "elitism": False},
            {"n_parents": 1, "n_offspring": 1, "elitism": True},
        )
        parents = [_solution(10.0)]
        offspring = [_solution(1.0)]

        for configuration in configurations:
            with self.subTest(configuration=configuration):
                method = _selection_adapter(**configuration)
                upstream_result = [object()]
                with patch.object(
                    LLaMEA_Algorithm,
                    "selection",
                    return_value=upstream_result,
                ) as upstream_selection:
                    selected = method.selection(parents, offspring)

                self.assertIs(selected, upstream_result)
                upstream_selection.assert_called_once_with(parents, offspring)

    def test_writes_explicit_llamea_log_directory_marker(self):
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as profiler_dir:
            profiler = Mock()
            profiler._log_dir = profiler_dir
            logger = Mock()
            logger.dirname = "/repo/GUI/logs/llamea/exp-test"

            _write_llamea_log_dir_marker(logger, profiler)

            marker_path = os.path.join(
                profiler_dir,
                "llamea_output_dir.txt",
            )
            with open(marker_path, encoding="utf-8") as file:
                self.assertEqual(file.read(), logger.dirname)

    @patch("llm4ad.method.llamea.llamea.shutil.move")
    @patch("llm4ad.method.llamea.llamea.os.makedirs")
    def test_relocates_third_party_logs_under_gui_logs_llamea(
        self,
        makedirs,
        move,
    ):
        logger = Mock()
        source_dir = "/working/exp-06-20_201712-LLaMEA-test-model-"
        logger.dirname = source_dir
        profiler = Mock()
        profiler._log_dir = "/repo/GUI/logs/20260620_201712_Task_LLaMEA"

        _relocate_llamea_log_dir(logger, profiler)

        target_parent = "/repo/GUI/logs/llamea"
        target_dir = (
            "/repo/GUI/logs/llamea/"
            "exp-06-20_201712-LLaMEA-test-model-"
        )
        makedirs.assert_called_once_with(target_parent, exist_ok=True)
        move.assert_called_once_with(source_dir, target_dir)
        self.assertEqual(logger.dirname, target_dir)

    @patch("llm4ad.method.llamea.llamea.shutil.move")
    @patch("llm4ad.method.llamea.llamea.os.makedirs")
    def test_reuses_existing_llamea_method_folder(
        self,
        makedirs,
        move,
    ):
        logger = Mock()
        source_dir = "/working/exp-06-24_120000-LLaMEA-test-model-"
        logger.dirname = source_dir
        profiler = Mock()
        profiler._log_dir = (
            "/repo/GUI/logs/llamea/"
            "20260624_120000_Task_LLaMEA"
        )

        _relocate_llamea_log_dir(logger, profiler)

        target_parent = "/repo/GUI/logs/llamea"
        target_dir = os.path.join(
            target_parent,
            "exp-06-24_120000-LLaMEA-test-model-",
        )
        makedirs.assert_called_once_with(target_parent, exist_ok=True)
        move.assert_called_once_with(source_dir, target_dir)
        self.assertEqual(logger.dirname, target_dir)

    @patch("llm4ad.method.llamea.llamea.shutil.move")
    @patch("llm4ad.method.llamea.llamea.os.makedirs")
    def test_reuses_llamea_test_folder_for_test_runs(
        self,
        makedirs,
        move,
    ):
        logger = Mock()
        source_dir = "/working/exp-06-24_130000-LLaMEA-test-model-"
        logger.dirname = source_dir
        profiler = Mock()
        profiler._log_dir = (
            "/repo/GUI/logs/llamea/test/"
            "20260624_130000_Task_LLaMEA"
        )

        _relocate_llamea_log_dir(logger, profiler)

        target_parent = "/repo/GUI/logs/llamea/test"
        target_dir = os.path.join(
            target_parent,
            "exp-06-24_130000-LLaMEA-test-model-",
        )
        makedirs.assert_called_once_with(target_parent, exist_ok=True)
        move.assert_called_once_with(source_dir, target_dir)
        self.assertEqual(logger.dirname, target_dir)

    @patch("llm4ad.method.llamea.llamea.generate_evaluator", return_value=lambda _: 0.0)
    @patch("llm4ad.method.llamea.llamea.LLaMEA_Algorithm.__init__", return_value=None)
    def test_defaults_to_threading_backend_to_avoid_pickling_logger(
        self,
        algorithm_init,
        _generate_evaluator,
    ):
        LLaMEA(llm=_DummyLLM(), evaluation=_DummyEvaluation())

        self.assertEqual(
            algorithm_init.call_args.kwargs["parallel_backend"],
            "threading",
        )

    @patch("llm4ad.method.llamea.llamea.generate_evaluator", return_value=lambda _: 0.0)
    @patch("llm4ad.method.llamea.llamea.LLaMEA_Algorithm.__init__", return_value=None)
    def test_maps_gui_max_sample_nums_to_llamea_budget(
        self,
        algorithm_init,
        _generate_evaluator,
    ):
        LLaMEA(
            llm=_DummyLLM(),
            evaluation=_DummyEvaluation(),
            max_sample_nums=20,
        )

        self.assertEqual(algorithm_init.call_args.kwargs["budget"], 20)

    @patch("llm4ad.method.llamea.llamea.generate_evaluator", return_value=lambda _: 0.0)
    @patch("llm4ad.method.llamea.llamea.LLaMEA_Algorithm.__init__", return_value=None)
    def test_forwards_six_gui_core_parameters(
        self,
        algorithm_init,
        _generate_evaluator,
    ):
        LLaMEA(
            llm=_DummyLLM(),
            evaluation=_DummyEvaluation(),
            max_sample_nums=35,
            n_parents=7,
            n_offsprings=4,
            num_evaluators=3,
            eval_timeout=45,
            elitism=False,
        )

        forwarded = algorithm_init.call_args.kwargs
        self.assertEqual(forwarded["budget"], 35)
        self.assertEqual(forwarded["n_parents"], 7)
        self.assertEqual(forwarded["n_offspring"], 4)
        self.assertEqual(forwarded["max_workers"], 3)
        self.assertEqual(forwarded["eval_timeout"], 45)
        self.assertIs(forwarded["elitism"], False)

    def test_rejects_invalid_gui_core_parameters(self):
        invalid_cases = (
            {"max_sample_nums": 0},
            {"n_parents": 0},
            {"n_offsprings": 0},
            {"num_evaluators": 0},
            {"eval_timeout": 0},
            {"max_sample_nums": 4, "n_parents": 5},
        )

        for parameters in invalid_cases:
            with self.subTest(parameters=parameters):
                with self.assertRaises(ValueError):
                    LLaMEA(
                        llm=_DummyLLM(),
                        evaluation=_DummyEvaluation(),
                        **parameters,
                    )

    @patch("llm4ad.method.llamea.llamea.generate_evaluator", return_value=lambda _: 0.0)
    @patch("llm4ad.method.llamea.llamea.LLaMEA_Algorithm.__init__", return_value=None)
    def test_uses_evaluation_prompts_when_gui_does_not_provide_them(
        self,
        algorithm_init,
        _generate_evaluator,
    ):
        LLaMEA(llm=_DummyLLM(), evaluation=_DummyEvaluation())

        self.assertEqual(
            algorithm_init.call_args.kwargs["task_prompt"],
            _DummyEvaluation.task_description,
        )
        self.assertEqual(
            algorithm_init.call_args.kwargs["example_prompt"],
            _DummyEvaluation.template_program,
        )

    @patch("llm4ad.method.llamea.llamea._relocate_llamea_log_dir")
    @patch("llm4ad.method.llamea.llamea.generate_evaluator", return_value=lambda _: 0.0)
    @patch("llm4ad.method.llamea.llamea.LLaMEA_Algorithm.__init__", return_value=None)
    def test_initializes_profiler_for_gui_sample_files(
        self,
        _algorithm_init,
        _generate_evaluator,
        _relocate,
    ):
        profiler = Mock()
        llm = _DummyLLM()
        evaluation = _DummyEvaluation()

        method = LLaMEA(llm=llm, evaluation=evaluation, profiler=profiler)

        profiler.record_parameters.assert_called_once_with(
            llm,
            evaluation,
            method,
        )


if __name__ == "__main__":
    unittest.main()
