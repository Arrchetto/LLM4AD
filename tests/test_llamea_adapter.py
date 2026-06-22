import unittest
from unittest.mock import Mock, patch

from llm4ad.method.llamea.llamea import LLaMEA, _relocate_llamea_log_dir


class _DummyLLM:
    model = "test-model"

    def query(self, messages):
        return ""


class _DummyEvaluation:
    task_description = "task from evaluation"
    template_program = "def candidate():\n    return 0\n"


class LlameaAdapterTest(unittest.TestCase):
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
