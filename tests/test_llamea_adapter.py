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


if __name__ == "__main__":
    unittest.main()
