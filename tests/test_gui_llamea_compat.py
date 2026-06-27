import tempfile
import unittest
from pathlib import Path

from llm4ad.gui import (
    _constructor_parameters,
    _resolve_llm_class,
    _validate_method_evaluation_compatibility,
)
from llm4ad.task.optimization.tsp_eoh_matrix import TSPEoHMatrixEvaluation
from llm4ad.method.llamea.llamea_llm import LlameaLLM
from llm4ad.tools.llm.llm_api_https import HttpsApi
from llm4ad.tools.profiler import ProfilerBase

from GUI.run_gui import (
    _complete_method_parameters,
    _profiler_name_for_method,
    _redact_sensitive_mapping,
)


class GuiLlameaCompatibilityTest(unittest.TestCase):
    def test_component_name_is_not_forwarded_to_constructor(self):
        config = {"name": "TSPEoHMatrixEvaluation", "timeout_seconds": 60}
        self.assertEqual(
            _constructor_parameters(config),
            {"timeout_seconds": 60},
        )
        self.assertEqual(config["name"], "TSPEoHMatrixEvaluation")

    def test_class_evaluation_accepts_llamea(self):
        class ClassEvaluation:
            supported_methods = ("LLaMEA",)

        _validate_method_evaluation_compatibility(
            "LLaMEA",
            ClassEvaluation(),
        )

    def test_class_evaluation_rejects_unsupported_method(self):
        class OrienteeringClassEvaluation:
            supported_methods = ("LLaMEA",)

        with self.assertRaisesRegex(
            ValueError,
            "OrienteeringClassEvaluation.*EoH",
        ):
            _validate_method_evaluation_compatibility(
                "EoH",
                OrienteeringClassEvaluation(),
            )

    def test_legacy_evaluation_without_restriction_accepts_any_method(self):
        class LegacyEvaluation:
            pass

        _validate_method_evaluation_compatibility(
            "EoH",
            LegacyEvaluation(),
        )

    def test_llamea_uses_its_compatible_https_client(self):
        resolved = _resolve_llm_class(
            method_name="LLaMEA",
            llm_name="HttpsApi",
            default_class=HttpsApi,
        )

        self.assertIs(resolved, LlameaLLM)

    def test_other_methods_keep_the_selected_https_client(self):
        resolved = _resolve_llm_class(
            method_name="EoH",
            llm_name="HttpsApi",
            default_class=HttpsApi,
        )

        self.assertIs(resolved, HttpsApi)

    def test_tsp_matrix_task_only_accepts_eoh(self):
        task = TSPEoHMatrixEvaluation(safe_evaluate=False)
        _validate_method_evaluation_compatibility("EoH", task)
        with self.assertRaisesRegex(ValueError, "TSPEoHMatrixEvaluation.*LLaMEA"):
            _validate_method_evaluation_compatibility("LLaMEA", task)

    def test_eoh_uses_eoh_profiler_in_gui(self):
        self.assertEqual(_profiler_name_for_method("EoH"), "EoHProfiler")
        self.assertEqual(_profiler_name_for_method("LLaMEA"), "ProfilerBase")

    def test_gui_preserves_explicit_num_samplers(self):
        parameters = _complete_method_parameters(
            {"name": "EoH", "num_samplers": 1, "num_evaluators": 4}
        )
        self.assertEqual(parameters["num_samplers"], 1)

    def test_gui_legacy_config_falls_back_to_num_evaluators(self):
        parameters = _complete_method_parameters(
            {"name": "EoH", "num_evaluators": 3}
        )
        self.assertEqual(parameters["num_samplers"], 3)

    def test_profiler_redacts_llm_credentials(self):
        class LLMConfig:
            def __init__(self):
                self._key = "secret-api-key"
                self.access_token = "secret-token"
                self.model = "test-model"

        class EmptyConfig:
            pass

        with tempfile.TemporaryDirectory() as tmp:
            profiler = ProfilerBase(log_dir=tmp, create_random_path=False)
            profiler.record_parameters(LLMConfig(), EmptyConfig(), EmptyConfig())
            for handler in list(profiler._logger_txt.handlers):
                profiler._logger_txt.removeHandler(handler)
                handler.close()
            run_log = (Path(tmp) / "run_log.txt").read_text(encoding="utf-8")
        self.assertNotIn("secret-api-key", run_log)
        self.assertNotIn("secret-token", run_log)
        self.assertIn("test-model", run_log)
        self.assertIn("<redacted>", run_log)

    def test_gui_console_output_redacts_credentials_without_mutating_config(self):
        config = {"key": "secret-api-key", "access_token": "secret-token", "model": "m"}
        redacted = _redact_sensitive_mapping(config)
        self.assertEqual(redacted["key"], "<redacted>")
        self.assertEqual(redacted["access_token"], "<redacted>")
        self.assertEqual(redacted["model"], "m")
        self.assertEqual(config["key"], "secret-api-key")


if __name__ == "__main__":
    unittest.main()
