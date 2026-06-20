import unittest

from llm4ad.gui import _resolve_llm_class
from llm4ad.method.llamea.llamea_llm import LlameaLLM
from llm4ad.tools.llm.llm_api_https import HttpsApi


class GuiLlameaCompatibilityTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
