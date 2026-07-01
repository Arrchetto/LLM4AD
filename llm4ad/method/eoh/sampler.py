from __future__ import annotations

from typing import Tuple, List, Dict

from .prompt import EoHPrompt
from ...base import LLM, SampleTrimmer, Function, Program
from ...base.modify_code import ModifyCode


class EoHSampler:
    def __init__(self, llm: LLM, template_program: str | Program):
        self.llm = llm
        self._template_program = template_program

    def get_thought_and_function(self, prompt: str) -> Tuple[str, Function]:
        response = self.llm.draw_sample(prompt)
        thought = self.__class__.trim_thought_from_response(response)
        code = SampleTrimmer.trim_preface_of_function(response)

        function = SampleTrimmer.sample_to_function(code, self._template_program)
        return thought, function

    @classmethod
    def trim_thought_from_response(cls, response: str) -> str | None:
        """Return the explicitly boxed algorithm description.

        Code commonly contains dictionary and set literals, so accepting the
        first arbitrary ``{...}`` block can register output-schema text as the
        algorithm description and corrupt later EoH crossover prompts.
        """
        marker = r"\boxed"
        marker_start = response.find(marker)
        if marker_start < 0:
            return None
        opening = response.find("{", marker_start + len(marker))
        if opening < 0:
            return None

        depth = 0
        for position in range(opening, len(response)):
            character = response[position]
            if character == "{":
                depth += 1
            elif character == "}":
                depth -= 1
                if depth == 0:
                    return response[opening:position + 1]
        return None
