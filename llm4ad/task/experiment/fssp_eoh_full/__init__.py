"""EoH task: evolve a complete FSSP solver."""

from .evaluation import FSSPEoHFullEvaluation
from .template import task_description, template_program

__all__ = ["FSSPEoHFullEvaluation", "task_description", "template_program"]
