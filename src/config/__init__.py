"""
Configuration package for disclosure evaluator.
"""

from .manager import ConfigManager
from .prompts import (
    SYSTEM_STEP_EVALUATION_PROMPT,
    SYSTEM_SCORE_REASONING_PROMPT,
    USER_STEP_TEMPLATE_PROMPT,
)

__all__ = [
    "ConfigManager",
    "SYSTEM_STEP_EVALUATION_PROMPT",
    "SYSTEM_SCORE_REASONING_PROMPT",
    "USER_STEP_TEMPLATE_PROMPT",
]
