"""
Configuration manager for the disclosure evaluator.
"""

import json
from typing import Any, Dict
from .prompts import (
    SYSTEM_STEP_EVALUATION_PROMPT,
    SYSTEM_SCORE_REASONING_PROMPT,
    USER_STEP_TEMPLATE_PROMPT,
)


class ConfigManager:
    """Configuration manager for the disclosure evaluator"""

    def __init__(self, config_path: str = "config.json"):
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            raise ValueError(f"Configuration file not found: {self.config_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in configuration file: {e}")

    def get(self, key_path: str, default: Any = None) -> Any:
        """Get configuration value using dot notation (e.g., 'openai.model')"""
        keys = key_path.split(".")
        value = self.config

        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default

    def get_openai_config(self) -> Dict[str, Any]:
        """Get OpenAI configuration"""
        return self.config.get("openai", {})

    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration"""
        return self.config.get("logging", {})

    def get_evaluation_config(self) -> Dict[str, Any]:
        """Get evaluation configuration"""
        return self.config.get("evaluation", {})

    def get_output_config(self) -> Dict[str, Any]:
        """Get output configuration"""
        return self.config.get("output", {})

    def get_prompt(self, prompt_key: str) -> str:
        """Get prompt content by key (e.g., 'system.step_evaluation')"""
        # Map prompt keys to internal constants
        prompt_map = {
            "system.step_evaluation": SYSTEM_STEP_EVALUATION_PROMPT,
            "system.score_reasoning": SYSTEM_SCORE_REASONING_PROMPT,
            "user.step_template": USER_STEP_TEMPLATE_PROMPT,
        }

        if prompt_key not in prompt_map:
            raise ValueError(f"Prompt key not found: {prompt_key}")

        return prompt_map[prompt_key]

    def get_llm_config(self) -> Dict[str, Any]:
        """Get LLM configuration"""
        return self.config.get("llm", {})

    def get_provider_config(self, provider: str) -> Dict[str, Any]:
        """Get specific provider configuration"""
        llm_config = self.get_llm_config()
        return llm_config.get(provider, {})

    def get_current_provider(self) -> str:
        """Get current LLM provider"""
        llm_config = self.get_llm_config()
        return llm_config.get("provider", "openai")
