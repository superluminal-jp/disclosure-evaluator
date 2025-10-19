"""
Unit tests for ConfigManager and create_llm_provider function.
Tests lines 267-390: ConfigManager class and lines 341-358: create_llm_provider function
"""

import json
import pytest
from unittest.mock import Mock, patch, mock_open
from pathlib import Path

from main import (
    ConfigManager,
    create_llm_provider,
    OpenAIProvider,
    AnthropicProvider,
    BedrockAnthropicProvider,
)


class TestConfigManager:
    """Test ConfigManager class."""

    def test_config_manager_init_success(self, temp_config_file):
        """Test ConfigManager initialization with valid config file."""
        config_manager = ConfigManager(temp_config_file)

        assert config_manager.config_path == temp_config_file
        assert isinstance(config_manager.config, dict)
        assert config_manager.config["application"]["name"] == "Test"

    def test_config_manager_init_file_not_found(self):
        """Test ConfigManager initialization with missing config file."""
        with pytest.raises(
            ValueError, match="Configuration file not found: nonexistent.json"
        ):
            ConfigManager("nonexistent.json")

    def test_config_manager_init_invalid_json(self, tmp_path):
        """Test ConfigManager initialization with invalid JSON."""
        invalid_config_file = tmp_path / "invalid.json"
        invalid_config_file.write_text("{ invalid json }")

        with pytest.raises(ValueError, match="Invalid JSON in configuration file:"):
            ConfigManager(str(invalid_config_file))

    def test_config_manager_get_success(self, temp_config_file):
        """Test ConfigManager.get() with valid key path."""
        config_manager = ConfigManager(temp_config_file)

        # Test nested key access
        assert config_manager.get("application.name") == "Test"
        assert config_manager.get("application.version") == "1.0.0"
        assert config_manager.get("llm.provider") == "openai"
        assert config_manager.get("llm.openai.model") == "gpt-4"

    def test_config_manager_get_with_default(self, temp_config_file):
        """Test ConfigManager.get() with default value for missing key."""
        config_manager = ConfigManager(temp_config_file)

        # Test with default value
        assert config_manager.get("nonexistent.key", "default_value") == "default_value"
        assert config_manager.get("nonexistent.key") is None

    def test_config_manager_get_key_error(self, temp_config_file):
        """Test ConfigManager.get() with KeyError handling."""
        config_manager = ConfigManager(temp_config_file)

        # Test missing nested key
        assert config_manager.get("application.nonexistent") is None
        assert config_manager.get("llm.nonexistent.key") is None

    def test_config_manager_get_type_error(self, temp_config_file):
        """Test ConfigManager.get() with TypeError handling."""
        config_manager = ConfigManager(temp_config_file)

        # Test accessing non-dict value as dict
        assert config_manager.get("application.name.nonexistent") is None

    def test_config_manager_get_openai_config(self, temp_config_file):
        """Test ConfigManager.get_openai_config()."""
        config_manager = ConfigManager(temp_config_file)
        openai_config = config_manager.get_openai_config()

        assert isinstance(openai_config, dict)
        assert openai_config["model"] == "gpt-4"
        assert openai_config["temperature"] == 0.1
        assert openai_config["max_tokens"] == 2000

    def test_config_manager_get_openai_config_missing(self, tmp_path):
        """Test ConfigManager.get_openai_config() when section is missing."""
        config = {"application": {"name": "Test"}}
        config_file = tmp_path / "config.json"
        with open(config_file, "w") as f:
            json.dump(config, f)

        config_manager = ConfigManager(str(config_file))
        openai_config = config_manager.get_openai_config()

        assert openai_config == {}

    def test_config_manager_get_logging_config(self, temp_config_file):
        """Test ConfigManager.get_logging_config()."""
        config_manager = ConfigManager(temp_config_file)
        logging_config = config_manager.get_logging_config()

        assert isinstance(logging_config, dict)
        assert logging_config["level"] == "INFO"
        assert logging_config["format"] == "%(message)s"

    def test_config_manager_get_evaluation_config(self, temp_config_file):
        """Test ConfigManager.get_evaluation_config()."""
        config_manager = ConfigManager(temp_config_file)
        evaluation_config = config_manager.get_evaluation_config()

        assert isinstance(evaluation_config, dict)
        assert evaluation_config["parallel"]["enabled"] is True
        assert evaluation_config["parallel"]["max_workers"] == 3

    def test_config_manager_get_output_config(self, temp_config_file):
        """Test ConfigManager.get_output_config()."""
        config_manager = ConfigManager(temp_config_file)
        output_config = config_manager.get_output_config()

        assert isinstance(output_config, dict)
        assert output_config["format"] == "json"

    def test_config_manager_get_llm_config(self, temp_config_file):
        """Test ConfigManager.get_llm_config()."""
        config_manager = ConfigManager(temp_config_file)
        llm_config = config_manager.get_llm_config()

        assert isinstance(llm_config, dict)
        assert llm_config["provider"] == "openai"
        assert "openai" in llm_config
        assert "anthropic" in llm_config
        assert "bedrock" in llm_config

    def test_config_manager_get_prompt_success(self, temp_config_file):
        """Test ConfigManager.get_prompt() with valid prompt key."""
        config_manager = ConfigManager(temp_config_file)

        # Test valid prompt keys
        prompt1 = config_manager.get_prompt("system.step_evaluation")
        assert isinstance(prompt1, str)
        assert "システムプロンプト" in prompt1

        prompt2 = config_manager.get_prompt("system.score_reasoning")
        assert isinstance(prompt2, str)
        assert "スコア推論" in prompt2

        prompt3 = config_manager.get_prompt("user.step_template")
        assert isinstance(prompt3, str)
        assert "ユーザープロンプト" in prompt3

    def test_config_manager_get_prompt_invalid_key(self, temp_config_file):
        """Test ConfigManager.get_prompt() with invalid prompt key."""
        config_manager = ConfigManager(temp_config_file)

        with pytest.raises(ValueError, match="Prompt key not found: invalid.key"):
            config_manager.get_prompt("invalid.key")

    def test_config_manager_get_provider_config(self, temp_config_file):
        """Test ConfigManager.get_provider_config()."""
        config_manager = ConfigManager(temp_config_file)

        # Test OpenAI provider config
        openai_config = config_manager.get_provider_config("openai")
        assert isinstance(openai_config, dict)
        assert openai_config["model"] == "gpt-4"

        # Test Anthropic provider config
        anthropic_config = config_manager.get_provider_config("anthropic")
        assert isinstance(anthropic_config, dict)
        assert anthropic_config["model"] == "claude-3"

        # Test Bedrock provider config
        bedrock_config = config_manager.get_provider_config("bedrock")
        assert isinstance(bedrock_config, dict)
        assert bedrock_config["model"] == "claude-3"

    def test_config_manager_get_provider_config_missing(self, temp_config_file):
        """Test ConfigManager.get_provider_config() with missing provider."""
        config_manager = ConfigManager(temp_config_file)

        missing_config = config_manager.get_provider_config("nonexistent")
        assert missing_config == {}

    def test_config_manager_get_current_provider(self, temp_config_file):
        """Test ConfigManager.get_current_provider()."""
        config_manager = ConfigManager(temp_config_file)

        current_provider = config_manager.get_current_provider()
        assert current_provider == "openai"

    def test_config_manager_get_current_provider_default(self, tmp_path):
        """Test ConfigManager.get_current_provider() with default value."""
        config = {"application": {"name": "Test"}}
        config_file = tmp_path / "config.json"
        with open(config_file, "w") as f:
            json.dump(config, f)

        config_manager = ConfigManager(str(config_file))
        current_provider = config_manager.get_current_provider()

        assert current_provider == "openai"  # Default value


class TestCreateLLMProvider:
    """Test create_llm_provider factory function."""

    def test_create_llm_provider_openai(self, temp_config_file, mock_openai_client):
        """Test creating OpenAI provider."""
        with patch("main.ConfigManager") as mock_config_class:
            mock_config = Mock()
            mock_config.get_current_provider.return_value = "openai"
            mock_config.get_provider_config.return_value = {
                "model": "gpt-4",
                "temperature": 0.1,
                "max_tokens": 2000,
            }
            mock_config_class.return_value = mock_config

            with patch("main.OpenAIProvider") as mock_provider_class:
                mock_provider = Mock()
                mock_provider_class.return_value = mock_provider

                provider = create_llm_provider("openai")

                mock_config.get_provider_config.assert_called_once_with("openai")
                mock_provider_class.assert_called_once()
                assert provider == mock_provider

    def test_create_llm_provider_anthropic(
        self, temp_config_file, mock_anthropic_client
    ):
        """Test creating Anthropic provider."""
        with patch("main.ConfigManager") as mock_config_class:
            mock_config = Mock()
            mock_config.get_current_provider.return_value = "anthropic"
            mock_config.get_provider_config.return_value = {
                "model": "claude-3",
                "temperature": 0.1,
                "max_tokens": 2000,
            }
            mock_config_class.return_value = mock_config

            with patch("main.AnthropicProvider") as mock_provider_class:
                mock_provider = Mock()
                mock_provider_class.return_value = mock_provider

                provider = create_llm_provider("anthropic")

                mock_config.get_provider_config.assert_called_once_with("anthropic")
                mock_provider_class.assert_called_once()
                assert provider == mock_provider

    def test_create_llm_provider_bedrock(self, temp_config_file, mock_bedrock_client):
        """Test creating Bedrock provider."""
        with patch("main.ConfigManager") as mock_config_class:
            mock_config = Mock()
            mock_config.get_current_provider.return_value = "bedrock"
            mock_config.get_provider_config.return_value = {
                "model": "claude-3",
                "temperature": 0.1,
                "max_tokens": 2000,
            }
            mock_config_class.return_value = mock_config

            with patch("main.BedrockAnthropicProvider") as mock_provider_class:
                mock_provider = Mock()
                mock_provider_class.return_value = mock_provider

                provider = create_llm_provider("bedrock")

                mock_config.get_provider_config.assert_called_once_with("bedrock")
                mock_provider_class.assert_called_once()
                assert provider == mock_provider

    def test_create_llm_provider_bedrock_nova(
        self, temp_config_file, mock_bedrock_client
    ):
        """Test creating Bedrock Nova provider."""
        with patch("main.ConfigManager") as mock_config_class:
            mock_config = Mock()
            mock_config.get_current_provider.return_value = "bedrock_nova"
            mock_config.get_provider_config.return_value = {
                "model": "nova-model",
                "temperature": 0.1,
                "max_tokens": 2000,
            }
            mock_config_class.return_value = mock_config

            with patch("main.BedrockAnthropicProvider") as mock_provider_class:
                mock_provider = Mock()
                mock_provider_class.return_value = mock_provider

                provider = create_llm_provider("bedrock_nova")

                mock_config.get_provider_config.assert_called_once_with("bedrock_nova")
                mock_provider_class.assert_called_once()
                assert provider == mock_provider

    def test_create_llm_provider_unsupported(self, temp_config_file):
        """Test creating unsupported provider raises ValueError."""
        with patch("main.ConfigManager") as mock_config_class:
            mock_config = Mock()
            mock_config.get_current_provider.return_value = "unsupported"
            mock_config.get_provider_config.return_value = {}
            mock_config_class.return_value = mock_config

            with pytest.raises(
                ValueError, match="Unsupported LLM provider: unsupported"
            ):
                create_llm_provider("unsupported")

    def test_create_llm_provider_default_provider(
        self, temp_config_file, mock_openai_client
    ):
        """Test creating provider with default (None) provider name."""
        with patch("main.ConfigManager") as mock_config_class:
            mock_config = Mock()
            mock_config.get_current_provider.return_value = "openai"
            mock_config.get_provider_config.return_value = {
                "model": "gpt-4",
                "temperature": 0.1,
                "max_tokens": 2000,
            }
            mock_config_class.return_value = mock_config

            with patch("main.OpenAIProvider") as mock_provider_class:
                mock_provider = Mock()
                mock_provider_class.return_value = mock_provider

                provider = create_llm_provider()  # No provider specified

                mock_config.get_current_provider.assert_called_once()
                mock_config.get_provider_config.assert_called_once_with("openai")
                mock_provider_class.assert_called_once()
                assert provider == mock_provider

    def test_create_llm_provider_with_config_manager_global(
        self, temp_config_file, mock_openai_client
    ):
        """Test create_llm_provider uses global config_manager."""
        # This test verifies that the function uses the global config_manager instance
        # when no provider is specified and it falls back to the current provider
        with patch("main.config_manager") as mock_global_config:
            mock_global_config.get_current_provider.return_value = "openai"
            mock_global_config.get_provider_config.return_value = {
                "model": "gpt-4",
                "temperature": 0.1,
                "max_tokens": 2000,
            }

            with patch("main.OpenAIProvider") as mock_provider_class:
                mock_provider = Mock()
                mock_provider_class.return_value = mock_provider

                provider = create_llm_provider()  # No provider specified

                mock_global_config.get_current_provider.assert_called_once()
                mock_global_config.get_provider_config.assert_called_once_with("openai")
                mock_provider_class.assert_called_once()
                assert provider == mock_provider
