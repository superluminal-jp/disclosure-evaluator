"""
Unit tests for LLM provider classes.
Tests lines 111-264: LLMProvider, OpenAIProvider, AnthropicProvider, BedrockAnthropicProvider
"""

import json
import pytest
from unittest.mock import Mock, patch, MagicMock
import boto3

from src.llm import (
    LLMProvider,
    OpenAIProvider,
    AnthropicProvider,
    BedrockAnthropicProvider,
)


class TestLLMProvider:
    """Test LLMProvider abstract base class."""

    def test_llm_provider_init(self):
        """Test LLMProvider initialization."""
        config = {"test": "value"}
        provider = LLMProvider(config)
        assert provider.config == config

    def test_llm_provider_generate_response_raises_not_implemented(self):
        """Test that generate_response raises NotImplementedError."""
        provider = LLMProvider({})
        with pytest.raises(NotImplementedError):
            provider.generate_response([{"role": "user", "content": "test"}])


class TestOpenAIProvider:
    """Test OpenAIProvider class."""

    def test_openai_provider_init_success(self, mock_openai_client):
        """Test OpenAIProvider initialization with valid config."""
        config = {
            "api_key": "test-key",
            "model": "gpt-4",
            "temperature": 0.1,
            "max_tokens": 2000,
        }

        with patch("src.llm.providers.OpenAI", return_value=mock_openai_client):
            provider = OpenAIProvider(config)

            assert provider.config == config
            assert provider.model == "gpt-4"
            assert provider.temperature == 0.1
            assert provider.max_tokens == 2000

    def test_openai_provider_init_defaults(self, mock_openai_client):
        """Test OpenAIProvider initialization with default values."""
        config = {"api_key": "test-key"}

        with patch("src.llm.providers.OpenAI", return_value=mock_openai_client):
            provider = OpenAIProvider(config)

            assert provider.model == "gpt-4"  # Default from mock
            assert provider.temperature == 0.1
            assert provider.max_tokens == 2000

    def test_openai_provider_generate_response_success(self, mock_openai_client):
        """Test successful OpenAI response generation."""
        config = {"api_key": "test-key", "model": "gpt-4"}
        messages = [{"role": "user", "content": "test message"}]

        with patch("src.llm.providers.OpenAI", return_value=mock_openai_client):
            provider = OpenAIProvider(config)
            result = provider.generate_response(messages)

            assert result == "Mock OpenAI response"
            mock_openai_client.chat.completions.create.assert_called_once_with(
                model="gpt-4",
                messages=messages,
                temperature=0.1,
                max_completion_tokens=2000,
            )

    def test_openai_provider_generate_response_temperature_fallback(
        self, mock_openai_client
    ):
        """Test OpenAI response generation with temperature fallback."""
        config = {"api_key": "test-key", "model": "gpt-5-nano", "temperature": 0.1}
        messages = [{"role": "user", "content": "test message"}]

        # Mock temperature error followed by success
        def side_effect(*args, **kwargs):
            if "temperature" in kwargs:
                raise Exception("temperature does not support 0.1 with this model")
            return Mock(choices=[Mock(message=Mock(content="Mock OpenAI response"))])

        mock_openai_client.chat.completions.create.side_effect = side_effect

        with patch("src.llm.providers.OpenAI", return_value=mock_openai_client):
            provider = OpenAIProvider(config)
            result = provider.generate_response(messages)

            assert result == "Mock OpenAI response"
            # Should be called twice: once with temperature, once without
            assert mock_openai_client.chat.completions.create.call_count == 2

    def test_openai_provider_generate_response_api_error(self, mock_openai_client):
        """Test OpenAI API error handling."""
        config = {"api_key": "test-key", "model": "gpt-4"}
        messages = [{"role": "user", "content": "test message"}]

        # Mock API error
        mock_openai_client.chat.completions.create.side_effect = Exception("API Error")

        with patch("src.llm.providers.OpenAI", return_value=mock_openai_client):
            provider = OpenAIProvider(config)

            with pytest.raises(Exception, match="OpenAI API error: API Error"):
                provider.generate_response(messages)


class TestAnthropicProvider:
    """Test AnthropicProvider class."""

    def test_anthropic_provider_init_success(self, mock_anthropic_client):
        """Test AnthropicProvider initialization with valid config."""
        config = {
            "api_key": "test-key",
            "model": "claude-3-5-sonnet-20241022",
            "temperature": 0.1,
            "max_tokens": 2000,
        }

        with patch(
            "src.llm.providers.anthropic.Anthropic", return_value=mock_anthropic_client
        ):
            provider = AnthropicProvider(config)

            assert provider.config == config
            assert provider.model == "claude-3-5-sonnet-20241022"
            assert provider.temperature == 0.1
            assert provider.max_tokens == 2000

    def test_anthropic_provider_init_defaults(self, mock_anthropic_client):
        """Test AnthropicProvider initialization with default values."""
        config = {"api_key": "test-key"}

        with patch(
            "src.llm.providers.anthropic.Anthropic", return_value=mock_anthropic_client
        ):
            provider = AnthropicProvider(config)

            assert provider.model == "claude-3-5-sonnet-20241022"
            assert provider.temperature == 0.1
            assert provider.max_tokens == 2000

    def test_anthropic_provider_generate_response_success(self, mock_anthropic_client):
        """Test successful Anthropic response generation."""
        config = {"api_key": "test-key", "model": "claude-3"}
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "test message"},
        ]

        with patch(
            "src.llm.providers.anthropic.Anthropic", return_value=mock_anthropic_client
        ):
            provider = AnthropicProvider(config)
            result = provider.generate_response(messages)

            assert result == "Mock Anthropic response"
            mock_anthropic_client.messages.create.assert_called_once_with(
                model="claude-3",
                max_tokens=2000,
                temperature=0.1,
                system="You are a helpful assistant.",
                messages=[{"role": "user", "content": "test message"}],
            )

    def test_anthropic_provider_generate_response_no_system_message(
        self, mock_anthropic_client
    ):
        """Test Anthropic response generation without system message."""
        config = {"api_key": "test-key", "model": "claude-3"}
        messages = [{"role": "user", "content": "test message"}]

        with patch(
            "src.llm.providers.anthropic.Anthropic", return_value=mock_anthropic_client
        ):
            provider = AnthropicProvider(config)
            result = provider.generate_response(messages)

            assert result == "Mock Anthropic response"
            mock_anthropic_client.messages.create.assert_called_once_with(
                model="claude-3",
                max_tokens=2000,
                temperature=0.1,
                system="",
                messages=[{"role": "user", "content": "test message"}],
            )

    def test_anthropic_provider_generate_response_multiple_user_messages(
        self, mock_anthropic_client
    ):
        """Test Anthropic response generation with multiple user messages."""
        config = {"api_key": "test-key", "model": "claude-3"}
        messages = [
            {"role": "system", "content": "System message"},
            {"role": "user", "content": "First user message"},
            {"role": "user", "content": "Second user message"},
        ]

        with patch(
            "src.llm.providers.anthropic.Anthropic", return_value=mock_anthropic_client
        ):
            provider = AnthropicProvider(config)
            result = provider.generate_response(messages)

            assert result == "Mock Anthropic response"
            mock_anthropic_client.messages.create.assert_called_once_with(
                model="claude-3",
                max_tokens=2000,
                temperature=0.1,
                system="System message",
                messages=[
                    {
                        "role": "user",
                        "content": "First user message\n\nSecond user message",
                    }
                ],
            )

    def test_anthropic_provider_generate_response_api_error(
        self, mock_anthropic_client
    ):
        """Test Anthropic API error handling."""
        config = {"api_key": "test-key", "model": "claude-3"}
        messages = [{"role": "user", "content": "test message"}]

        # Mock API error
        mock_anthropic_client.messages.create.side_effect = Exception("API Error")

        with patch(
            "src.llm.providers.anthropic.Anthropic", return_value=mock_anthropic_client
        ):
            provider = AnthropicProvider(config)

            with pytest.raises(Exception, match="Anthropic API error: API Error"):
                provider.generate_response(messages)


class TestBedrockAnthropicProvider:
    """Test BedrockAnthropicProvider class."""

    def test_bedrock_provider_init_success(self, mock_bedrock_client):
        """Test BedrockAnthropicProvider initialization with valid config."""
        config = {
            "model": "arn:aws:bedrock:us-east-1:123456789012:inference-profile/global.anthropic.claude-sonnet-4-20250514-v1:0",
            "temperature": 0.1,
            "max_tokens": 2000,
        }

        with patch("src.llm.providers.boto3.client", return_value=mock_bedrock_client):
            with patch.dict("os.environ", {"AWS_REGION": "us-east-1"}):
                provider = BedrockAnthropicProvider(config)

                assert provider.config == config
                assert provider.model_id == config["model"]
                assert provider.temperature == 0.1
                assert provider.max_tokens == 2000
                assert provider.bedrock_client == mock_bedrock_client

    def test_bedrock_provider_init_defaults(self, mock_bedrock_client):
        """Test BedrockAnthropicProvider initialization with default values."""
        config = {}

        with patch("src.llm.providers.boto3.client", return_value=mock_bedrock_client):
            with patch.dict("os.environ", {"AWS_REGION": "us-east-1"}):
                provider = BedrockAnthropicProvider(config)

                assert (
                    provider.model_id
                    == "global.anthropic.claude-sonnet-4-20250514-v1:0"
                )
                assert provider.temperature == 0.1
                assert provider.max_tokens == 2000

    def test_bedrock_provider_init_region_from_env(self, mock_bedrock_client):
        """Test BedrockAnthropicProvider initialization with region from environment."""
        config = {}

        with patch(
            "src.llm.providers.boto3.client", return_value=mock_bedrock_client
        ) as mock_client:
            with patch.dict("os.environ", {"AWS_REGION": "us-west-2"}):
                provider = BedrockAnthropicProvider(config)

                mock_client.assert_called_once_with(
                    "bedrock-runtime", region_name="us-west-2"
                )

    def test_bedrock_provider_generate_response_anthropic_model(
        self, mock_bedrock_client
    ):
        """Test Bedrock response generation for Anthropic model."""
        config = {
            "model": "arn:aws:bedrock:us-east-1:123456789012:inference-profile/global.anthropic.claude-sonnet-4-20250514-v1:0",
            "temperature": 0.1,
            "max_tokens": 2000,
        }
        messages = [
            {"role": "system", "content": "System message"},
            {"role": "user", "content": "User message"},
        ]

        # Mock response for Anthropic model
        mock_response = {"body": Mock(), "contentType": "application/json"}
        mock_response["body"].read.return_value = json.dumps(
            {"content": [{"text": "Bedrock Anthropic response"}]}
        ).encode()
        mock_bedrock_client.invoke_model.return_value = mock_response

        with patch("src.llm.providers.boto3.client", return_value=mock_bedrock_client):
            provider = BedrockAnthropicProvider(config)
            result = provider.generate_response(messages)

            assert result == "Bedrock Anthropic response"
            mock_bedrock_client.invoke_model.assert_called_once()

            # Verify request body structure
            call_args = mock_bedrock_client.invoke_model.call_args
            assert call_args[1]["modelId"] == config["model"]
            request_body = json.loads(call_args[1]["body"])
            assert request_body["anthropic_version"] == "bedrock-2023-05-31"
            assert request_body["system"] == "System message"
            assert request_body["messages"] == [
                {"role": "user", "content": "User message"}
            ]

    def test_bedrock_provider_generate_response_nova_model(self, mock_bedrock_client):
        """Test Bedrock response generation for Nova model."""
        config = {
            "model": "arn:aws:bedrock:us-east-1:123456789012:inference-profile/us.amazon.nova-premier-v1:0",
            "temperature": 0.1,
            "max_tokens": 2000,
        }
        messages = [
            {"role": "system", "content": "System message"},
            {"role": "user", "content": "User message"},
        ]

        # Mock response for Nova model
        mock_response = {"body": Mock(), "contentType": "application/json"}
        mock_response["body"].read.return_value = json.dumps(
            {"output": {"message": {"content": [{"text": "Bedrock Nova response"}]}}}
        ).encode()
        mock_bedrock_client.invoke_model.return_value = mock_response

        with patch("src.llm.providers.boto3.client", return_value=mock_bedrock_client):
            provider = BedrockAnthropicProvider(config)
            result = provider.generate_response(messages)

            assert result == "Bedrock Nova response"
            mock_bedrock_client.invoke_model.assert_called_once()

            # Verify request body structure for Nova
            call_args = mock_bedrock_client.invoke_model.call_args
            request_body = json.loads(call_args[1]["body"])
            assert request_body["messages"] == [
                {"role": "system", "content": "System message"},
                {"role": "user", "content": "User message"},
            ]

    def test_bedrock_provider_generate_response_no_system_message(
        self, mock_bedrock_client
    ):
        """Test Bedrock response generation without system message."""
        config = {
            "model": "arn:aws:bedrock:us-east-1:123456789012:inference-profile/global.anthropic.claude-sonnet-4-20250514-v1:0",
            "temperature": 0.1,
            "max_tokens": 2000,
        }
        messages = [{"role": "user", "content": "User message"}]

        mock_response = {"body": Mock(), "contentType": "application/json"}
        mock_response["body"].read.return_value = json.dumps(
            {"content": [{"text": "Bedrock response"}]}
        ).encode()
        mock_bedrock_client.invoke_model.return_value = mock_response

        with patch("src.llm.providers.boto3.client", return_value=mock_bedrock_client):
            provider = BedrockAnthropicProvider(config)
            result = provider.generate_response(messages)

            assert result == "Bedrock response"

            # Verify request body doesn't include system message
            call_args = mock_bedrock_client.invoke_model.call_args
            request_body = json.loads(call_args[1]["body"])
            assert "system" not in request_body

    def test_bedrock_provider_generate_response_remove_none_messages(
        self, mock_bedrock_client
    ):
        """Test Bedrock response generation removes None messages."""
        config = {
            "model": "arn:aws:bedrock:us-east-1:123456789012:inference-profile/us.amazon.nova-premier-v1:0",
            "temperature": 0.1,
            "max_tokens": 2000,
        }
        messages = [{"role": "user", "content": "User message"}]

        mock_response = {"body": Mock(), "contentType": "application/json"}
        mock_response["body"].read.return_value = json.dumps(
            {"output": {"message": {"content": [{"text": "Bedrock response"}]}}}
        ).encode()
        mock_bedrock_client.invoke_model.return_value = mock_response

        with patch("src.llm.providers.boto3.client", return_value=mock_bedrock_client):
            provider = BedrockAnthropicProvider(config)
            result = provider.generate_response(messages)

            assert result == "Bedrock response"

            # Verify None messages are removed
            call_args = mock_bedrock_client.invoke_model.call_args
            request_body = json.loads(call_args[1]["body"])
            assert len(request_body["messages"]) == 1
            assert request_body["messages"][0] == {
                "role": "user",
                "content": "User message",
            }

    def test_bedrock_provider_generate_response_api_error(self, mock_bedrock_client):
        """Test Bedrock API error handling."""
        config = {
            "model": "arn:aws:bedrock:us-east-1:123456789012:inference-profile/global.anthropic.claude-sonnet-4-20250514-v1:0",
            "temperature": 0.1,
            "max_tokens": 2000,
        }
        messages = [{"role": "user", "content": "test message"}]

        # Mock API error
        mock_bedrock_client.invoke_model.side_effect = Exception("Bedrock API Error")

        with patch("src.llm.providers.boto3.client", return_value=mock_bedrock_client):
            provider = BedrockAnthropicProvider(config)

            with pytest.raises(Exception, match="Bedrock API error: Bedrock API Error"):
                provider.generate_response(messages)

    def test_bedrock_provider_generate_response_json_decode_error(
        self, mock_bedrock_client
    ):
        """Test Bedrock JSON decode error handling."""
        config = {
            "model": "arn:aws:bedrock:us-east-1:123456789012:inference-profile/global.anthropic.claude-sonnet-4-20250514-v1:0",
            "temperature": 0.1,
            "max_tokens": 2000,
        }
        messages = [{"role": "user", "content": "test message"}]

        # Mock invalid JSON response
        mock_response = {"body": Mock(), "contentType": "application/json"}
        mock_response["body"].read.return_value = b"invalid json"
        mock_bedrock_client.invoke_model.return_value = mock_response

        with patch("src.llm.providers.boto3.client", return_value=mock_bedrock_client):
            provider = BedrockAnthropicProvider(config)

            with pytest.raises(Exception, match="Bedrock API error:"):
                provider.generate_response(messages)
