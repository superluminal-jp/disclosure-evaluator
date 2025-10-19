"""
LLM provider factory for creating provider instances.
"""

from typing import Optional
from .providers import (
    LLMProvider,
    OpenAIProvider,
    AnthropicProvider,
    BedrockAnthropicProvider,
)


def create_llm_provider(provider_name: str = None, config_manager=None) -> LLMProvider:
    """Create LLM provider based on configuration"""
    if provider_name is None:
        provider_name = config_manager.get_current_provider()

    provider_config = config_manager.get_provider_config(provider_name)

    if provider_name == "openai":
        return OpenAIProvider(provider_config)
    elif provider_name == "anthropic":
        return AnthropicProvider(provider_config)
    elif provider_name == "bedrock":
        return BedrockAnthropicProvider(provider_config)
    elif provider_name == "bedrock_nova":
        return BedrockAnthropicProvider(provider_config)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider_name}")
