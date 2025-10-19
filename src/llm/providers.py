"""
LLM provider implementations for various AI services.
"""

import json
import os
from typing import List, Dict, Any
from openai import OpenAI
import anthropic
import boto3


class LLMProvider:
    """Abstract base class for LLM providers"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def generate_response(self, messages: List[Dict[str, str]]) -> str:
        """Generate response from LLM"""
        raise NotImplementedError


class OpenAIProvider(LLMProvider):
    """OpenAI provider implementation"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.client = OpenAI(api_key=config.get("api_key"))
        self.model = config.get("model", "gpt-4")
        self.temperature = config.get("temperature", 0.1)
        self.max_tokens = config.get("max_tokens", 2000)

    def generate_response(self, messages: List[Dict[str, str]]) -> str:
        """Generate response using OpenAI API

        Handles temperature parameter compatibility for different models:
        - Some models (like gpt-5-nano) only support default temperature (1.0)
        - If temperature is not 1.0, it's included in the first attempt
        - If temperature parameter is rejected, automatically retries without it
        """
        try:
            # Prepare API parameters
            api_params = {
                "model": self.model,
                "messages": messages,
                "max_completion_tokens": self.max_tokens,
            }

            # Only include temperature if it's not the default value (1) for models that don't support custom temperature
            # Some newer models like gpt-5-nano only support the default temperature of 1
            if self.temperature != 1.0:
                api_params["temperature"] = self.temperature

            response = self.client.chat.completions.create(**api_params)
            return response.choices[0].message.content
        except Exception as e:
            # If temperature parameter is not supported, retry without it
            if "temperature" in str(e) and "does not support" in str(e):
                try:
                    api_params = {
                        "model": self.model,
                        "messages": messages,
                        "max_completion_tokens": self.max_tokens,
                    }
                    response = self.client.chat.completions.create(**api_params)
                    return response.choices[0].message.content
                except Exception as retry_e:
                    raise Exception(f"OpenAI API error: {str(retry_e)}")
            else:
                raise Exception(f"OpenAI API error: {str(e)}")


class AnthropicProvider(LLMProvider):
    """Anthropic provider implementation"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.client = anthropic.Anthropic(api_key=config.get("api_key"))
        self.model = config.get("model", "claude-3-5-sonnet-20241022")
        self.temperature = config.get("temperature", 0.1)
        self.max_tokens = config.get("max_tokens", 2000)

    def generate_response(self, messages: List[Dict[str, str]]) -> str:
        """Generate response using Anthropic API"""
        try:
            # Convert OpenAI format to Anthropic format
            system_message = ""
            user_messages = []

            for message in messages:
                if message["role"] == "system":
                    system_message = message["content"]
                else:
                    user_messages.append(message)

            # Combine user messages into a single content
            user_content = "\n\n".join([msg["content"] for msg in user_messages])

            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=system_message,
                messages=[{"role": "user", "content": user_content}],
            )
            return response.content[0].text
        except Exception as e:
            raise Exception(f"Anthropic API error: {str(e)}")


class BedrockAnthropicProvider(LLMProvider):
    """AWS Bedrock Anthropic provider implementation"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_id = config.get(
            "model",
            "jp.anthropic.claude-haiku-4-5-20251001-v1:0",
        )
        self.temperature = config.get("temperature", 0.1)
        self.max_tokens = config.get("max_tokens", 2000)

        # Initialize Bedrock client
        # AWS credentials are automatically handled by boto3 from environment
        region = os.getenv("AWS_REGION", "ap-northeast-1")
        self.bedrock_client = boto3.client("bedrock-runtime", region_name=region)

    def generate_response(self, messages: List[Dict[str, str]]) -> str:
        """Generate response using AWS Bedrock API"""
        try:
            # Convert OpenAI format to appropriate format
            system_message = ""
            user_messages = []

            for message in messages:
                if message["role"] == "system":
                    system_message = message["content"]
                else:
                    user_messages.append(message)

            # Combine user messages into a single content
            user_content = "\n\n".join([msg["content"] for msg in user_messages])

            # Determine model type and prepare appropriate request body
            if "anthropic" in self.model_id.lower():
                # Anthropic Claude model
                request_body = {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                    "messages": [{"role": "user", "content": user_content}],
                }
                if system_message:
                    request_body["system"] = system_message
            else:
                # Amazon Nova or other models
                request_body = {
                    "messages": [
                        (
                            {"role": "system", "content": system_message}
                            if system_message
                            else None
                        ),
                        {"role": "user", "content": user_content},
                    ],
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                }
                # Remove None values
                request_body["messages"] = [
                    msg for msg in request_body["messages"] if msg is not None
                ]

            # Call Bedrock API
            response = self.bedrock_client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(request_body),
                contentType="application/json",
            )

            # Parse response based on model type
            response_body = json.loads(response["body"].read())

            if "anthropic" in self.model_id.lower():
                # Anthropic Claude response format
                return response_body["content"][0]["text"]
            else:
                # Amazon Nova or other models response format
                return response_body["output"]["message"]["content"][0]["text"]

        except Exception as e:
            raise Exception(f"Bedrock API error: {str(e)}")
