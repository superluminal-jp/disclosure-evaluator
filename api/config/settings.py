"""
FastAPI application settings and configuration.

This module defines all configuration settings for the FastAPI application,
including environment variables, default values, and validation rules.
"""

import os
from typing import List, Optional
from pydantic import Field, validator
from pydantic_settings import BaseSettings


class FastAPISettings(BaseSettings):
    """FastAPI application settings."""

    # Application settings
    debug: bool = Field(default=False, description="Enable debug mode")
    environment: str = Field(default="development", description="Environment name")
    api_version: str = Field(default="1.0.0", description="API version")

    # Server settings
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")
    workers: int = Field(default=1, description="Number of worker processes")

    # Security settings
    secret_key: str = Field(
        default="your-secret-key-here", description="Secret key for JWT"
    )
    access_token_expire_minutes: int = Field(
        default=30, description="Access token expiration in minutes"
    )
    allowed_origins: List[str] = Field(
        default=["*"], description="Allowed CORS origins"
    )
    allowed_hosts: List[str] = Field(default=["*"], description="Allowed hosts")

    # API settings
    enable_docs: bool = Field(default=True, description="Enable API documentation")
    rate_limit_requests: int = Field(
        default=100, description="Rate limit requests per minute"
    )
    rate_limit_window: int = Field(
        default=60, description="Rate limit window in seconds"
    )

    # Evaluation settings
    max_document_size: int = Field(
        default=1048576, description="Maximum document size in bytes"
    )
    max_batch_size: int = Field(default=100, description="Maximum batch size")
    default_timeout: int = Field(default=300, description="Default timeout in seconds")
    max_concurrent_workers: int = Field(
        default=5, description="Maximum concurrent workers"
    )

    # LLM Provider settings
    default_provider: str = Field(default="openai", description="Default LLM provider")
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    anthropic_api_key: Optional[str] = Field(
        default=None, description="Anthropic API key"
    )
    aws_access_key_id: Optional[str] = Field(
        default=None, description="AWS access key ID"
    )
    aws_secret_access_key: Optional[str] = Field(
        default=None, description="AWS secret access key"
    )
    aws_region: str = Field(default="us-east-1", description="AWS region")

    # Logging settings
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format",
    )

    # Database settings (for future use)
    database_url: Optional[str] = Field(default=None, description="Database URL")

    # Redis settings (for caching and rate limiting)
    redis_url: Optional[str] = Field(default=None, description="Redis URL")

    class Config:
        """Pydantic configuration."""

        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"  # Ignore extra environment variables

    @validator("allowed_origins", pre=True)
    def parse_allowed_origins(cls, v):
        """Parse allowed origins from string or list."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v

    @validator("allowed_hosts", pre=True)
    def parse_allowed_hosts(cls, v):
        """Parse allowed hosts from string or list."""
        if isinstance(v, str):
            return [host.strip() for host in v.split(",")]
        return v

    @validator("environment")
    def validate_environment(cls, v):
        """Validate environment value."""
        allowed_envs = ["development", "staging", "production"]
        if v not in allowed_envs:
            raise ValueError(f"Environment must be one of {allowed_envs}")
        return v

    @validator("default_provider")
    def validate_provider(cls, v):
        """Validate LLM provider."""
        allowed_providers = ["openai", "anthropic", "bedrock", "bedrock_nova"]
        if v not in allowed_providers:
            raise ValueError(f"Provider must be one of {allowed_providers}")
        return v

    @validator("log_level")
    def validate_log_level(cls, v):
        """Validate log level."""
        allowed_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in allowed_levels:
            raise ValueError(f"Log level must be one of {allowed_levels}")
        return v.upper()

    def get_provider_config(self, provider: str) -> dict:
        """Get configuration for a specific LLM provider."""
        configs = {
            "openai": {"api_key": self.openai_api_key, "required": ["api_key"]},
            "anthropic": {"api_key": self.anthropic_api_key, "required": ["api_key"]},
            "bedrock": {
                "aws_access_key_id": self.aws_access_key_id,
                "aws_secret_access_key": self.aws_secret_access_key,
                "aws_region": self.aws_region,
                "required": [
                    "aws_access_key_id",
                    "aws_secret_access_key",
                    "aws_region",
                ],
            },
            "bedrock_nova": {
                "aws_access_key_id": self.aws_access_key_id,
                "aws_secret_access_key": self.aws_secret_access_key,
                "aws_region": self.aws_region,
                "required": [
                    "aws_access_key_id",
                    "aws_secret_access_key",
                    "aws_region",
                ],
            },
        }

        if provider not in configs:
            raise ValueError(f"Unknown provider: {provider}")

        config = configs[provider]
        missing = [key for key in config["required"] if not config.get(key)]

        if missing:
            raise ValueError(
                f"Missing required configuration for {provider}: {missing}"
            )

        return {k: v for k, v in config.items() if k != "required"}

    def is_provider_configured(self, provider: str) -> bool:
        """Check if a provider is properly configured."""
        try:
            self.get_provider_config(provider)
            return True
        except ValueError:
            return False


# Global settings instance
settings = FastAPISettings()
