"""
FastAPI request models.

This module defines Pydantic models for API request validation.
"""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, validator


class EvaluationRequest(BaseModel):
    """Request model for single document evaluation."""

    document_content: str = Field(
        ...,
        min_length=1,
        max_length=1048576,
        description="The text content to evaluate",
    )
    context: Optional[str] = Field(
        None, max_length=512000, description="Additional context information"
    )
    output_text: Optional[str] = Field(
        None, max_length=1024, description="Desired output text format"
    )
    provider: Optional[str] = Field(None, description="LLM provider preference")
    options: Optional[Dict[str, Any]] = Field(None, description="Evaluation options")

    @validator("provider")
    def validate_provider(cls, v):
        """Validate LLM provider."""
        if v is not None:
            allowed_providers = ["openai", "anthropic", "bedrock", "bedrock_nova"]
            if v not in allowed_providers:
                raise ValueError(f"Provider must be one of {allowed_providers}")
        return v


class BatchDocument(BaseModel):
    """Model for a document in a batch request."""

    document_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Unique identifier for this document",
    )
    content: str = Field(
        ..., min_length=1, max_length=1048576, description="Document content"
    )
    context: Optional[str] = Field(
        None, max_length=512000, description="Additional context"
    )
    metadata: Optional[Dict[str, Any]] = Field(None, description="Document metadata")


class BatchOptions(BaseModel):
    """Options for batch processing."""

    max_concurrent: int = Field(
        default=5, ge=1, le=10, description="Maximum concurrent evaluations"
    )
    timeout_per_document: int = Field(
        default=300, ge=30, le=1800, description="Timeout per document in seconds"
    )
    retry_failed: bool = Field(
        default=True, description="Whether to retry failed documents"
    )
    notification_webhook: Optional[str] = Field(
        None, description="Webhook for notifications"
    )


class BatchRequest(BaseModel):
    """Request model for batch evaluation."""

    documents: List[BatchDocument] = Field(
        ..., min_items=1, max_items=100, description="List of documents to evaluate"
    )
    batch_options: Optional[BatchOptions] = Field(
        None, description="Batch processing options"
    )
    callback_url: Optional[str] = Field(
        None, description="Webhook URL for completion notification"
    )
    priority: str = Field(default="NORMAL", description="Processing priority")
    provider: Optional[str] = Field(None, description="LLM provider preference")

    @validator("priority")
    def validate_priority(cls, v):
        """Validate priority value."""
        allowed_priorities = ["LOW", "NORMAL", "HIGH", "URGENT"]
        if v not in allowed_priorities:
            raise ValueError(f"Priority must be one of {allowed_priorities}")
        return v

    @validator("provider")
    def validate_provider(cls, v):
        """Validate LLM provider."""
        if v is not None:
            allowed_providers = ["openai", "anthropic", "bedrock", "bedrock_nova"]
            if v not in allowed_providers:
                raise ValueError(f"Provider must be one of {allowed_providers}")
        return v


class StatusRequest(BaseModel):
    """Request model for status checking."""

    request_id: str = Field(
        ..., min_length=1, max_length=100, description="Request identifier"
    )


class HealthRequest(BaseModel):
    """Request model for health check."""

    # Empty model for health check requests
    pass
