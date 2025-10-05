"""
FastAPI error models.

This module defines Pydantic models for API error responses.
"""

from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class APIError(BaseModel):
    """Model for API error details."""

    error_code: str = Field(..., description="Machine-readable error code")
    error_message: str = Field(..., description="Human-readable error message")
    error_type: str = Field(..., description="Error category")
    correlation_id: Optional[str] = Field(None, description="Request correlation ID")
    timestamp: str = Field(..., description="Error occurrence time")
    details: Optional[Dict[str, Any]] = Field(
        None, description="Additional error context"
    )
    resolution_guidance: Optional[str] = Field(
        None, description="Suggested resolution steps"
    )


class APIErrorResponse(BaseModel):
    """Response model for API errors."""

    error_code: str = Field(..., description="Machine-readable error code")
    error_message: str = Field(..., description="Human-readable error message")
    error_type: str = Field(..., description="Error category")
    correlation_id: Optional[str] = Field(None, description="Request correlation ID")
    timestamp: str = Field(..., description="Error occurrence time")
    details: Optional[Dict[str, Any]] = Field(
        None, description="Additional error context"
    )
    resolution_guidance: Optional[str] = Field(
        None, description="Suggested resolution steps"
    )

    class Config:
        """Pydantic configuration."""

        json_encoders = {
            # Add any custom encoders if needed
        }
