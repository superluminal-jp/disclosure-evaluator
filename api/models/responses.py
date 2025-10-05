"""
FastAPI response models.

This module defines Pydantic models for API response serialization.
"""

from typing import Optional, Dict, Any, List
from datetime import datetime
from pydantic import BaseModel, Field


class EvaluationStep(BaseModel):
    """Model for an individual evaluation step."""

    step_number: int = Field(
        ..., ge=1, description="Step number in the evaluation process"
    )
    step_name: str = Field(..., description="Name of the evaluation step")
    result: str = Field(..., description="Result of this step")
    reasoning: str = Field(..., description="Reasoning for the step result")


class CriterionResult(BaseModel):
    """Model for a criterion evaluation result."""

    criterion_id: str = Field(..., description="Criterion identifier")
    criterion_name: str = Field(..., description="Human-readable criterion name")
    score: float = Field(..., ge=1.0, le=5.0, description="Score for this criterion")
    reasoning: str = Field(..., description="Detailed reasoning for the score")
    steps: List[EvaluationStep] = Field(..., description="Individual evaluation steps")


class EvaluationMetadata(BaseModel):
    """Metadata for evaluation response."""

    api_version: str = Field(default="1.0.0", description="API version used")
    processing_time: float = Field(
        ..., ge=0.0, description="Server processing time in seconds"
    )
    correlation_id: str = Field(..., description="Request correlation ID")
    timestamp: str = Field(..., description="Evaluation timestamp")


class EvaluationResponse(BaseModel):
    """Response model for single document evaluation."""

    evaluation_id: str = Field(..., description="Unique identifier for this evaluation")
    document_summary: str = Field(..., description="Brief summary of the document")
    criteria_results: List[CriterionResult] = Field(
        ..., description="Results for each evaluation criterion"
    )
    overall_score: float = Field(
        ..., ge=0.0, le=1.0, description="Overall disclosure score"
    )
    recommendation: str = Field(..., description="Final disclosure recommendation")
    processing_time: float = Field(
        ..., ge=0.0, description="Processing time in seconds"
    )
    metadata: EvaluationMetadata = Field(..., description="Evaluation metadata")


class BatchResponse(BaseModel):
    """Response model for batch evaluation submission."""

    batch_id: str = Field(..., description="Unique identifier for the batch")
    status: str = Field(..., description="Current batch status")
    total_documents: int = Field(
        ..., ge=1, description="Total number of documents in batch"
    )
    processed_documents: int = Field(
        default=0, ge=0, description="Number of documents processed"
    )
    failed_documents: int = Field(
        default=0, ge=0, description="Number of documents that failed"
    )
    estimated_completion: Optional[str] = Field(
        None, description="Estimated completion time"
    )
    results: Optional[List[EvaluationResponse]] = Field(
        None, description="Individual evaluation results (if completed)"
    )


class BatchStatusResponse(BaseModel):
    """Response model for batch status checking."""

    batch_id: str = Field(..., description="Batch identifier")
    status: str = Field(..., description="Current batch status")
    total_documents: int = Field(..., description="Total documents in batch")
    processed_documents: int = Field(
        ..., description="Documents processed successfully"
    )
    failed_documents: int = Field(..., description="Documents that failed")
    progress: float = Field(
        ..., ge=0.0, le=100.0, description="Overall progress percentage"
    )
    estimated_completion: Optional[str] = Field(
        None, description="Estimated completion time"
    )
    last_updated: str = Field(..., description="Last status update time")


class BatchResultsResponse(BaseModel):
    """Response model for batch results retrieval."""

    batch_id: str = Field(..., description="Batch identifier")
    status: str = Field(..., description="Batch completion status")
    results: List[EvaluationResponse] = Field(
        ..., description="Individual evaluation results"
    )
    summary: Dict[str, Any] = Field(..., description="Batch processing summary")


class StatusResponse(BaseModel):
    """Response model for request status checking."""

    request_id: str = Field(..., description="Request identifier")
    status: str = Field(..., description="Current processing status")
    progress: float = Field(..., ge=0.0, le=100.0, description="Progress percentage")
    message: str = Field(..., description="Human-readable status message")
    last_updated: str = Field(..., description="Last status update time")
    estimated_completion: Optional[str] = Field(
        None, description="Estimated completion time"
    )
    errors: Optional[List[Dict[str, Any]]] = Field(
        None, description="Any processing errors"
    )


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str = Field(..., description="API health status")
    timestamp: str = Field(..., description="Health check timestamp")
    version: str = Field(default="1.0.0", description="API version")
    uptime: float = Field(..., ge=0.0, description="API uptime in seconds")


class AsyncResponse(BaseModel):
    """Response model for asynchronous operations."""

    request_id: str = Field(..., description="Request identifier")
    status: str = Field(..., description="Request status")
    message: str = Field(..., description="Status message")
    estimated_completion: Optional[str] = Field(
        None, description="Estimated completion time"
    )
