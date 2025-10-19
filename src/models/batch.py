"""
Batch processing models for document evaluation.
"""

from datetime import datetime, timedelta
from enum import Enum
from typing import List, Optional, Dict, Any, TYPE_CHECKING
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from .evaluation import DisclosureEvaluationResult


class BatchStatus(Enum):
    """Batch processing status enumeration"""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIALLY_FAILED = "partially_failed"


class DocumentStatus(Enum):
    """Document processing status enumeration"""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ProcessingPhase(Enum):
    """Batch processing phase enumeration"""

    INITIALIZING = "initializing"
    DISCOVERING = "discovering"
    PROCESSING = "processing"
    AGGREGATING = "aggregating"
    COMPLETED = "completed"


class BatchConfiguration(BaseModel):
    """Configuration settings for batch processing"""

    model_config = {"extra": "forbid"}

    max_concurrent_workers: int = Field(
        default=5, ge=1, le=20, description="Maximum parallel workers"
    )
    max_retry_attempts: int = Field(
        default=3, ge=0, le=10, description="Maximum retry attempts per document"
    )
    timeout_seconds: int = Field(
        default=300, ge=30, le=3600, description="Timeout per document in seconds"
    )
    progress_update_interval: int = Field(
        default=10, ge=1, le=100, description="Progress update frequency"
    )
    enable_resumption: bool = Field(default=True, description="Enable batch resumption")
    output_formats: List[str] = Field(
        default=["json", "summary"], description="Output formats to generate"
    )
    file_size_limit: int = Field(
        default=50 * 1024 * 1024, ge=1024, description="Maximum file size in bytes"
    )
    memory_limit_mb: int = Field(default=2048, ge=256, description="Memory limit in MB")
    api_rate_limit_delay: float = Field(
        default=0.1, ge=0.0, le=10.0, description="Delay between API calls"
    )
    retry_delay_seconds: int = Field(
        default=30, ge=1, le=300, description="Delay between retries"
    )
    exponential_backoff: bool = Field(
        default=True, description="Use exponential backoff for retries"
    )


class BatchSummaryStatistics(BaseModel):
    """Statistical summary of batch evaluation results"""

    model_config = {"extra": "forbid"}

    average_score: float = Field(
        ..., ge=1.0, le=5.0, description="Average evaluation score"
    )
    score_distribution: Dict[int, int] = Field(
        ..., description="Distribution of scores (1-5 scale)"
    )
    most_common_criteria: List[str] = Field(
        ..., description="Most frequently triggered criteria"
    )
    processing_efficiency: float = Field(
        ..., ge=0.0, description="Documents processed per minute"
    )
    error_rate: float = Field(
        ..., ge=0.0, le=1.0, description="Percentage of failed documents"
    )


class DocumentResult(BaseModel):
    """Result of individual document evaluation"""

    model_config = {"extra": "forbid"}

    document_id: str = Field(..., description="Document identifier")
    evaluation_result: Optional["DisclosureEvaluationResult"] = Field(
        None, description="Full evaluation result"
    )
    processing_time: timedelta = Field(
        ..., description="Time taken to process this document"
    )
    success: bool = Field(..., description="Whether evaluation was successful")


class DocumentError(BaseModel):
    """Error information for failed document processing"""

    model_config = {"extra": "forbid"}

    document_id: str = Field(..., description="Document identifier")
    error_type: str = Field(..., description="Type of error")
    error_message: str = Field(..., description="Detailed error message")
    retry_count: int = Field(default=0, ge=0, description="Number of retry attempts")
    occurred_at: datetime = Field(..., description="When the error occurred")


class BatchEvaluation(BaseModel):
    """Batch evaluation metadata and status"""

    model_config = {"extra": "forbid"}

    batch_id: str = Field(..., description="Unique batch identifier")
    created_at: datetime = Field(..., description="Batch creation timestamp")
    status: BatchStatus = Field(..., description="Current processing status")
    total_documents: int = Field(..., ge=0, description="Total number of documents")
    processed_documents: int = Field(
        default=0, ge=0, description="Number of documents processed"
    )
    successful_documents: int = Field(
        default=0, ge=0, description="Number of successful documents"
    )
    failed_documents: int = Field(
        default=0, ge=0, description="Number of failed documents"
    )
    processing_started_at: Optional[datetime] = Field(
        None, description="When processing began"
    )
    processing_completed_at: Optional[datetime] = Field(
        None, description="When processing finished"
    )
    error_summary: Optional[str] = Field(
        None, description="Summary of errors encountered"
    )
    correlation_id: str = Field(..., description="Batch-level correlation ID")
    configuration: BatchConfiguration = Field(
        ..., description="Processing configuration"
    )


class BatchDocument(BaseModel):
    """Individual document within a batch"""

    model_config = {"extra": "forbid"}

    document_id: str = Field(..., description="Unique document identifier")
    batch_id: str = Field(..., description="Reference to parent batch")
    file_path: str = Field(..., description="Path to the document file")
    file_name: str = Field(..., description="Original filename")
    file_size: int = Field(..., ge=0, description="File size in bytes")
    mime_type: str = Field(..., description="Detected MIME type")
    status: DocumentStatus = Field(..., description="Processing status")
    processing_started_at: Optional[datetime] = Field(
        None, description="When processing began"
    )
    processing_completed_at: Optional[datetime] = Field(
        None, description="When processing finished"
    )
    evaluation_result: Optional["DisclosureEvaluationResult"] = Field(
        None, description="Evaluation results if successful"
    )
    error_message: Optional[str] = Field(None, description="Error details if failed")
    retry_count: int = Field(default=0, ge=0, description="Number of retry attempts")
    correlation_id: str = Field(..., description="Document-level correlation ID")
    context: Optional[str] = Field(
        None, description="Additional context for evaluation"
    )
    output_text: Optional[str] = Field(None, description="Output text for evaluation")


class BatchProgress(BaseModel):
    """Current state of batch processing"""

    model_config = {"extra": "forbid"}

    batch_id: str = Field(..., description="Reference to the batch")
    current_phase: ProcessingPhase = Field(..., description="Current processing phase")
    total_documents: int = Field(..., ge=0, description="Total documents to process")
    processed_documents: int = Field(
        default=0, ge=0, description="Documents processed so far"
    )
    successful_documents: int = Field(
        default=0, ge=0, description="Successfully processed documents"
    )
    failed_documents: int = Field(default=0, ge=0, description="Failed documents")
    progress_percentage: float = Field(
        ..., ge=0.0, le=100.0, description="Completion percentage"
    )
    estimated_completion: Optional[datetime] = Field(
        None, description="Estimated completion time"
    )
    current_document: Optional[str] = Field(
        None, description="Currently processing document"
    )
    active_workers: int = Field(
        default=0, ge=0, description="Number of active worker threads"
    )
    error_count: int = Field(
        default=0, ge=0, description="Total number of errors encountered"
    )
    last_updated: datetime = Field(..., description="Last progress update timestamp")


class BatchResult(BaseModel):
    """Consolidated results of a batch evaluation"""

    model_config = {"extra": "forbid"}

    batch_id: str = Field(..., description="Reference to the batch")
    total_documents: int = Field(..., ge=0, description="Total number of documents")
    successful_evaluations: int = Field(
        ..., ge=0, description="Number of successful evaluations"
    )
    failed_evaluations: int = Field(
        ..., ge=0, description="Number of failed evaluations"
    )
    success_rate: float = Field(
        ..., ge=0.0, le=1.0, description="Success rate (0.0-1.0)"
    )
    processing_duration: timedelta = Field(..., description="Total processing time")
    average_evaluation_time: Optional[timedelta] = Field(
        None, description="Average time per document"
    )
    summary_statistics: Optional[BatchSummaryStatistics] = Field(
        None, description="Statistical summary"
    )
    individual_results: List[DocumentResult] = Field(
        default=[], description="Individual document results"
    )
    error_summary: List[DocumentError] = Field(
        default=[], description="Summary of errors"
    )
    generated_at: datetime = Field(..., description="When the result was generated")


class DocumentInput(BaseModel):
    """Input specification for a document in a batch"""

    model_config = {"extra": "forbid"}

    file_path: str = Field(..., description="Path to the document file")
    file_name: Optional[str] = Field(None, description="Original filename")
    context: Optional[str] = Field(
        None, description="Additional context for evaluation"
    )
    output_text: Optional[str] = Field(None, description="Output text for evaluation")


# Resolve forward references after all models are defined
def _resolve_forward_references():
    """Resolve forward references for circular imports"""
    try:
        from .evaluation import DisclosureEvaluationResult

        DocumentResult.model_rebuild()
        BatchDocument.model_rebuild()
    except ImportError:
        # This will be resolved when the evaluation module is imported
        pass


# Call the resolution function
_resolve_forward_references()
