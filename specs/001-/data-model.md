# Data Model: 複数文書の一括評価

**Feature**: Batch Document Evaluation  
**Date**: 2025-10-04  
**Status**: Complete

## Entity Definitions

### BatchEvaluation

Represents a collection of documents to be evaluated together with batch-level metadata and processing status.

**Fields**:

- `batch_id: str` - Unique identifier for the batch (format: `batch_YYYYMMDD_HHMMSS`)
- `created_at: datetime` - Batch creation timestamp
- `status: BatchStatus` - Current processing status (PENDING, PROCESSING, COMPLETED, FAILED, PARTIALLY_FAILED)
- `total_documents: int` - Total number of documents in the batch
- `processed_documents: int` - Number of documents processed
- `successful_documents: int` - Number of successfully processed documents
- `failed_documents: int` - Number of failed documents
- `processing_started_at: Optional[datetime]` - When batch processing began
- `processing_completed_at: Optional[datetime]` - When batch processing finished
- `error_summary: Optional[str]` - Summary of errors encountered
- `correlation_id: str` - Batch-level correlation ID for logging
- `configuration: BatchConfiguration` - Processing configuration settings

**Validation Rules**:

- `batch_id` must be unique across all batches
- `total_documents` must be > 0
- `processed_documents` cannot exceed `total_documents`
- `successful_documents + failed_documents` cannot exceed `processed_documents`
- `status` transitions: PENDING → PROCESSING → (COMPLETED | FAILED | PARTIALLY_FAILED)

**State Transitions**:

```
PENDING → PROCESSING (when batch processing starts)
PROCESSING → COMPLETED (all documents processed successfully)
PROCESSING → PARTIALLY_FAILED (some documents failed, some succeeded)
PROCESSING → FAILED (all documents failed or critical error)
```

### BatchDocument

Represents a single document within a batch with its processing status and evaluation results.

**Fields**:

- `document_id: str` - Unique identifier within the batch
- `batch_id: str` - Reference to parent batch
- `file_path: str` - Path to the document file
- `file_name: str` - Original filename
- `file_size: int` - File size in bytes
- `mime_type: str` - Detected MIME type
- `status: DocumentStatus` - Processing status (PENDING, PROCESSING, COMPLETED, FAILED)
- `processing_started_at: Optional[datetime]` - When document processing began
- `processing_completed_at: Optional[datetime]` - When document processing finished
- `evaluation_result: Optional[DisclosureEvaluationResult]` - Evaluation results if successful
- `error_message: Optional[str]` - Error details if processing failed
- `retry_count: int` - Number of retry attempts (default: 0)
- `correlation_id: str` - Document-level correlation ID for logging

**Validation Rules**:

- `document_id` must be unique within the batch
- `file_path` must exist and be readable
- `file_size` must be > 0
- `mime_type` must be supported document type
- `retry_count` cannot exceed maximum retry limit (configurable)
- `status` transitions: PENDING → PROCESSING → (COMPLETED | FAILED)

**State Transitions**:

```
PENDING → PROCESSING (when document processing starts)
PROCESSING → COMPLETED (evaluation successful)
PROCESSING → FAILED (evaluation failed, no more retries)
FAILED → PROCESSING (retry attempt)
```

### BatchResult

Represents the consolidated results of a batch evaluation with summary statistics and individual results.

**Fields**:

- `batch_id: str` - Reference to the batch
- `total_documents: int` - Total number of documents in batch
- `successful_evaluations: int` - Number of successful evaluations
- `failed_evaluations: int` - Number of failed evaluations
- `success_rate: float` - Percentage of successful evaluations (0.0-1.0)
- `processing_duration: timedelta` - Total processing time
- `average_evaluation_time: timedelta` - Average time per document
- `summary_statistics: BatchSummaryStatistics` - Statistical summary of results
- `individual_results: List[DocumentResult]` - Individual document results
- `error_summary: List[DocumentError]` - Summary of errors encountered
- `generated_at: datetime` - When the result was generated

**Validation Rules**:

- `success_rate` must be between 0.0 and 1.0
- `successful_evaluations + failed_evaluations` must equal `total_documents`
- `processing_duration` must be > 0
- `average_evaluation_time` must be > 0 if `successful_evaluations` > 0

### BatchProgress

Represents the current state of batch processing with progress indicators and error information.

**Fields**:

- `batch_id: str` - Reference to the batch
- `current_phase: ProcessingPhase` - Current processing phase (INITIALIZING, DISCOVERING, PROCESSING, AGGREGATING, COMPLETED)
- `total_documents: int` - Total documents to process
- `processed_documents: int` - Documents processed so far
- `successful_documents: int` - Successfully processed documents
- `failed_documents: int` - Failed documents
- `progress_percentage: float` - Completion percentage (0.0-100.0)
- `estimated_completion: Optional[datetime]` - Estimated completion time
- `current_document: Optional[str]` - Currently processing document
- `active_workers: int` - Number of active worker threads
- `error_count: int` - Total number of errors encountered
- `last_updated: datetime` - Last progress update timestamp

**Validation Rules**:

- `progress_percentage` must be between 0.0 and 100.0
- `processed_documents` cannot exceed `total_documents`
- `successful_documents + failed_documents` cannot exceed `processed_documents`
- `active_workers` must be >= 0

## Supporting Types

### BatchStatus (Enum)

```python
class BatchStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIALLY_FAILED = "partially_failed"
```

### DocumentStatus (Enum)

```python
class DocumentStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
```

### ProcessingPhase (Enum)

```python
class ProcessingPhase(Enum):
    INITIALIZING = "initializing"
    DISCOVERING = "discovering"
    PROCESSING = "processing"
    AGGREGATING = "aggregating"
    COMPLETED = "completed"
```

### BatchConfiguration

Configuration settings for batch processing.

**Fields**:

- `max_concurrent_workers: int` - Maximum number of parallel workers (default: 5)
- `max_retry_attempts: int` - Maximum retry attempts per document (default: 3)
- `timeout_seconds: int` - Timeout per document in seconds (default: 300)
- `progress_update_interval: int` - Progress update frequency in documents (default: 10)
- `enable_resumption: bool` - Whether to enable batch resumption (default: True)
- `output_formats: List[str]` - Output formats to generate (default: ["json", "summary"])

### BatchSummaryStatistics

Statistical summary of batch evaluation results.

**Fields**:

- `average_score: float` - Average evaluation score across all documents
- `score_distribution: Dict[int, int]` - Distribution of scores (1-5 scale)
- `most_common_criteria: List[str]` - Most frequently triggered criteria
- `processing_efficiency: float` - Documents processed per minute
- `error_rate: float` - Percentage of failed documents

### DocumentResult

Result of individual document evaluation.

**Fields**:

- `document_id: str` - Document identifier
- `evaluation_result: DisclosureEvaluationResult` - Full evaluation result
- `processing_time: timedelta` - Time taken to process this document
- `success: bool` - Whether evaluation was successful

### DocumentError

Error information for failed document processing.

**Fields**:

- `document_id: str` - Document identifier
- `error_type: str` - Type of error (e.g., "FILE_READ_ERROR", "LLM_API_ERROR", "TIMEOUT")
- `error_message: str` - Detailed error message
- `retry_count: int` - Number of retry attempts
- `occurred_at: datetime` - When the error occurred

## Relationships

### BatchEvaluation ↔ BatchDocument

- **One-to-Many**: One batch contains multiple documents
- **Cascade**: Deleting a batch deletes all associated documents
- **Constraint**: Document batch_id must reference existing batch

### BatchEvaluation ↔ BatchResult

- **One-to-One**: Each batch has exactly one result
- **Constraint**: Result batch_id must reference existing batch

### BatchEvaluation ↔ BatchProgress

- **One-to-One**: Each batch has exactly one progress tracker
- **Constraint**: Progress batch_id must reference existing batch

## Data Validation Rules

### Cross-Entity Validation

- Batch document counts must be consistent across all entities
- Progress indicators must match actual processing state
- Result statistics must match individual document results
- Correlation IDs must be unique within their scope

### Business Logic Validation

- Batch size limits must be enforced (configurable maximum)
- Document file size limits must be enforced
- Processing timeouts must be respected
- Retry limits must be enforced

### Data Integrity Constraints

- All timestamps must be in chronological order
- Status transitions must follow defined state machine
- Numeric fields must be non-negative where appropriate
- String fields must not exceed maximum lengths

## Performance Considerations

### Indexing Strategy

- Primary keys on batch_id and document_id for fast lookups
- Index on status fields for filtering active/failed batches
- Index on timestamps for chronological queries
- Composite index on (batch_id, status) for batch status queries

### Memory Management

- Lazy loading of evaluation results for large batches
- Pagination for batch result retrieval
- Cleanup of temporary files after processing
- Configurable batch size limits to prevent memory exhaustion

### Concurrency Control

- Atomic status updates to prevent race conditions
- Thread-safe progress tracking
- Proper locking for shared state access
- Graceful handling of concurrent batch operations

---

**Data Model Status**: Complete  
**Validation Rules**: Defined  
**Relationships**: Mapped  
**Performance**: Considered  
**Ready for Implementation**: ✅
