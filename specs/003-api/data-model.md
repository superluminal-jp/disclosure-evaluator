# Data Model: API 提供: 他のシステムからの評価リクエスト受付

**Date**: 2025-10-05  
**Feature**: API 提供: 他のシステムからの評価リクエスト受付  
**Phase**: 1 - Design & Contracts

---

## Core Entities

### APIRequest

Represents an incoming evaluation request from an external system.

**Fields**:

- `request_id: str` - Unique identifier for the request
- `api_key: str` - Authentication key for the external system
- `correlation_id: str` - Request tracking identifier
- `timestamp: datetime` - Request submission time
- `client_info: ClientInfo` - Information about the requesting system
- `request_type: RequestType` - Single document or batch request
- `status: RequestStatus` - Current processing status

**Validation Rules**:

- `request_id` must be unique across all requests
- `api_key` must be valid and active
- `correlation_id` follows format: `api_YYYYMMDD_HHMMSS_XXXX`
- `timestamp` must be within last 24 hours for processing

**State Transitions**:

```
PENDING → PROCESSING → COMPLETED
PENDING → PROCESSING → FAILED
PENDING → VALIDATION_FAILED
```

### APIResponse

Represents the response sent back to external systems.

**Fields**:

- `request_id: str` - Matches the original request ID
- `correlation_id: str` - Request tracking identifier
- `status: ResponseStatus` - Success, error, or processing status
- `timestamp: datetime` - Response generation time
- `data: Optional[Dict]` - Response payload (results, errors, etc.)
- `metadata: ResponseMetadata` - Additional response information
- `errors: List[APIError]` - Any errors that occurred

**Validation Rules**:

- `request_id` must match the original request
- `status` must be valid ResponseStatus enum value
- `data` is required for successful responses
- `errors` is required for error responses

### EvaluationRequest

Represents a single document evaluation request.

**Fields**:

- `document_content: str` - The text content to evaluate
- `context: Optional[str]` - Additional context information
- `output_text: Optional[str]` - Desired output text format
- `provider: Optional[str]` - LLM provider preference
- `options: EvaluationOptions` - Additional evaluation parameters

**Validation Rules**:

- `document_content` must be non-empty and ≤1MB
- `context` is optional but ≤500KB if provided
- `output_text` is optional but ≤1KB if provided
- `provider` must be valid LLM provider if specified

### EvaluationResponse

Represents the result of a single document evaluation.

**Fields**:

- `evaluation_id: str` - Unique identifier for this evaluation
- `document_summary: str` - Brief summary of the document
- `criteria_results: List[CriterionResult]` - Results for each criterion
- `overall_score: float` - Overall disclosure score (0.0-1.0)
- `recommendation: DisclosureRecommendation` - Final recommendation
- `processing_time: float` - Time taken for evaluation in seconds
- `metadata: EvaluationMetadata` - Additional evaluation information

**Validation Rules**:

- `evaluation_id` must be unique
- `overall_score` must be between 0.0 and 1.0
- `criteria_results` must contain all required criteria
- `processing_time` must be positive

### BatchRequest

Represents a batch evaluation request with multiple documents.

**Fields**:

- `batch_id: str` - Unique identifier for the batch
- `documents: List[BatchDocument]` - List of documents to evaluate
- `batch_options: BatchOptions` - Batch processing configuration
- `callback_url: Optional[str]` - Webhook URL for completion notification
- `priority: BatchPriority` - Processing priority level

**Validation Rules**:

- `batch_id` must be unique
- `documents` must contain 1-100 documents
- Each document must have unique `document_id`
- `callback_url` must be valid URL if provided

### BatchResponse

Represents the response for a batch evaluation request.

**Fields**:

- `batch_id: str` - Matches the original batch ID
- `status: BatchStatus` - Current batch processing status
- `total_documents: int` - Total number of documents in batch
- `processed_documents: int` - Number of documents processed
- `failed_documents: int` - Number of documents that failed
- `estimated_completion: Optional[datetime]` - Estimated completion time
- `results: Optional[List[EvaluationResponse]]` - Individual results (if completed)

**Validation Rules**:

- `batch_id` must match the original request
- `total_documents` must equal sum of processed + failed
- `results` only present when status is COMPLETED

### APIStatus

Represents the current status of an API request.

**Fields**:

- `request_id: str` - Original request identifier
- `status: ProcessingStatus` - Current processing status
- `progress: float` - Progress percentage (0.0-100.0)
- `message: str` - Human-readable status message
- `last_updated: datetime` - Last status update time
- `estimated_completion: Optional[datetime]` - Estimated completion time
- `errors: List[ProcessingError]` - Any processing errors

**Validation Rules**:

- `progress` must be between 0.0 and 100.0
- `last_updated` must be more recent than request timestamp
- `errors` only present when status indicates failure

### APIAuthentication

Represents authentication information for external systems.

**Fields**:

- `api_key: str` - Unique API key identifier
- `client_name: str` - Name of the external system
- `permissions: List[Permission]` - Granted permissions
- `rate_limits: RateLimits` - Rate limiting configuration
- `created_at: datetime` - API key creation time
- `expires_at: Optional[datetime]` - API key expiration time
- `is_active: bool` - Whether the key is currently active

**Validation Rules**:

- `api_key` must be unique and secure
- `client_name` must be non-empty
- `permissions` must contain at least one valid permission
- `expires_at` must be in the future if specified

### APIError

Represents error information for failed API requests.

**Fields**:

- `error_code: str` - Machine-readable error code
- `error_message: str` - Human-readable error message
- `error_type: ErrorType` - Category of error
- `correlation_id: str` - Request correlation ID
- `timestamp: datetime` - Error occurrence time
- `details: Optional[Dict]` - Additional error context
- `resolution_guidance: Optional[str]` - Suggested resolution steps

**Validation Rules**:

- `error_code` must be valid ErrorType enum value
- `error_message` must be non-empty
- `correlation_id` must match the original request

---

## Supporting Types

### ClientInfo

Information about the requesting external system.

**Fields**:

- `client_name: str` - Name of the external system
- `client_version: str` - Version of the external system
- `user_agent: str` - User agent string
- `ip_address: str` - Source IP address

### ResponseMetadata

Additional metadata for API responses.

**Fields**:

- `api_version: str` - API version used
- `processing_time: float` - Server processing time
- `rate_limit_remaining: int` - Remaining rate limit
- `rate_limit_reset: datetime` - Rate limit reset time

### EvaluationOptions

Configuration options for evaluation processing.

**Fields**:

- `parallel_processing: bool` - Enable parallel evaluation
- `max_workers: int` - Maximum parallel workers
- `timeout: int` - Processing timeout in seconds
- `retry_attempts: int` - Number of retry attempts

### BatchOptions

Configuration options for batch processing.

**Fields**:

- `max_concurrent: int` - Maximum concurrent evaluations
- `timeout_per_document: int` - Timeout per document in seconds
- `retry_failed: bool` - Whether to retry failed documents
- `notification_webhook: Optional[str]` - Webhook for notifications

### CriterionResult

Result for a single evaluation criterion.

**Fields**:

- `criterion_id: str` - Criterion identifier
- `criterion_name: str` - Human-readable criterion name
- `score: float` - Score for this criterion (1.0-5.0)
- `reasoning: str` - Detailed reasoning for the score
- `steps: List[EvaluationStep]` - Individual evaluation steps

### BatchDocument

Individual document within a batch request.

**Fields**:

- `document_id: str` - Unique identifier for this document
- `content: str` - Document content
- `context: Optional[str]` - Additional context
- `metadata: Optional[Dict]` - Document metadata

---

## Enums

### RequestType

- `SINGLE_DOCUMENT` - Single document evaluation
- `BATCH_DOCUMENTS` - Batch document evaluation

### RequestStatus

- `PENDING` - Request received, awaiting processing
- `PROCESSING` - Currently being processed
- `COMPLETED` - Processing completed successfully
- `FAILED` - Processing failed
- `VALIDATION_FAILED` - Request validation failed

### ResponseStatus

- `SUCCESS` - Request processed successfully
- `ERROR` - Request failed with error
- `PROCESSING` - Request is still being processed
- `VALIDATION_ERROR` - Request validation failed

### BatchStatus

- `PENDING` - Batch received, awaiting processing
- `PROCESSING` - Batch is being processed
- `COMPLETED` - All documents processed successfully
- `PARTIALLY_COMPLETED` - Some documents processed, some failed
- `FAILED` - Batch processing failed

### ProcessingStatus

- `QUEUED` - Request queued for processing
- `PROCESSING` - Currently being processed
- `COMPLETED` - Processing completed
- `FAILED` - Processing failed
- `CANCELLED` - Processing cancelled

### ErrorType

- `VALIDATION_ERROR` - Request validation failed
- `AUTHENTICATION_ERROR` - Authentication failed
- `AUTHORIZATION_ERROR` - Authorization failed
- `RATE_LIMIT_ERROR` - Rate limit exceeded
- `PROCESSING_ERROR` - Processing failed
- `TIMEOUT_ERROR` - Processing timeout
- `SERVICE_ERROR` - External service error

### BatchPriority

- `LOW` - Low priority processing
- `NORMAL` - Normal priority processing
- `HIGH` - High priority processing
- `URGENT` - Urgent priority processing

### DisclosureRecommendation

- `DISCLOSE` - Recommend disclosure
- `LIKELY_DISCLOSE` - Likely to disclose
- `UNCLEAR` - Unclear, requires human review
- `LIKELY_NON_DISCLOSE` - Likely not to disclose
- `NON_DISCLOSE` - Recommend non-disclosure

---

## Relationships

### APIRequest → APIResponse

- One-to-one relationship
- Response always matches request ID

### APIRequest → EvaluationRequest

- One-to-one for single document requests
- One-to-many for batch requests

### BatchRequest → BatchResponse

- One-to-one relationship
- Response tracks batch processing status

### APIRequest → APIStatus

- One-to-many relationship
- Multiple status updates during processing

### APIAuthentication → APIRequest

- One-to-many relationship
- Multiple requests per API key

### APIRequest → APIError

- One-to-many relationship
- Multiple errors possible per request

---

## Validation Rules Summary

1. **Request Validation**:

   - All required fields must be present
   - String fields must be non-empty
   - Numeric fields must be within valid ranges
   - Timestamps must be valid ISO 8601 format

2. **Authentication Validation**:

   - API key must be valid and active
   - Rate limits must not be exceeded
   - Permissions must be sufficient for requested operation

3. **Content Validation**:

   - Document content must be within size limits
   - Content must be valid UTF-8 text
   - Special characters must be properly encoded

4. **Response Validation**:
   - Response must match request ID
   - Status must be consistent with processing state
   - Error responses must include error details
