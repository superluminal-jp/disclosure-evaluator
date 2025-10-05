# Research: API 提供: 他のシステムからの評価リクエスト受付

**Date**: 2025-10-05  
**Feature**: API 提供: 他のシステムからの評価リクエスト受付  
**Phase**: 0 - Research & Analysis

---

## Research Findings

### 1. AWS Lambda + FastAPI Integration Patterns

**Decision**: Use AWS Lambda Runtime Interface Client (RIC) with FastAPI for dual deployment support

**Rationale**:

- FastAPI can run in Lambda using Mangum adapter or AWS Lambda Runtime Interface Client
- Single codebase supports both serverless (Lambda) and containerized (FastAPI) deployments
- Existing evaluator.py modules can be imported and used directly in Lambda functions

**Alternatives Considered**:

- Separate Lambda functions vs FastAPI-only: Rejected due to deployment flexibility requirements
- AWS API Gateway + Lambda vs FastAPI: Chose FastAPI for better developer experience and OpenAPI documentation

**Implementation Pattern**:

```python
# lambda_handler.py
from awslambdaric import lambda_handler
from api.main import app

def handler(event, context):
    return lambda_handler(app, event, context)

# api/main.py
from fastapi import FastAPI
from api.endpoints import evaluation, batch, status, health

app = FastAPI()
app.include_router(evaluation.router)
app.include_router(batch.router)
app.include_router(status.router)
app.include_router(health.router)
```

### 2. Authentication & Authorization Strategy

**Decision**: API Key authentication with optional JWT tokens for advanced use cases

**Rationale**:

- API Keys provide simple authentication for external systems
- JWT tokens enable more sophisticated authorization patterns
- AWS IAM integration possible for enterprise customers

**Alternatives Considered**:

- OAuth 2.0: Rejected due to complexity for simple API access
- Basic Auth: Rejected due to security concerns
- No authentication: Rejected due to security requirements

**Implementation**:

- API Key validation middleware
- Rate limiting per API key
- Optional JWT token validation for advanced features

### 3. Request/Response Models with Pydantic

**Decision**: Use Pydantic v2 for request/response validation and serialization

**Rationale**:

- Type safety and automatic validation
- JSON Schema generation for OpenAPI documentation
- Integration with FastAPI automatic documentation
- Performance improvements in Pydantic v2

**Key Models**:

- `EvaluationRequest`: Document content, context, options
- `EvaluationResponse`: Results, correlation ID, metadata
- `BatchRequest`: Multiple documents, batch options
- `BatchResponse`: Batch ID, status, individual results
- `StatusResponse`: Processing status, progress, errors

### 4. Async Processing for Long-Running Evaluations

**Decision**: Asynchronous processing with status tracking for batch operations

**Rationale**:

- Lambda 15-minute timeout requires async processing for large batches
- Status endpoints allow external systems to check progress
- S3 for batch state persistence (reuse existing batch system)

**Implementation Pattern**:

- Immediate response with correlation ID for single documents
- Async processing with status tracking for batch operations
- S3-based state persistence for batch processing
- Webhook notifications (optional) for completion

### 5. Error Handling & Resilience

**Decision**: Structured error responses with correlation IDs and retry guidance

**Rationale**:

- Consistent error format across all endpoints
- Correlation IDs for debugging and support
- Graceful degradation for LLM service failures
- Circuit breaker pattern for external service calls

**Error Categories**:

- Validation errors (400): Invalid request format
- Authentication errors (401): Invalid API key
- Rate limit errors (429): Too many requests
- Service errors (500): Internal processing failures
- Timeout errors (504): Processing timeout

### 6. Rate Limiting Strategy

**Decision**: Token bucket algorithm with per-API-key limits

**Rationale**:

- Prevents abuse and ensures fair usage
- Configurable limits per API key
- Different limits for single vs batch requests
- Graceful handling of rate limit exceeded

**Limits**:

- Single documents: 100 requests/minute per API key
- Batch requests: 10 requests/minute per API key
- Concurrent processing: 5 active batches per API key

### 7. Monitoring & Observability

**Decision**: Structured logging with correlation IDs and CloudWatch integration

**Rationale**:

- Reuse existing correlation ID system from evaluator.py
- CloudWatch Logs for centralized logging
- CloudWatch Metrics for API usage monitoring
- X-Ray tracing for request flow analysis

**Logging Strategy**:

- Request/response logging with correlation IDs
- Performance metrics (latency, throughput)
- Error tracking and alerting
- Business metrics (evaluations per day, success rates)

### 8. Deployment Architecture

**Decision**: Multi-deployment support with infrastructure as code

**Rationale**:

- Lambda for serverless scaling and cost optimization
- FastAPI for containerized deployments (ECS, EKS)
- Terraform for infrastructure management
- SAM for Lambda-specific deployments

**Deployment Options**:

1. **Lambda**: Serverless, auto-scaling, pay-per-use
2. **FastAPI Container**: ECS/EKS deployment for consistent performance
3. **Hybrid**: Lambda for API Gateway, FastAPI for complex processing

### 9. Testing Strategy

**Decision**: Comprehensive testing with contract tests and integration tests

**Rationale**:

- Contract tests ensure API stability
- Integration tests validate end-to-end functionality
- Mock AWS services for reliable testing
- Performance testing for scalability validation

**Test Categories**:

- Unit tests: Individual service functions
- Contract tests: API endpoint contracts
- Integration tests: Full request/response cycles
- Performance tests: Load testing and benchmarking

### 10. Security Considerations

**Decision**: Defense in depth with multiple security layers

**Rationale**:

- API keys for authentication
- Input validation and sanitization
- Rate limiting and DDoS protection
- Secure secrets management
- Audit logging for compliance

**Security Measures**:

- API key rotation and management
- Request size limits and validation
- CORS configuration for web clients
- VPC integration for private deployments
- Encryption in transit and at rest

---

## Research Summary

The research confirms that AWS Lambda + FastAPI integration is well-supported and provides the flexibility needed for both serverless and containerized deployments. The existing evaluator.py modules can be reused with minimal modifications, and the API layer can be implemented following clean architecture principles.

Key technical decisions:

1. **Dual Deployment**: Lambda + FastAPI for maximum flexibility
2. **Authentication**: API Key + optional JWT for different use cases
3. **Async Processing**: Status tracking for long-running operations
4. **Error Handling**: Structured responses with correlation IDs
5. **Monitoring**: CloudWatch integration with existing logging system

The implementation will follow constitutional principles with clean architecture, comprehensive testing, and structured observability.
