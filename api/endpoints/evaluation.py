"""
Evaluation endpoints for FastAPI.

This module provides single document evaluation functionality for the API.
"""

import logging
from fastapi import APIRouter, Request, HTTPException, Depends
from fastapi.responses import JSONResponse

from api.config.settings import FastAPISettings
from api.models.requests import EvaluationRequest
from api.models.responses import EvaluationResponse, AsyncResponse
from api.models.errors import APIErrorResponse
from api.services.evaluation_service import FastAPIEvaluationService

# Configure logging
LOGGER = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Initialize settings
settings = FastAPISettings()

# Initialize service
evaluation_service = FastAPIEvaluationService(settings)


@router.post(
    "/evaluation",
    response_model=EvaluationResponse,
    summary="Evaluate single document",
    description="Submit a single document for disclosure evaluation",
    tags=["Evaluation"],
)
async def evaluate_document(
    request: EvaluationRequest, http_request: Request
) -> EvaluationResponse:
    """
    Evaluate a single document for disclosure compliance.

    Args:
        request: Evaluation request data
        http_request: HTTP request object

    Returns:
        EvaluationResponse: Evaluation result

    Raises:
        HTTPException: If evaluation fails
    """
    try:
        correlation_id = getattr(http_request.state, "correlation_id", "unknown")
        LOGGER.info(
            f"Evaluation request received with correlation ID: {correlation_id}"
        )

        # Validate provider configuration if specified
        if request.provider and not settings.is_provider_configured(request.provider):
            raise HTTPException(
                status_code=400,
                detail=f"Provider '{request.provider}' is not properly configured",
            )

        # Perform evaluation
        result = evaluation_service.evaluate_document(
            request=request, correlation_id=correlation_id
        )

        LOGGER.info(
            f"Evaluation completed successfully for correlation ID: {correlation_id}"
        )
        return result

    except ValueError as e:
        LOGGER.error(f"Evaluation failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        LOGGER.error(f"Unexpected error during evaluation: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An internal server error occurred during evaluation",
        )


@router.post(
    "/evaluation/async",
    response_model=AsyncResponse,
    summary="Submit document for async evaluation",
    description="Submit a document for asynchronous evaluation processing",
    tags=["Evaluation"],
)
async def evaluate_document_async(
    request: EvaluationRequest, http_request: Request
) -> AsyncResponse:
    """
    Submit a document for asynchronous evaluation processing.

    This endpoint accepts the evaluation request and returns immediately
    with a request ID for status tracking.

    Args:
        request: Evaluation request data
        http_request: HTTP request object

    Returns:
        AsyncResponse: Async processing response

    Raises:
        HTTPException: If request submission fails
    """
    try:
        correlation_id = getattr(http_request.state, "correlation_id", "unknown")
        LOGGER.info(
            f"Async evaluation request received with correlation ID: {correlation_id}"
        )

        # Validate provider configuration if specified
        if request.provider and not settings.is_provider_configured(request.provider):
            raise HTTPException(
                status_code=400,
                detail=f"Provider '{request.provider}' is not properly configured",
            )

        # Generate request ID (in a real implementation, this would be stored in a queue)
        import uuid

        request_id = f"eval_{uuid.uuid4().hex[:8]}"

        # In a real implementation, you would:
        # 1. Store the request in a queue (Redis, SQS, etc.)
        # 2. Start a background worker to process it
        # 3. Return the request ID for status tracking

        LOGGER.info(f"Async evaluation request submitted with ID: {request_id}")

        return AsyncResponse(
            request_id=request_id,
            status="ACCEPTED",
            message="Request accepted for processing",
            estimated_completion=None,  # Would be calculated based on queue length
        )

    except HTTPException:
        raise
    except Exception as e:
        LOGGER.error(
            f"Unexpected error during async evaluation submission: {str(e)}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail="An internal server error occurred during request submission",
        )
