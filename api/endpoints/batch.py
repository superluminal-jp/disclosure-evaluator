"""
Batch evaluation endpoints for FastAPI.

This module provides batch document evaluation functionality for the API.
"""

import logging
from fastapi import APIRouter, Request, HTTPException, Path
from fastapi.responses import JSONResponse

from api.config.settings import FastAPISettings
from api.models.requests import BatchRequest
from api.models.responses import (
    BatchResponse,
    BatchStatusResponse,
    BatchResultsResponse,
)
from api.models.errors import APIErrorResponse
from api.services.batch_service import FastAPIBatchService

# Configure logging
LOGGER = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Initialize settings
settings = FastAPISettings()

# Initialize service
batch_service = FastAPIBatchService(settings)


@router.post(
    "/batch",
    response_model=BatchResponse,
    summary="Submit batch evaluation",
    description="Submit multiple documents for batch evaluation",
    tags=["Batch"],
)
async def submit_batch(request: BatchRequest, http_request: Request) -> BatchResponse:
    """
    Submit multiple documents for batch evaluation.

    Args:
        request: Batch request data
        http_request: HTTP request object

    Returns:
        BatchResponse: Batch submission response

    Raises:
        HTTPException: If batch submission fails
    """
    try:
        correlation_id = getattr(http_request.state, "correlation_id", "unknown")
        LOGGER.info(
            f"Batch request received with {len(request.documents)} documents, correlation ID: {correlation_id}"
        )

        # Validate batch size
        if len(request.documents) > settings.max_batch_size:
            raise HTTPException(
                status_code=400,
                detail=f"Batch size exceeds maximum allowed size of {settings.max_batch_size}",
            )

        # Validate provider configuration if specified
        if request.provider and not settings.is_provider_configured(request.provider):
            raise HTTPException(
                status_code=400,
                detail=f"Provider '{request.provider}' is not properly configured",
            )

        # Create batch
        result = batch_service.create_batch(
            request=request, correlation_id=correlation_id
        )

        LOGGER.info(f"Batch created successfully with ID: {result.batch_id}")
        return result

    except HTTPException:
        raise
    except ValueError as e:
        LOGGER.error(f"Batch creation failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        LOGGER.error(f"Unexpected error during batch creation: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An internal server error occurred during batch creation",
        )


@router.get(
    "/batch/{batch_id}",
    response_model=BatchStatusResponse,
    summary="Get batch status",
    description="Check the status of a batch evaluation",
    tags=["Batch"],
)
async def get_batch_status(
    batch_id: str = Path(..., description="Batch identifier"),
    http_request: Request = None,
) -> BatchStatusResponse:
    """
    Get the status of a batch evaluation.

    Args:
        batch_id: Batch identifier
        http_request: HTTP request object

    Returns:
        BatchStatusResponse: Batch status information

    Raises:
        HTTPException: If batch not found or status retrieval fails
    """
    try:
        correlation_id = getattr(http_request.state, "correlation_id", "unknown")
        LOGGER.info(
            f"Batch status requested for ID: {batch_id}, correlation ID: {correlation_id}"
        )

        # Get batch status
        result = batch_service.get_batch_status(
            batch_id=batch_id, correlation_id=correlation_id
        )

        LOGGER.info(f"Batch status retrieved for ID: {batch_id}")
        return result

    except ValueError as e:
        LOGGER.error(f"Batch status retrieval failed: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        LOGGER.error(
            f"Unexpected error during batch status retrieval: {str(e)}", exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail="An internal server error occurred during status retrieval",
        )


@router.get(
    "/batch/{batch_id}/results",
    response_model=BatchResultsResponse,
    summary="Get batch results",
    description="Retrieve results for a completed batch evaluation",
    tags=["Batch"],
)
async def get_batch_results(
    batch_id: str = Path(..., description="Batch identifier"),
    http_request: Request = None,
) -> BatchResultsResponse:
    """
    Retrieve results for a completed batch evaluation.

    Args:
        batch_id: Batch identifier
        http_request: HTTP request object

    Returns:
        BatchResultsResponse: Batch results

    Raises:
        HTTPException: If batch not found, not completed, or results retrieval fails
    """
    try:
        correlation_id = getattr(http_request.state, "correlation_id", "unknown")
        LOGGER.info(
            f"Batch results requested for ID: {batch_id}, correlation ID: {correlation_id}"
        )

        # Get batch results
        result = batch_service.get_batch_results(
            batch_id=batch_id, correlation_id=correlation_id
        )

        LOGGER.info(f"Batch results retrieved for ID: {batch_id}")
        return result

    except ValueError as e:
        LOGGER.error(f"Batch results retrieval failed: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        LOGGER.error(
            f"Unexpected error during batch results retrieval: {str(e)}", exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail="An internal server error occurred during results retrieval",
        )
