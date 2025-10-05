"""
Status checking endpoints for FastAPI.

This module provides request status checking functionality for the API.
"""

import logging
from fastapi import APIRouter, Request, HTTPException, Path
from fastapi.responses import JSONResponse

from api.config.settings import FastAPISettings
from api.models.requests import StatusRequest
from api.models.responses import StatusResponse
from api.models.errors import APIErrorResponse
from api.services.status_service import FastAPIStatusService

# Configure logging
LOGGER = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Initialize settings
settings = FastAPISettings()

# Initialize service
status_service = FastAPIStatusService(settings)


@router.get(
    "/status/{request_id}",
    response_model=StatusResponse,
    summary="Get request status",
    description="Check the status of a specific request",
    tags=["Status"],
)
async def get_request_status(
    request_id: str = Path(..., description="Request identifier"),
    http_request: Request = None,
) -> StatusResponse:
    """
    Get the status of a specific request.

    Args:
        request_id: Request identifier
        http_request: HTTP request object

    Returns:
        StatusResponse: Request status information

    Raises:
        HTTPException: If request not found or status retrieval fails
    """
    try:
        correlation_id = getattr(http_request.state, "correlation_id", "unknown")
        LOGGER.info(
            f"Status requested for ID: {request_id}, correlation ID: {correlation_id}"
        )

        # Get request status
        result = status_service.get_request_status(
            request_id=request_id, correlation_id=correlation_id
        )

        LOGGER.info(f"Request status retrieved for ID: {request_id}")
        return result

    except ValueError as e:
        LOGGER.error(f"Request status retrieval failed: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        LOGGER.error(
            f"Unexpected error during request status retrieval: {str(e)}", exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail="An internal server error occurred during status retrieval",
        )


@router.post(
    "/status",
    response_model=StatusResponse,
    summary="Get request status (POST)",
    description="Check the status of a specific request using POST method",
    tags=["Status"],
)
async def get_request_status_post(
    request: StatusRequest, http_request: Request
) -> StatusResponse:
    """
    Get the status of a specific request using POST method.

    This endpoint provides the same functionality as the GET endpoint
    but uses POST method for cases where request ID might be sensitive
    or when you need to include additional parameters.

    Args:
        request: Status request data
        http_request: HTTP request object

    Returns:
        StatusResponse: Request status information

    Raises:
        HTTPException: If request not found or status retrieval fails
    """
    try:
        correlation_id = getattr(http_request.state, "correlation_id", "unknown")
        LOGGER.info(
            f"Status requested for ID: {request.request_id}, correlation ID: {correlation_id}"
        )

        # Get request status
        result = status_service.get_request_status(
            request_id=request.request_id, correlation_id=correlation_id
        )

        LOGGER.info(f"Request status retrieved for ID: {request.request_id}")
        return result

    except ValueError as e:
        LOGGER.error(f"Request status retrieval failed: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        LOGGER.error(
            f"Unexpected error during request status retrieval: {str(e)}", exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail="An internal server error occurred during status retrieval",
        )
