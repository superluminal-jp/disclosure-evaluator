"""
Health check endpoints for FastAPI.

This module provides health check functionality for the API.
"""

import logging
import time
from datetime import datetime
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse

from api.config.settings import FastAPISettings
from api.models.requests import HealthRequest
from api.models.responses import HealthResponse
from api.models.errors import APIErrorResponse

# Configure logging
LOGGER = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Initialize settings
settings = FastAPISettings()

# Track application start time for uptime calculation
start_time = time.time()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check endpoint",
    description="Check API health and availability",
    tags=["Health"],
)
async def health_check(request: Request) -> HealthResponse:
    """
    Check API health and availability.

    Returns:
        HealthResponse: Health status information
    """
    try:
        correlation_id = getattr(request.state, "correlation_id", "unknown")
        LOGGER.info(f"Health check requested with correlation ID: {correlation_id}")

        # Calculate uptime
        uptime = time.time() - start_time

        # Check if API is healthy
        # In a real implementation, you might check:
        # - Database connectivity
        # - External service availability
        # - Memory usage
        # - Disk space
        # etc.

        health_status = "healthy"

        # You could add more sophisticated health checks here
        # For example:
        # - Check if LLM providers are configured
        # - Check if required services are available
        # - Check system resources

        try:
            # Check if at least one LLM provider is configured
            providers_configured = any(
                [
                    settings.is_provider_configured("openai"),
                    settings.is_provider_configured("anthropic"),
                    settings.is_provider_configured("bedrock"),
                    settings.is_provider_configured("bedrock_nova"),
                ]
            )

            if not providers_configured:
                health_status = "unhealthy"
                LOGGER.warning("No LLM providers are properly configured")

        except Exception as e:
            LOGGER.error(f"Error checking provider configuration: {str(e)}")
            health_status = "unhealthy"

        response = HealthResponse(
            status=health_status,
            timestamp=datetime.now().isoformat(),
            version=settings.api_version,
            uptime=uptime,
        )

        LOGGER.info(f"Health check completed: {health_status}")
        return response

    except Exception as e:
        LOGGER.error(f"Health check failed: {str(e)}", exc_info=True)

        # Return unhealthy status on error
        return HealthResponse(
            status="unhealthy",
            timestamp=datetime.now().isoformat(),
            version=settings.api_version,
            uptime=time.time() - start_time,
        )


@router.get(
    "/health/detailed",
    summary="Detailed health check",
    description="Get detailed health information including component status",
    tags=["Health"],
)
async def detailed_health_check(request: Request) -> JSONResponse:
    """
    Get detailed health information including component status.

    Returns:
        JSONResponse: Detailed health information
    """
    try:
        correlation_id = getattr(request.state, "correlation_id", "unknown")
        LOGGER.info(
            f"Detailed health check requested with correlation ID: {correlation_id}"
        )

        # Check various components
        components = {
            "api": {
                "status": "healthy",
                "version": settings.api_version,
                "uptime": time.time() - start_time,
            },
            "llm_providers": {
                "openai": settings.is_provider_configured("openai"),
                "anthropic": settings.is_provider_configured("anthropic"),
                "bedrock": settings.is_provider_configured("bedrock"),
                "bedrock_nova": settings.is_provider_configured("bedrock_nova"),
            },
            "settings": {
                "debug": settings.debug,
                "environment": settings.environment,
                "max_document_size": settings.max_document_size,
                "max_batch_size": settings.max_batch_size,
            },
        }

        # Determine overall health
        overall_healthy = all(
            [
                components["api"]["status"] == "healthy",
                any(components["llm_providers"].values()),
            ]
        )

        response_data = {
            "status": "healthy" if overall_healthy else "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "components": components,
            "correlation_id": correlation_id,
        }

        status_code = 200 if overall_healthy else 503

        return JSONResponse(status_code=status_code, content=response_data)

    except Exception as e:
        LOGGER.error(f"Detailed health check failed: {str(e)}", exc_info=True)

        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "correlation_id": getattr(request.state, "correlation_id", "unknown"),
            },
        )
