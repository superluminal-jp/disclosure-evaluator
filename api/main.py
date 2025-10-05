"""
FastAPI application entry point for document disclosure evaluation API.

This module creates and configures the FastAPI application with all
endpoints, middleware, and error handlers.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

# Add the parent directory to the path to import evaluator modules
sys.path.append(str(Path(__file__).parent.parent))

from api.config.settings import FastAPISettings
from api.endpoints import evaluation, batch, status, health
from api.middleware.logging import LoggingMiddleware
from api.middleware.auth import AuthMiddleware
from api.middleware.rate_limit import RateLimitMiddleware
from api.models.errors import APIErrorResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
LOGGER = logging.getLogger(__name__)

# Initialize settings
settings = FastAPISettings()

# Create FastAPI application
app = FastAPI(
    title="Disclosure Evaluator API",
    description="API for external systems to submit document evaluation requests",
    version="1.0.0",
    docs_url="/docs" if settings.enable_docs else None,
    redoc_url="/redoc" if settings.enable_docs else None,
    openapi_url="/openapi.json" if settings.enable_docs else None,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add trusted host middleware
app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.allowed_hosts)

# Add custom middleware
app.add_middleware(LoggingMiddleware)
app.add_middleware(AuthMiddleware)
app.add_middleware(RateLimitMiddleware)

# Include routers
app.include_router(health.router, prefix="/v1", tags=["Health"])
app.include_router(evaluation.router, prefix="/v1", tags=["Evaluation"])
app.include_router(batch.router, prefix="/v1", tags=["Batch"])
app.include_router(status.router, prefix="/v1", tags=["Status"])


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Handle request validation errors."""
    LOGGER.error(f"Validation error: {exc}")

    return JSONResponse(
        status_code=400,
        content=APIErrorResponse(
            error_code="VALIDATION_ERROR",
            error_message="Invalid request format",
            error_type="VALIDATION_ERROR",
            correlation_id=getattr(request.state, "correlation_id", "unknown"),
            details={"validation_errors": exc.errors()},
            resolution_guidance="Please check the request format and try again",
        ).model_dump(),
    )


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(
    request: Request, exc: StarletteHTTPException
) -> JSONResponse:
    """Handle HTTP exceptions."""
    LOGGER.error(f"HTTP error {exc.status_code}: {exc.detail}")

    return JSONResponse(
        status_code=exc.status_code,
        content=APIErrorResponse(
            error_code=f"HTTP_{exc.status_code}",
            error_message=str(exc.detail),
            error_type="PROCESSING_ERROR",
            correlation_id=getattr(request.state, "correlation_id", "unknown"),
            resolution_guidance="Please check the request and try again",
        ).model_dump(),
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle general exceptions."""
    LOGGER.error(f"Unexpected error: {exc}", exc_info=True)

    return JSONResponse(
        status_code=500,
        content=APIErrorResponse(
            error_code="INTERNAL_ERROR",
            error_message="An internal server error occurred",
            error_type="SERVICE_ERROR",
            correlation_id=getattr(request.state, "correlation_id", "unknown"),
            timestamp=datetime.now().isoformat(),
            resolution_guidance="Please try again later or contact support",
        ).model_dump(),
    )


@app.on_event("startup")
async def startup_event() -> None:
    """Application startup event."""
    LOGGER.info("FastAPI application starting up...")
    LOGGER.info(f"API version: {app.version}")
    LOGGER.info(f"Debug mode: {settings.debug}")
    LOGGER.info(f"Allowed origins: {settings.allowed_origins}")


@app.on_event("shutdown")
async def shutdown_event() -> None:
    """Application shutdown event."""
    LOGGER.info("FastAPI application shutting down...")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level="info",
    )
