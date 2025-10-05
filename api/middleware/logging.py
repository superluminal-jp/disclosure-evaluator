"""
Logging middleware for FastAPI.

This middleware provides request/response logging functionality.
"""

import logging
import time
import uuid
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

# Configure logging
LOGGER = logging.getLogger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for logging HTTP requests and responses.

    This middleware logs all incoming requests and outgoing responses
    with correlation IDs for request tracking.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process the request and response with logging.

        Args:
            request: Incoming HTTP request
            call_next: Next middleware/handler in the chain

        Returns:
            Response: HTTP response
        """
        # Generate correlation ID if not present
        correlation_id = request.headers.get("X-Correlation-ID")
        if not correlation_id:
            correlation_id = f"api_{uuid.uuid4().hex[:8]}"

        # Add correlation ID to request state
        request.state.correlation_id = correlation_id

        # Log request
        start_time = time.time()
        LOGGER.info(
            f"Request started: {request.method} {request.url.path} "
            f"[correlation_id: {correlation_id}]"
        )

        # Process request
        try:
            response = await call_next(request)

            # Calculate processing time
            process_time = time.time() - start_time

            # Log response
            LOGGER.info(
                f"Request completed: {request.method} {request.url.path} "
                f"-> {response.status_code} "
                f"[{process_time:.3f}s] [correlation_id: {correlation_id}]"
            )

            # Add correlation ID to response headers
            response.headers["X-Correlation-ID"] = correlation_id

            return response

        except Exception as e:
            # Calculate processing time
            process_time = time.time() - start_time

            # Log error
            LOGGER.error(
                f"Request failed: {request.method} {request.url.path} "
                f"-> ERROR: {str(e)} "
                f"[{process_time:.3f}s] [correlation_id: {correlation_id}]",
                exc_info=True,
            )

            # Re-raise the exception
            raise
