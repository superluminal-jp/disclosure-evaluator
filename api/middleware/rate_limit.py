"""
Rate limiting middleware for FastAPI.

This middleware provides rate limiting functionality to prevent API abuse.
"""

import logging
import time
from typing import Callable, Dict, Optional
from fastapi import Request, Response, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware

# Configure logging
LOGGER = logging.getLogger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Middleware for rate limiting API requests.

    This middleware implements a simple in-memory rate limiter
    to prevent API abuse and ensure fair usage.
    """

    def __init__(self, app, requests_per_minute: int = 100):
        """
        Initialize the rate limiting middleware.

        Args:
            app: FastAPI application
            requests_per_minute: Maximum requests per minute per client
        """
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.requests: Dict[str, list] = {}
        self.cleanup_interval = 60  # Clean up old entries every 60 seconds
        self.last_cleanup = time.time()

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process the request with rate limiting.

        Args:
            request: Incoming HTTP request
            call_next: Next middleware/handler in the chain

        Returns:
            Response: HTTP response
        """
        # Skip rate limiting for health checks
        if self._should_skip_rate_limit(request):
            return await call_next(request)

        # Get client identifier
        client_id = self._get_client_id(request)

        # Check rate limit
        if not self._check_rate_limit(client_id):
            LOGGER.warning(f"Rate limit exceeded for client: {client_id}")
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded. Please try again later.",
                headers={"Retry-After": "60"},
            )

        # Record the request
        self._record_request(client_id)

        # Clean up old entries periodically
        self._cleanup_old_entries()

        return await call_next(request)

    def _should_skip_rate_limit(self, request: Request) -> bool:
        """
        Check if rate limiting should be skipped for this request.

        Args:
            request: HTTP request

        Returns:
            bool: True if rate limiting should be skipped
        """
        skip_paths = ["/health", "/docs", "/redoc", "/openapi.json"]

        return any(request.url.path.startswith(path) for path in skip_paths)

    def _get_client_id(self, request: Request) -> str:
        """
        Get a unique identifier for the client.

        Args:
            request: HTTP request

        Returns:
            str: Client identifier
        """
        # Try to get client IP from various headers
        client_ip = (
            request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
            or request.headers.get("X-Real-IP")
            or request.client.host
            if request.client
            else "unknown"
        )

        # Use API key if available for more granular rate limiting
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return f"api_key:{api_key[:8]}"

        return f"ip:{client_ip}"

    def _check_rate_limit(self, client_id: str) -> bool:
        """
        Check if the client has exceeded the rate limit.

        Args:
            client_id: Client identifier

        Returns:
            bool: True if within rate limit
        """
        now = time.time()
        minute_ago = now - 60

        # Get requests for this client
        client_requests = self.requests.get(client_id, [])

        # Remove requests older than 1 minute
        recent_requests = [
            req_time for req_time in client_requests if req_time > minute_ago
        ]

        # Check if within rate limit
        return len(recent_requests) < self.requests_per_minute

    def _record_request(self, client_id: str) -> None:
        """
        Record a request for rate limiting.

        Args:
            client_id: Client identifier
        """
        now = time.time()

        if client_id not in self.requests:
            self.requests[client_id] = []

        self.requests[client_id].append(now)

    def _cleanup_old_entries(self) -> None:
        """
        Clean up old rate limiting entries to prevent memory leaks.
        """
        now = time.time()

        # Only cleanup every cleanup_interval seconds
        if now - self.last_cleanup < self.cleanup_interval:
            return

        self.last_cleanup = now
        minute_ago = now - 60

        # Remove old entries
        for client_id in list(self.requests.keys()):
            self.requests[client_id] = [
                req_time
                for req_time in self.requests[client_id]
                if req_time > minute_ago
            ]

            # Remove empty entries
            if not self.requests[client_id]:
                del self.requests[client_id]

        LOGGER.debug(
            f"Rate limit cleanup completed. Active clients: {len(self.requests)}"
        )
