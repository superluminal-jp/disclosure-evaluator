"""
Authentication middleware for FastAPI.

This middleware provides authentication functionality for the API.
"""

import logging
from typing import Callable, Optional
from fastapi import Request, Response, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware

# Configure logging
LOGGER = logging.getLogger(__name__)


class AuthMiddleware(BaseHTTPMiddleware):
    """
    Middleware for API authentication.

    This middleware handles API key and JWT token authentication
    for protected endpoints.
    """

    def __init__(self, app, require_auth: bool = False):
        """
        Initialize the authentication middleware.

        Args:
            app: FastAPI application
            require_auth: Whether to require authentication for all endpoints
        """
        super().__init__(app)
        self.require_auth = require_auth

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process the request with authentication.

        Args:
            request: Incoming HTTP request
            call_next: Next middleware/handler in the chain

        Returns:
            Response: HTTP response
        """
        # Skip authentication for health checks and docs
        if self._should_skip_auth(request):
            return await call_next(request)

        # Check if authentication is required
        if not self.require_auth:
            return await call_next(request)

        # Authenticate request
        try:
            auth_result = await self._authenticate_request(request)
            if not auth_result["authenticated"]:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication failed",
                )

            # Add auth info to request state
            request.state.auth_user = auth_result.get("user")
            request.state.auth_method = auth_result.get("method")

        except HTTPException:
            raise
        except Exception as e:
            LOGGER.error(f"Authentication error: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication failed"
            )

        return await call_next(request)

    def _should_skip_auth(self, request: Request) -> bool:
        """
        Check if authentication should be skipped for this request.

        Args:
            request: HTTP request

        Returns:
            bool: True if auth should be skipped
        """
        skip_paths = ["/health", "/docs", "/redoc", "/openapi.json", "/favicon.ico"]

        return any(request.url.path.startswith(path) for path in skip_paths)

    async def _authenticate_request(self, request: Request) -> dict:
        """
        Authenticate the incoming request.

        Args:
            request: HTTP request

        Returns:
            dict: Authentication result
        """
        # Check for API key in headers
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return await self._authenticate_api_key(api_key)

        # Check for Bearer token in Authorization header
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
            return await self._authenticate_jwt_token(token)

        # No authentication provided
        return {"authenticated": False, "error": "No authentication provided"}

    async def _authenticate_api_key(self, api_key: str) -> dict:
        """
        Authenticate using API key.

        Args:
            api_key: API key from request

        Returns:
            dict: Authentication result
        """
        # In a real implementation, you would:
        # 1. Validate the API key against a database
        # 2. Check if the key is active and not expired
        # 3. Retrieve user information associated with the key

        # For now, we'll implement a simple validation
        # In production, use proper API key management
        valid_api_keys = {
            "test-api-key-123": {"user": "test-user", "permissions": ["read", "write"]},
            "admin-api-key-456": {"user": "admin", "permissions": ["admin"]},
        }

        if api_key in valid_api_keys:
            user_info = valid_api_keys[api_key]
            LOGGER.info(
                f"API key authentication successful for user: {user_info['user']}"
            )
            return {
                "authenticated": True,
                "user": user_info["user"],
                "method": "api_key",
                "permissions": user_info["permissions"],
            }

        LOGGER.warning(f"Invalid API key provided: {api_key[:8]}...")
        return {"authenticated": False, "error": "Invalid API key"}

    async def _authenticate_jwt_token(self, token: str) -> dict:
        """
        Authenticate using JWT token.

        Args:
            token: JWT token from request

        Returns:
            dict: Authentication result
        """
        # In a real implementation, you would:
        # 1. Verify the JWT signature
        # 2. Check token expiration
        # 3. Validate the token claims
        # 4. Retrieve user information from the token

        # For now, we'll implement a simple validation
        # In production, use proper JWT validation
        try:
            # This is a mock implementation
            # In reality, you would use a JWT library like PyJWT
            if token == "valid-jwt-token":
                LOGGER.info("JWT token authentication successful")
                return {
                    "authenticated": True,
                    "user": "jwt-user",
                    "method": "jwt",
                    "permissions": ["read", "write"],
                }

            LOGGER.warning("Invalid JWT token provided")
            return {"authenticated": False, "error": "Invalid JWT token"}

        except Exception as e:
            LOGGER.error(f"JWT token validation error: {str(e)}")
            return {"authenticated": False, "error": "Token validation failed"}
