"""
Tests for FastAPI health endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from api.main import app

# Create test client
client = TestClient(app)


class TestHealthEndpoints:
    """Test cases for health check endpoints."""

    def test_health_check_success(self):
        """Test successful health check."""
        response = client.get("/v1/health")

        assert response.status_code == 200
        data = response.json()

        assert "status" in data
        assert "timestamp" in data
        assert "version" in data
        assert "uptime" in data
        assert data["version"] == "1.0.0"
        assert data["uptime"] >= 0

    def test_health_check_with_correlation_id(self):
        """Test health check with correlation ID header."""
        response = client.get(
            "/v1/health", headers={"X-Correlation-ID": "test-correlation-123"}
        )

        assert response.status_code == 200
        assert response.headers["X-Correlation-ID"] == "test-correlation-123"

    def test_detailed_health_check(self):
        """Test detailed health check endpoint."""
        response = client.get("/v1/health/detailed")

        assert response.status_code in [200, 503]  # Could be healthy or unhealthy
        data = response.json()

        assert "status" in data
        assert "timestamp" in data
        assert "components" in data
        assert "api" in data["components"]
        assert "llm_providers" in data["components"]
        assert "settings" in data["components"]

    def test_health_check_response_format(self):
        """Test health check response format."""
        response = client.get("/v1/health")

        assert response.status_code == 200
        data = response.json()

        # Check required fields
        required_fields = ["status", "timestamp", "version", "uptime"]
        for field in required_fields:
            assert field in data

        # Check field types
        assert isinstance(data["status"], str)
        assert isinstance(data["timestamp"], str)
        assert isinstance(data["version"], str)
        assert isinstance(data["uptime"], (int, float))

        # Check status values
        assert data["status"] in ["healthy", "unhealthy"]

    def test_health_check_cors_headers(self):
        """Test CORS headers in health check response."""
        response = client.get("/v1/health")

        assert response.status_code == 200
        # CORS headers should be present (configured in main.py)
        assert "access-control-allow-origin" in response.headers

    def test_health_check_performance(self):
        """Test health check response time."""
        import time

        start_time = time.time()
        response = client.get("/v1/health")
        end_time = time.time()

        assert response.status_code == 200
        assert (end_time - start_time) < 1.0  # Should respond within 1 second

    def test_health_check_with_invalid_path(self):
        """Test health check with invalid path."""
        response = client.get("/v1/health/invalid")

        assert response.status_code == 404

    def test_health_check_method_not_allowed(self):
        """Test health check with unsupported HTTP method."""
        response = client.post("/v1/health")

        assert response.status_code == 405  # Method Not Allowed
