"""
Tests for FastAPI status endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from api.main import app

# Create test client
client = TestClient(app)


class TestStatusEndpoints:
    """Test cases for status checking endpoints."""

    def test_get_request_status_success(self):
        """Test successful request status retrieval."""
        request_id = "eval_20250105_123456"

        response = client.get(f"/v1/status/{request_id}")

        # Note: This test uses mock status service, so it should always succeed
        assert response.status_code == 200
        data = response.json()

        assert "request_id" in data
        assert "status" in data
        assert "progress" in data
        assert "message" in data
        assert "last_updated" in data
        assert data["request_id"] == request_id

    def test_get_request_status_with_correlation_id(self):
        """Test request status retrieval with correlation ID header."""
        request_id = "eval_20250105_123456"

        response = client.get(
            f"/v1/status/{request_id}",
            headers={"X-Correlation-ID": "test-correlation-123"},
        )

        assert response.status_code == 200
        assert "X-Correlation-ID" in response.headers
        assert response.headers["X-Correlation-ID"] == "test-correlation-123"

    def test_get_request_status_response_format(self):
        """Test request status response format."""
        request_id = "eval_20250105_123456"

        response = client.get(f"/v1/status/{request_id}")

        assert response.status_code == 200
        data = response.json()

        # Check required fields
        required_fields = [
            "request_id",
            "status",
            "progress",
            "message",
            "last_updated",
        ]
        for field in required_fields:
            assert field in data

        # Check field types
        assert isinstance(data["request_id"], str)
        assert isinstance(data["status"], str)
        assert isinstance(data["progress"], (int, float))
        assert isinstance(data["message"], str)
        assert isinstance(data["last_updated"], str)

        # Check progress range
        assert 0.0 <= data["progress"] <= 100.0

        # Check status values
        valid_statuses = ["QUEUED", "PROCESSING", "COMPLETED", "FAILED", "CANCELLED"]
        assert data["status"] in valid_statuses

    def test_post_request_status_success(self):
        """Test successful request status retrieval using POST method."""
        request_data = {"request_id": "eval_20250105_123456"}

        response = client.post("/v1/status", json=request_data)

        assert response.status_code == 200
        data = response.json()

        assert "request_id" in data
        assert "status" in data
        assert "progress" in data
        assert "message" in data
        assert "last_updated" in data
        assert data["request_id"] == request_data["request_id"]

    def test_post_request_status_validation_error(self):
        """Test request status retrieval with validation error."""
        request_data = {"request_id": ""}  # Empty request ID should fail validation

        response = client.post("/v1/status", json=request_data)

        assert response.status_code == 400
        data = response.json()
        assert "error_code" in data
        assert "VALIDATION_ERROR" in data["error_code"]

    def test_post_request_status_missing_field(self):
        """Test request status retrieval with missing required field."""
        request_data = {
            # Missing request_id field
        }

        response = client.post("/v1/status", json=request_data)

        assert response.status_code == 422  # Unprocessable Entity

    def test_get_request_status_different_request_types(self):
        """Test request status retrieval for different request types."""
        # Test batch request
        batch_request_id = "batch_20250105_123456"
        response = client.get(f"/v1/status/{batch_request_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["request_id"] == batch_request_id

        # Test evaluation request
        eval_request_id = "eval_20250105_123456"
        response = client.get(f"/v1/status/{eval_request_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["request_id"] == eval_request_id

        # Test unknown request type
        unknown_request_id = "unknown_20250105_123456"
        response = client.get(f"/v1/status/{unknown_request_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["request_id"] == unknown_request_id

    def test_status_endpoints_with_correlation_id(self):
        """Test status endpoints with correlation ID header."""
        request_id = "eval_20250105_123456"

        response = client.get(
            f"/v1/status/{request_id}",
            headers={"X-Correlation-ID": "test-correlation-123"},
        )

        assert response.status_code == 200
        assert "X-Correlation-ID" in response.headers
        assert response.headers["X-Correlation-ID"] == "test-correlation-123"

    def test_status_endpoints_method_not_allowed(self):
        """Test status endpoints with unsupported HTTP methods."""
        # Test PUT on GET endpoint
        response = client.put("/v1/status/test_request_id")
        assert response.status_code == 405  # Method Not Allowed

        # Test DELETE on POST endpoint
        response = client.delete("/v1/status")
        assert response.status_code == 405  # Method Not Allowed

    def test_status_malformed_json(self):
        """Test status endpoints with malformed JSON."""
        response = client.post(
            "/v1/status",
            data="invalid json",
            headers={"Content-Type": "application/json"},
        )

        assert response.status_code == 422  # Unprocessable Entity

    def test_status_request_id_validation(self):
        """Test request ID validation."""
        # Test with very long request ID
        long_request_id = "a" * 101  # Exceeds max length
        response = client.get(f"/v1/status/{long_request_id}")
        # Should still work as path parameter validation is different

        # Test with special characters
        special_request_id = "request_id_with-special.chars"
        response = client.get(f"/v1/status/{special_request_id}")
        assert response.status_code == 200

    def test_status_response_consistency(self):
        """Test that status responses are consistent."""
        request_id = "eval_20250105_123456"

        # Make multiple requests and check consistency
        response1 = client.get(f"/v1/status/{request_id}")
        response2 = client.get(f"/v1/status/{request_id}")

        assert response1.status_code == 200
        assert response2.status_code == 200

        data1 = response1.json()
        data2 = response2.json()

        # Request ID should be consistent
        assert data1["request_id"] == data2["request_id"]
        assert data1["request_id"] == request_id

        # Status should be consistent (mock service returns same status)
        assert data1["status"] == data2["status"]
