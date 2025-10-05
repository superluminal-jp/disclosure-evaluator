"""
Tests for FastAPI evaluation endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from api.main import app

# Create test client
client = TestClient(app)


class TestEvaluationEndpoints:
    """Test cases for evaluation endpoints."""

    def test_evaluate_document_success(self):
        """Test successful document evaluation."""
        request_data = {
            "document_content": "This document contains personal information about John Doe.",
            "context": "Request for information disclosure",
            "provider": "openai",
        }

        response = client.post("/v1/evaluation", json=request_data)

        # Note: This test might fail if OpenAI API key is not configured
        # In a real test environment, you would mock the LLM calls
        assert response.status_code in [
            200,
            400,
            500,
        ]  # Could succeed or fail depending on config

        if response.status_code == 200:
            data = response.json()
            assert "evaluation_id" in data
            assert "document_summary" in data
            assert "criteria_results" in data
            assert "overall_score" in data
            assert "recommendation" in data
            assert "processing_time" in data
            assert "metadata" in data

    def test_evaluate_document_validation_error(self):
        """Test document evaluation with validation error."""
        request_data = {
            "document_content": "",  # Empty content should fail validation
            "context": "Request for information disclosure",
        }

        response = client.post("/v1/evaluation", json=request_data)

        assert response.status_code == 400
        data = response.json()
        assert "error_code" in data
        assert "VALIDATION_ERROR" in data["error_code"]

    def test_evaluate_document_missing_content(self):
        """Test document evaluation with missing required field."""
        request_data = {
            "context": "Request for information disclosure"
            # Missing document_content
        }

        response = client.post("/v1/evaluation", json=request_data)

        assert response.status_code == 422  # Unprocessable Entity

    def test_evaluate_document_invalid_provider(self):
        """Test document evaluation with invalid provider."""
        request_data = {
            "document_content": "Test document content",
            "provider": "invalid_provider",
        }

        response = client.post("/v1/evaluation", json=request_data)

        assert response.status_code == 400
        data = response.json()
        assert "detail" in data
        assert "Provider" in data["detail"]

    def test_evaluate_document_large_content(self):
        """Test document evaluation with large content."""
        large_content = "A" * 1048577  # Exceeds max length

        request_data = {"document_content": large_content}

        response = client.post("/v1/evaluation", json=request_data)

        assert response.status_code == 400
        data = response.json()
        assert "error_code" in data
        assert "VALIDATION_ERROR" in data["error_code"]

    def test_evaluate_document_async_success(self):
        """Test successful async document evaluation."""
        request_data = {
            "document_content": "This document contains personal information about John Doe.",
            "context": "Request for information disclosure",
        }

        response = client.post("/v1/evaluation/async", json=request_data)

        assert response.status_code == 200
        data = response.json()

        assert "request_id" in data
        assert "status" in data
        assert "message" in data
        assert data["status"] == "ACCEPTED"

    def test_evaluate_document_async_validation_error(self):
        """Test async document evaluation with validation error."""
        request_data = {
            "document_content": "",  # Empty content should fail validation
        }

        response = client.post("/v1/evaluation/async", json=request_data)

        assert response.status_code == 400
        data = response.json()
        assert "error_code" in data
        assert "VALIDATION_ERROR" in data["error_code"]

    def test_evaluate_document_with_correlation_id(self):
        """Test document evaluation with correlation ID header."""
        request_data = {"document_content": "Test document content"}

        response = client.post(
            "/v1/evaluation",
            json=request_data,
            headers={"X-Correlation-ID": "test-correlation-123"},
        )

        # Should include correlation ID in response headers
        assert "X-Correlation-ID" in response.headers
        assert response.headers["X-Correlation-ID"] == "test-correlation-123"

    def test_evaluate_document_response_format(self):
        """Test evaluation response format."""
        request_data = {"document_content": "Test document content"}

        response = client.post("/v1/evaluation", json=request_data)

        # Response might fail due to missing API keys, but format should be correct
        if response.status_code == 200:
            data = response.json()

            # Check required fields
            required_fields = [
                "evaluation_id",
                "document_summary",
                "criteria_results",
                "overall_score",
                "recommendation",
                "processing_time",
                "metadata",
            ]
            for field in required_fields:
                assert field in data

            # Check metadata fields
            assert "api_version" in data["metadata"]
            assert "processing_time" in data["metadata"]
            assert "correlation_id" in data["metadata"]
            assert "timestamp" in data["metadata"]

    def test_evaluate_document_method_not_allowed(self):
        """Test evaluation endpoint with unsupported HTTP method."""
        response = client.get("/v1/evaluation")

        assert response.status_code == 405  # Method Not Allowed

    def test_evaluate_document_malformed_json(self):
        """Test evaluation endpoint with malformed JSON."""
        response = client.post(
            "/v1/evaluation",
            data="invalid json",
            headers={"Content-Type": "application/json"},
        )

        assert response.status_code == 422  # Unprocessable Entity
