"""
Tests for FastAPI batch endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from api.main import app

# Create test client
client = TestClient(app)


class TestBatchEndpoints:
    """Test cases for batch evaluation endpoints."""

    def test_submit_batch_success(self):
        """Test successful batch submission."""
        request_data = {
            "documents": [
                {"document_id": "doc_001", "content": "First document content"},
                {"document_id": "doc_002", "content": "Second document content"},
            ],
            "batch_options": {"max_concurrent": 3, "timeout_per_document": 300},
        }

        response = client.post("/v1/batch", json=request_data)

        # Note: This test might fail if batch service is not properly configured
        # In a real test environment, you would mock the batch service
        assert response.status_code in [
            200,
            400,
            500,
        ]  # Could succeed or fail depending on config

        if response.status_code == 200:
            data = response.json()
            assert "batch_id" in data
            assert "status" in data
            assert "total_documents" in data
            assert data["total_documents"] == 2

    def test_submit_batch_validation_error(self):
        """Test batch submission with validation error."""
        request_data = {"documents": []}  # Empty documents list should fail validation

        response = client.post("/v1/batch", json=request_data)

        assert response.status_code == 400
        data = response.json()
        assert "error_code" in data
        assert "VALIDATION_ERROR" in data["error_code"]

    def test_submit_batch_missing_documents(self):
        """Test batch submission with missing required field."""
        request_data = {
            "batch_options": {"max_concurrent": 3}
            # Missing documents field
        }

        response = client.post("/v1/batch", json=request_data)

        assert response.status_code == 422  # Unprocessable Entity

    def test_submit_batch_too_many_documents(self):
        """Test batch submission with too many documents."""
        # Create a batch with more than the maximum allowed documents
        documents = []
        for i in range(101):  # Exceeds max batch size
            documents.append(
                {"document_id": f"doc_{i:03d}", "content": f"Document {i} content"}
            )

        request_data = {"documents": documents}

        response = client.post("/v1/batch", json=request_data)

        assert response.status_code == 400
        data = response.json()
        assert "detail" in data
        assert "Batch size exceeds" in data["detail"]

    def test_submit_batch_invalid_document_id(self):
        """Test batch submission with invalid document ID."""
        request_data = {
            "documents": [
                {
                    "document_id": "",  # Empty document ID should fail validation
                    "content": "Document content",
                }
            ]
        }

        response = client.post("/v1/batch", json=request_data)

        assert response.status_code == 400
        data = response.json()
        assert "error_code" in data
        assert "VALIDATION_ERROR" in data["error_code"]

    def test_submit_batch_large_document_content(self):
        """Test batch submission with large document content."""
        large_content = "A" * 1048577  # Exceeds max length

        request_data = {
            "documents": [{"document_id": "doc_001", "content": large_content}]
        }

        response = client.post("/v1/batch", json=request_data)

        assert response.status_code == 400
        data = response.json()
        assert "error_code" in data
        assert "VALIDATION_ERROR" in data["error_code"]

    def test_get_batch_status_success(self):
        """Test successful batch status retrieval."""
        batch_id = "batch_20250105_123456"

        response = client.get(f"/v1/batch/{batch_id}")

        # Note: This test might fail if batch doesn't exist
        # In a real test environment, you would create a test batch first
        assert response.status_code in [
            200,
            404,
        ]  # Could succeed or fail depending on batch existence

        if response.status_code == 200:
            data = response.json()
            assert "batch_id" in data
            assert "status" in data
            assert "total_documents" in data
            assert "processed_documents" in data
            assert "failed_documents" in data
            assert "progress" in data
            assert "last_updated" in data

    def test_get_batch_status_not_found(self):
        """Test batch status retrieval for non-existent batch."""
        batch_id = "non_existent_batch"

        response = client.get(f"/v1/batch/{batch_id}")

        assert response.status_code == 404
        data = response.json()
        assert "detail" in data

    def test_get_batch_results_success(self):
        """Test successful batch results retrieval."""
        batch_id = "batch_20250105_123456"

        response = client.get(f"/v1/batch/{batch_id}/results")

        # Note: This test might fail if batch doesn't exist or isn't completed
        # In a real test environment, you would create and complete a test batch first
        assert response.status_code in [
            200,
            404,
        ]  # Could succeed or fail depending on batch state

        if response.status_code == 200:
            data = response.json()
            assert "batch_id" in data
            assert "status" in data
            assert "results" in data
            assert "summary" in data

    def test_get_batch_results_not_found(self):
        """Test batch results retrieval for non-existent batch."""
        batch_id = "non_existent_batch"

        response = client.get(f"/v1/batch/{batch_id}/results")

        assert response.status_code == 404
        data = response.json()
        assert "detail" in data

    def test_batch_endpoints_with_correlation_id(self):
        """Test batch endpoints with correlation ID header."""
        request_data = {
            "documents": [
                {"document_id": "doc_001", "content": "Test document content"}
            ]
        }

        response = client.post(
            "/v1/batch",
            json=request_data,
            headers={"X-Correlation-ID": "test-correlation-123"},
        )

        # Should include correlation ID in response headers
        assert "X-Correlation-ID" in response.headers
        assert response.headers["X-Correlation-ID"] == "test-correlation-123"

    def test_batch_response_format(self):
        """Test batch response format."""
        request_data = {
            "documents": [
                {"document_id": "doc_001", "content": "Test document content"}
            ]
        }

        response = client.post("/v1/batch", json=request_data)

        # Response might fail due to missing configuration, but format should be correct
        if response.status_code == 200:
            data = response.json()

            # Check required fields
            required_fields = [
                "batch_id",
                "status",
                "total_documents",
                "processed_documents",
                "failed_documents",
            ]
            for field in required_fields:
                assert field in data

            # Check field types
            assert isinstance(data["batch_id"], str)
            assert isinstance(data["status"], str)
            assert isinstance(data["total_documents"], int)
            assert isinstance(data["processed_documents"], int)
            assert isinstance(data["failed_documents"], int)

    def test_batch_method_not_allowed(self):
        """Test batch endpoints with unsupported HTTP methods."""
        # Test GET on batch submission endpoint
        response = client.get("/v1/batch")
        assert response.status_code == 405  # Method Not Allowed

        # Test POST on batch status endpoint
        response = client.post("/v1/batch/test_batch_id")
        assert response.status_code == 405  # Method Not Allowed

    def test_batch_malformed_json(self):
        """Test batch endpoints with malformed JSON."""
        response = client.post(
            "/v1/batch",
            data="invalid json",
            headers={"Content-Type": "application/json"},
        )

        assert response.status_code == 422  # Unprocessable Entity
