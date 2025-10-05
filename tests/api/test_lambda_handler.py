"""
Tests for Lambda handler function.

This module contains tests for the AWS Lambda handler function,
including request validation, response formatting, and error handling.
"""

import json
import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from api.lambda_handler import (
    lambda_handler,
    _determine_operation,
    _handle_evaluation_request,
    _handle_batch_request,
    _handle_status_request,
    _handle_health_check,
    _calculate_overall_score,
    _determine_recommendation,
    _create_error_response,
)
from api.models.requests import LambdaEvaluationRequest, LambdaBatchRequest
from api.models.responses import LambdaEvaluationResponse, LambdaErrorResponse


class TestLambdaHandler:
    """Test cases for the main Lambda handler function."""

    def test_lambda_handler_evaluation_request(self):
        """Test Lambda handler with evaluation request."""
        event = {
            "operation": "evaluation",
            "document_content": "Test document content",
            "context": "Test context",
        }
        context = Mock()
        context.aws_request_id = "test-request-123"

        with patch("api.lambda_handler._handle_evaluation_request") as mock_handle:
            mock_handle.return_value = {
                "statusCode": 200,
                "body": json.dumps({"evaluation_id": "test-123"}),
            }

            result = lambda_handler(event, context)

            assert result["statusCode"] == 200
            mock_handle.assert_called_once()

    def test_lambda_handler_batch_request(self):
        """Test Lambda handler with batch request."""
        event = {
            "operation": "batch",
            "documents": [
                {"document_id": "doc1", "content": "Content 1"},
                {"document_id": "doc2", "content": "Content 2"},
            ],
        }
        context = Mock()
        context.aws_request_id = "test-request-123"

        with patch("api.lambda_handler._handle_batch_request") as mock_handle:
            mock_handle.return_value = {
                "statusCode": 202,
                "body": json.dumps({"batch_id": "batch-123"}),
            }

            result = lambda_handler(event, context)

            assert result["statusCode"] == 202
            mock_handle.assert_called_once()

    def test_lambda_handler_status_request(self):
        """Test Lambda handler with status request."""
        event = {"operation": "status", "batch_id": "batch-123"}
        context = Mock()
        context.aws_request_id = "test-request-123"

        with patch("api.lambda_handler._handle_status_request") as mock_handle:
            mock_handle.return_value = {
                "statusCode": 200,
                "body": json.dumps({"status": "COMPLETED"}),
            }

            result = lambda_handler(event, context)

            assert result["statusCode"] == 200
            mock_handle.assert_called_once()

    def test_lambda_handler_health_check(self):
        """Test Lambda handler with health check request."""
        event = {"operation": "health"}
        context = Mock()
        context.aws_request_id = "test-request-123"

        with patch("api.lambda_handler._handle_health_check") as mock_handle:
            mock_handle.return_value = {
                "statusCode": 200,
                "body": json.dumps({"status": "healthy"}),
            }

            result = lambda_handler(event, context)

            assert result["statusCode"] == 200
            mock_handle.assert_called_once()

    def test_lambda_handler_error(self):
        """Test Lambda handler with error."""
        event = {"operation": "unknown"}
        context = Mock()
        context.aws_request_id = "test-request-123"

        result = lambda_handler(event, context)

        assert (
            result["statusCode"] == 500
        )  # Changed from 400 to 500 for unknown operation
        assert "error_code" in json.loads(result["body"])

    def test_lambda_handler_exception(self):
        """Test Lambda handler with exception."""
        event = None
        context = Mock()
        context.aws_request_id = "test-request-123"

        result = lambda_handler(event, context)

        assert result["statusCode"] == 500
        assert "error_code" in json.loads(result["body"])


class TestDetermineOperation:
    """Test cases for operation determination."""

    def test_determine_operation_api_gateway_health(self):
        """Test operation determination for API Gateway health check."""
        event = {"httpMethod": "GET", "path": "/health"}

        operation = _determine_operation(event)
        assert operation == "health"

    def test_determine_operation_api_gateway_evaluation(self):
        """Test operation determination for API Gateway evaluation."""
        event = {"httpMethod": "POST", "path": "/evaluation"}

        operation = _determine_operation(event)
        assert operation == "evaluation"

    def test_determine_operation_api_gateway_batch(self):
        """Test operation determination for API Gateway batch."""
        event = {"httpMethod": "POST", "path": "/batch"}

        operation = _determine_operation(event)
        assert operation == "batch"

    def test_determine_operation_api_gateway_status(self):
        """Test operation determination for API Gateway status."""
        event = {"httpMethod": "GET", "path": "/batch/batch-123"}

        operation = _determine_operation(event)
        assert operation == "status"

    def test_determine_operation_direct_lambda_evaluation(self):
        """Test operation determination for direct Lambda evaluation."""
        event = {"operation": "evaluation", "document_content": "Test content"}

        operation = _determine_operation(event)
        assert operation == "evaluation"

    def test_determine_operation_direct_lambda_batch(self):
        """Test operation determination for direct Lambda batch."""
        event = {"batch_id": "batch-123", "documents": []}

        operation = _determine_operation(event)
        assert operation == "batch"

    def test_determine_operation_direct_lambda_document_content(self):
        """Test operation determination for direct Lambda with document content."""
        event = {"document_content": "Test content"}

        operation = _determine_operation(event)
        assert operation == "evaluation"

    def test_determine_operation_default_health(self):
        """Test operation determination default to health."""
        event = {}

        operation = _determine_operation(event)
        assert operation == "health"


class TestCalculateOverallScore:
    """Test cases for overall score calculation."""

    def test_calculate_overall_score_empty(self):
        """Test overall score calculation with empty list."""
        score = _calculate_overall_score([])
        assert score == 0.5

    def test_calculate_overall_score_single_criterion(self):
        """Test overall score calculation with single criterion."""

        class MockCriterion:
            def __init__(self, score):
                self.score = score

        criteria = [MockCriterion(3.0)]
        score = _calculate_overall_score(criteria)
        assert score == 0.5  # (3-1)/4 = 0.5

    def test_calculate_overall_score_multiple_criteria(self):
        """Test overall score calculation with multiple criteria."""

        class MockCriterion:
            def __init__(self, score):
                self.score = score

        criteria = [MockCriterion(1.0), MockCriterion(5.0)]
        score = _calculate_overall_score(criteria)
        assert score == 0.5  # (3-1)/4 = 0.5


class TestDetermineRecommendation:
    """Test cases for recommendation determination."""

    def test_determine_recommendation_empty(self):
        """Test recommendation determination with empty list."""
        recommendation = _determine_recommendation([])
        assert recommendation == "UNCLEAR"

    def test_determine_recommendation_non_disclose(self):
        """Test recommendation determination for non-disclosure."""

        class MockCriterion:
            def __init__(self, score):
                self.score = score

        criteria = [MockCriterion(1.0), MockCriterion(1.5)]
        recommendation = _determine_recommendation(criteria)
        assert recommendation == "NON_DISCLOSE"

    def test_determine_recommendation_likely_non_disclose(self):
        """Test recommendation determination for likely non-disclosure."""

        class MockCriterion:
            def __init__(self, score):
                self.score = score

        criteria = [MockCriterion(2.0), MockCriterion(2.5)]
        recommendation = _determine_recommendation(criteria)
        assert recommendation == "LIKELY_NON_DISCLOSE"

    def test_determine_recommendation_unclear(self):
        """Test recommendation determination for unclear."""

        class MockCriterion:
            def __init__(self, score):
                self.score = score

        criteria = [MockCriterion(3.0), MockCriterion(3.5)]
        recommendation = _determine_recommendation(criteria)
        assert recommendation == "UNCLEAR"

    def test_determine_recommendation_likely_disclose(self):
        """Test recommendation determination for likely disclosure."""

        class MockCriterion:
            def __init__(self, score):
                self.score = score

        criteria = [MockCriterion(4.0), MockCriterion(4.5)]
        recommendation = _determine_recommendation(criteria)
        assert recommendation == "LIKELY_DISCLOSE"

    def test_determine_recommendation_disclose(self):
        """Test recommendation determination for disclosure."""

        class MockCriterion:
            def __init__(self, score):
                self.score = score

        criteria = [MockCriterion(5.0), MockCriterion(5.0)]
        recommendation = _determine_recommendation(criteria)
        assert recommendation == "DISCLOSE"


class TestCreateErrorResponse:
    """Test cases for error response creation."""

    def test_create_error_response_validation_error(self):
        """Test error response creation for validation error."""
        response = _create_error_response(
            "VALIDATION_ERROR", "Invalid request format", "test-correlation-id"
        )

        assert response["statusCode"] == 400
        assert "X-Correlation-ID" in response["headers"]

        body = json.loads(response["body"])
        assert body["error_code"] == "VALIDATION_ERROR"
        assert body["error_message"] == "Invalid request format"
        assert body["correlation_id"] == "test-correlation-id"

    def test_create_error_response_processing_error(self):
        """Test error response creation for processing error."""
        response = _create_error_response(
            "PROCESSING_ERROR", "Internal server error", "test-correlation-id"
        )

        assert response["statusCode"] == 500
        assert "X-Correlation-ID" in response["headers"]

        body = json.loads(response["body"])
        assert body["error_code"] == "PROCESSING_ERROR"
        assert body["error_message"] == "Internal server error"
        assert body["correlation_id"] == "test-correlation-id"


class TestHandleHealthCheck:
    """Test cases for health check handling."""

    def test_handle_health_check_success(self):
        """Test successful health check."""
        event = {}
        context = Mock()
        context.aws_request_id = "test-request-123"

        result = _handle_health_check(event, context, "test-correlation-id")

        assert result["statusCode"] == 200
        assert "X-Correlation-ID" in result["headers"]

        body = json.loads(result["body"])
        assert body["status"] == "healthy"
        assert body["correlation_id"] == "test-correlation-id"
        assert "timestamp" in body
        assert "version" in body
        assert "uptime" in body

    def test_handle_health_check_with_context(self):
        """Test health check with context."""
        event = {}
        context = Mock()
        context.aws_request_id = "test-request-123"

        result = _handle_health_check(event, context, "test-correlation-id")

        assert result["statusCode"] == 200
        body = json.loads(result["body"])
        assert body["uptime"] >= 0.0
