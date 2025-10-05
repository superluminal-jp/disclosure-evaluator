#!/usr/bin/env python3
"""
Test script for Lambda function.

This script provides a simple way to test the Lambda function
locally before deployment.
"""

import json
import sys
import os
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from api.lambda_handler import lambda_handler


class MockContext:
    """Mock Lambda context for testing."""

    def __init__(self, request_id="test-request-123"):
        self.aws_request_id = request_id
        self.function_name = "disclosure-evaluator"
        self.function_version = "1.0.0"
        self.memory_limit_in_mb = 2048
        self.remaining_time_in_millis = 900000


def test_health_check():
    """Test health check endpoint."""
    print("Testing health check...")

    event = {"operation": "health"}
    context = MockContext()

    result = lambda_handler(event, context)

    print(f"Status Code: {result['statusCode']}")
    print(f"Response: {result['body']}")
    print()


def test_evaluation_request():
    """Test evaluation request."""
    print("Testing evaluation request...")

    event = {
        "operation": "evaluation",
        "document_content": "This document contains personal information about John Doe, including his address and phone number.",
        "context": "Request for information disclosure under FOI Act",
        "provider": "openai",
    }
    context = MockContext()

    result = lambda_handler(event, context)

    print(f"Status Code: {result['statusCode']}")
    print(f"Response: {result['body']}")
    print()


def test_batch_request():
    """Test batch request."""
    print("Testing batch request...")

    event = {
        "operation": "batch",
        "documents": [
            {
                "document_id": "doc_001",
                "content": "First document with personal information",
                "context": "Test batch document 1",
            },
            {
                "document_id": "doc_002",
                "content": "Second document with corporate information",
                "context": "Test batch document 2",
            },
        ],
        "batch_options": {"max_concurrent": 2, "timeout_per_document": 300},
    }
    context = MockContext()

    result = lambda_handler(event, context)

    print(f"Status Code: {result['statusCode']}")
    print(f"Response: {result['body']}")
    print()


def test_status_request():
    """Test status request."""
    print("Testing status request...")

    event = {"operation": "status", "batch_id": "batch_20250105_123456"}
    context = MockContext()

    result = lambda_handler(event, context)

    print(f"Status Code: {result['statusCode']}")
    print(f"Response: {result['body']}")
    print()


def test_api_gateway_event():
    """Test API Gateway event format."""
    print("Testing API Gateway event format...")

    event = {
        "httpMethod": "POST",
        "path": "/evaluation",
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(
            {"document_content": "Test document content", "context": "Test context"}
        ),
    }
    context = MockContext()

    result = lambda_handler(event, context)

    print(f"Status Code: {result['statusCode']}")
    print(f"Response: {result['body']}")
    print()


def test_error_handling():
    """Test error handling."""
    print("Testing error handling...")

    # Test invalid operation
    event = {"operation": "invalid_operation"}
    context = MockContext()

    result = lambda_handler(event, context)

    print(f"Status Code: {result['statusCode']}")
    print(f"Response: {result['body']}")
    print()

    # Test malformed request
    event = {
        "operation": "evaluation",
        "document_content": "",  # Empty content should cause validation error
    }
    context = MockContext()

    result = lambda_handler(event, context)

    print(f"Status Code: {result['statusCode']}")
    print(f"Response: {result['body']}")
    print()


def main():
    """Run all tests."""
    print("=" * 60)
    print("Lambda Function Test Suite")
    print("=" * 60)
    print()

    # Set up environment variables for testing
    os.environ.setdefault("LOG_LEVEL", "INFO")
    os.environ.setdefault("LOG_FORMAT", "text")
    os.environ.setdefault("ENABLE_STRUCTURED_LOGGING", "false")

    try:
        test_health_check()
        test_evaluation_request()
        test_batch_request()
        test_status_request()
        test_api_gateway_event()
        test_error_handling()

        print("=" * 60)
        print("All tests completed!")
        print("=" * 60)

    except Exception as e:
        print(f"Test failed with error: {str(e)}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
