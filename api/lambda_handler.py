#!/usr/bin/env python3
"""
AWS Lambda handler for disclosure evaluator.

This module provides the main entry point for AWS Lambda function execution,
handling both single document evaluation and batch processing requests.
"""

import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path

# Add the parent directory to the path to import evaluator modules
sys.path.append(str(Path(__file__).parent.parent))

from evaluator import (
    DisclosureEvaluator,
    BatchEvaluator,
    BatchConfiguration,
    DocumentInput,
    BatchStatus,
    BatchResult,
    BatchProgress,
    DisclosureEvaluationResult,
)
from api.models.requests import (
    LambdaEvaluationRequest,
    LambdaBatchRequest,
    LambdaStatusRequest,
)
from api.models.responses import (
    LambdaEvaluationResponse,
    LambdaBatchResponse,
    LambdaStatusResponse,
    LambdaErrorResponse,
    EvaluationMetadata,
)
from api.services.evaluation_service import EvaluationService
from api.services.batch_service import BatchService
from api.services.status_service import StatusService
from api.config.settings import LambdaSettings

# Configure logging for Lambda
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Initialize settings
settings = LambdaSettings()


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Main AWS Lambda handler function.

    Args:
        event: Lambda event object containing request data
        context: Lambda context object with runtime information

    Returns:
        Dict containing response data and status code
    """
    try:
        # Extract request information
        request_id = (
            context.aws_request_id
            if context
            else f"lambda_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        correlation_id = f"lambda_{request_id}"

        logger.info(f"Lambda function invoked with request ID: {request_id}")
        logger.info(f"Event: {json.dumps(event, default=str)}")

        # Parse the event to determine the operation type
        operation = _determine_operation(event)

        if operation == "evaluation":
            return _handle_evaluation_request(event, context, correlation_id)
        elif operation == "batch":
            return _handle_batch_request(event, context, correlation_id)
        elif operation == "status":
            return _handle_status_request(event, context, correlation_id)
        elif operation == "health":
            return _handle_health_check(event, context, correlation_id)
        else:
            return _create_error_response(
                "INVALID_OPERATION", f"Unknown operation: {operation}", correlation_id
            )

    except Exception as e:
        logger.error(f"Lambda handler error: {str(e)}", exc_info=True)
        return _create_error_response(
            "LAMBDA_ERROR",
            f"Internal server error: {str(e)}",
            getattr(context, "aws_request_id", "unknown"),
        )


def _determine_operation(event: Dict[str, Any]) -> str:
    """
    Determine the operation type from the Lambda event.

    Args:
        event: Lambda event object

    Returns:
        str: Operation type (evaluation, batch, status, health)
    """
    # Check for API Gateway event structure
    if "httpMethod" in event:
        path = event.get("path", "")
        method = event.get("httpMethod", "")

        if path == "/health" and method == "GET":
            return "health"
        elif path == "/evaluation" and method == "POST":
            return "evaluation"
        elif path.startswith("/batch") and method == "POST":
            return "batch"
        elif path.startswith("/batch/") and method == "GET":
            return "status"
        elif path.startswith("/status/") and method == "GET":
            return "status"

    # Check for direct Lambda invocation
    if "operation" in event:
        return event["operation"]

    # Check for batch processing event
    if "batch_id" in event and "documents" in event:
        return "batch"

    # Check for evaluation request
    if "document_content" in event:
        return "evaluation"

    # Default to health check
    return "health"


def _handle_evaluation_request(
    event: Dict[str, Any], context: Any, correlation_id: str
) -> Dict[str, Any]:
    """
    Handle single document evaluation request.

    Args:
        event: Lambda event object
        context: Lambda context object
        correlation_id: Request correlation ID

    Returns:
        Dict containing evaluation response
    """
    try:
        # Parse request from event
        if "body" in event:
            # API Gateway event
            request_data = (
                json.loads(event["body"])
                if isinstance(event["body"], str)
                else event["body"]
            )
        else:
            # Direct Lambda invocation
            request_data = event

        # Validate request
        try:
            request = LambdaEvaluationRequest(**request_data)
        except Exception as e:
            return _create_error_response(
                "VALIDATION_ERROR", f"Invalid request format: {str(e)}", correlation_id
            )

        # Initialize evaluation service
        evaluation_service = EvaluationService(settings)

        # Perform evaluation
        start_time = time.time()
        result = evaluation_service.evaluate_document(
            document_content=request.document_content,
            context=request.context or "",
            output_text=request.output_text or "",
            provider=request.provider,
            options=request.options,
            correlation_id=correlation_id,
        )
        processing_time = time.time() - start_time

        # Create response
        response = LambdaEvaluationResponse(
            api_version="1.0.0",
            request_id=correlation_id,
            status="COMPLETED",
            result={
                "evaluation_id": result.evaluation_timestamp,
                "document_summary": (
                    result.input_text[:100] + "..."
                    if len(result.input_text) > 100
                    else result.input_text
                ),
                "criteria_results": [
                    {
                        "criterion_id": criterion.criterion_id,
                        "criterion_name": criterion.criterion_name,
                        "score": criterion.score,
                        "reasoning": criterion.score_reasoning,
                        "steps": [
                            {
                                "step_number": i + 1,
                                "step_name": step.step,
                                "result": step.result,
                                "reasoning": step.reasoning,
                            }
                            for i, step in enumerate(criterion.steps)
                        ],
                    }
                    for criterion in result.criterion_evaluations
                ],
                "overall_score": _calculate_overall_score(result.criterion_evaluations),
                "recommendation": _determine_recommendation(
                    result.criterion_evaluations
                ),
                "processing_time": processing_time,
            },
            metadata=EvaluationMetadata(
                api_version="1.0.0",
                processing_time=processing_time,
                correlation_id=correlation_id,
                timestamp=datetime.now().isoformat(),
            ),
            correlation_id=correlation_id,
            timestamp=datetime.now().isoformat(),
        )

        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "X-Correlation-ID": correlation_id,
            },
            "body": json.dumps(response.model_dump(), default=str),
        }

    except Exception as e:
        logger.error(f"Evaluation request error: {str(e)}", exc_info=True)
        return _create_error_response(
            "EVALUATION_ERROR", f"Evaluation failed: {str(e)}", correlation_id
        )


def _handle_batch_request(
    event: Dict[str, Any], context: Any, correlation_id: str
) -> Dict[str, Any]:
    """
    Handle batch evaluation request.

    Args:
        event: Lambda event object
        context: Lambda context object
        correlation_id: Request correlation ID

    Returns:
        Dict containing batch response
    """
    try:
        # Parse request from event
        if "body" in event:
            request_data = (
                json.loads(event["body"])
                if isinstance(event["body"], str)
                else event["body"]
            )
        else:
            request_data = event

        # Validate request
        try:
            request = LambdaBatchRequest(**request_data)
        except Exception as e:
            return _create_error_response(
                "VALIDATION_ERROR",
                f"Invalid batch request format: {str(e)}",
                correlation_id,
            )

        # Initialize batch service
        batch_service = BatchService(settings)

        # Create batch
        batch_id = batch_service.create_batch(
            documents=request.documents,
            batch_options=request.batch_options,
            correlation_id=correlation_id,
        )

        # Start batch processing (asynchronous)
        batch_service.start_batch(batch_id)

        # Create response
        response = LambdaBatchResponse(
            batch_id=batch_id,
            status="PENDING",
            total_documents=len(request.documents),
            processed_documents=0,
            failed_documents=0,
            estimated_completion=datetime.now().isoformat(),
        )

        return {
            "statusCode": 202,
            "headers": {
                "Content-Type": "application/json",
                "X-Correlation-ID": correlation_id,
            },
            "body": json.dumps(response.model_dump(), default=str),
        }

    except Exception as e:
        logger.error(f"Batch request error: {str(e)}", exc_info=True)
        return _create_error_response(
            "BATCH_ERROR", f"Batch processing failed: {str(e)}", correlation_id
        )


def _handle_status_request(
    event: Dict[str, Any], context: Any, correlation_id: str
) -> Dict[str, Any]:
    """
    Handle status check request.

    Args:
        event: Lambda event object
        context: Lambda context object
        correlation_id: Request correlation ID

    Returns:
        Dict containing status response
    """
    try:
        # Extract batch_id or request_id from path
        batch_id = None
        if "pathParameters" in event and event["pathParameters"]:
            batch_id = event["pathParameters"].get("batch_id")
        elif "batch_id" in event:
            batch_id = event["batch_id"]

        if not batch_id:
            return _create_error_response(
                "VALIDATION_ERROR", "Missing batch_id or request_id", correlation_id
            )

        # Initialize status service
        status_service = StatusService(settings)

        # Get status
        status = status_service.get_status(batch_id)

        if not status:
            return _create_error_response(
                "NOT_FOUND", f"Batch or request {batch_id} not found", correlation_id
            )

        # Create response
        response = LambdaStatusResponse(
            request_id=batch_id,
            status=status.get("status", "UNKNOWN"),
            progress=status.get("progress", 0.0),
            message=status.get("message", ""),
            last_updated=status.get("last_updated", datetime.now().isoformat()),
            estimated_completion=status.get("estimated_completion"),
            errors=status.get("errors", []),
        )

        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "X-Correlation-ID": correlation_id,
            },
            "body": json.dumps(response.model_dump(), default=str),
        }

    except Exception as e:
        logger.error(f"Status request error: {str(e)}", exc_info=True)
        return _create_error_response(
            "STATUS_ERROR", f"Status check failed: {str(e)}", correlation_id
        )


def _handle_health_check(
    event: Dict[str, Any], context: Any, correlation_id: str
) -> Dict[str, Any]:
    """
    Handle health check request.

    Args:
        event: Lambda event object
        context: Lambda context object
        correlation_id: Request correlation ID

    Returns:
        Dict containing health response
    """
    try:
        # Check Lambda runtime health
        health_status = "healthy"
        uptime = 0.0

        if context:
            # Calculate uptime (approximate) - simplified for Lambda
            try:
                # For Lambda, we'll use a simple uptime calculation
                uptime = 0.0  # Lambda functions are stateless
            except Exception:
                uptime = 0.0

        response = {
            "status": health_status,
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0",
            "uptime": uptime,
            "correlation_id": correlation_id,
        }

        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "X-Correlation-ID": correlation_id,
            },
            "body": json.dumps(response, default=str),
        }

    except Exception as e:
        logger.error(f"Health check error: {str(e)}", exc_info=True)
        return _create_error_response(
            "HEALTH_ERROR", f"Health check failed: {str(e)}", correlation_id
        )


def _calculate_overall_score(criterion_evaluations: List[Any]) -> float:
    """
    Calculate overall score from criterion evaluations.

    Args:
        criterion_evaluations: List of criterion evaluation results

    Returns:
        float: Overall score (0.0-1.0)
    """
    if not criterion_evaluations:
        return 0.5

    # Calculate average score and normalize to 0.0-1.0
    total_score = sum(criterion.score for criterion in criterion_evaluations)
    avg_score = total_score / len(criterion_evaluations)

    # Normalize from 1-5 scale to 0.0-1.0 scale
    return (avg_score - 1.0) / 4.0


def _determine_recommendation(criterion_evaluations: List[Any]) -> str:
    """
    Determine disclosure recommendation from criterion evaluations.

    Args:
        criterion_evaluations: List of criterion evaluation results

    Returns:
        str: Recommendation (DISCLOSE, LIKELY_DISCLOSE, etc.)
    """
    if not criterion_evaluations:
        return "UNCLEAR"

    # Calculate average score
    avg_score = sum(criterion.score for criterion in criterion_evaluations) / len(
        criterion_evaluations
    )

    if avg_score <= 1.5:
        return "NON_DISCLOSE"
    elif avg_score <= 2.5:
        return "LIKELY_NON_DISCLOSE"
    elif avg_score <= 3.5:
        return "UNCLEAR"
    elif avg_score <= 4.5:
        return "LIKELY_DISCLOSE"
    else:
        return "DISCLOSE"


def _create_error_response(
    error_code: str, error_message: str, correlation_id: str
) -> Dict[str, Any]:
    """
    Create standardized error response.

    Args:
        error_code: Error code
        error_message: Error message
        correlation_id: Request correlation ID

    Returns:
        Dict containing error response
    """
    error_response = LambdaErrorResponse(
        error_code=error_code,
        error_message=error_message,
        error_type="PROCESSING_ERROR",
        correlation_id=correlation_id,
        timestamp=datetime.now().isoformat(),
        resolution_guidance="Please check the request format and try again",
    )

    return {
        "statusCode": 400 if error_code == "VALIDATION_ERROR" else 500,
        "headers": {
            "Content-Type": "application/json",
            "X-Correlation-ID": correlation_id,
        },
        "body": json.dumps(error_response.model_dump(), default=str),
    }


# For direct Lambda invocation testing
if __name__ == "__main__":
    # Test event for direct invocation
    test_event = {
        "operation": "evaluation",
        "document_content": "This document contains personal information about John Doe.",
        "context": "Test evaluation request",
    }

    class MockContext:
        aws_request_id = "test-request-123"

    result = lambda_handler(test_event, MockContext())
    print(json.dumps(result, indent=2, default=str))
