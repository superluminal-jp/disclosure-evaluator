"""
FastAPI batch service.

This service handles batch evaluation requests for the FastAPI application.
It manages the lifecycle of batch evaluations, including submission, status tracking, and result retrieval.
"""

import logging
import time
from typing import List, Optional, Dict, Any
from datetime import datetime

from api.config.settings import FastAPISettings
from api.models.requests import BatchRequest, BatchDocument
from api.models.responses import (
    BatchResponse,
    BatchStatusResponse,
    BatchResultsResponse,
)
from api.services.evaluation_service import FastAPIEvaluationService

# Import the existing batch evaluator modules
from evaluator import BatchEvaluator, BatchConfiguration, DocumentInput


class FastAPIBatchService:
    """
    Service for handling batch evaluation requests in FastAPI.

    This service manages batch processing lifecycle and integrates
    with the existing batch evaluation system.
    """

    def __init__(self, settings: FastAPISettings):
        """
        Initialize the FastAPI batch service.

        Args:
            settings: FastAPI configuration settings
        """
        self.settings = settings
        self.logger = logging.getLogger(f"{__name__}.FastAPIBatchService")

        # Initialize batch evaluator with FastAPI-optimized configuration
        self.batch_config = BatchConfiguration(
            max_concurrent_workers=settings.max_concurrent_workers,
            timeout_seconds=settings.default_timeout,
            max_retry_attempts=3,
            file_size_limit=settings.max_document_size,
            memory_limit_mb=1024,
            api_rate_limit_delay=1.0,
            retry_delay_seconds=5.0,
            exponential_backoff=True,
        )

        self.batch_evaluator = BatchEvaluator(config=self.batch_config)
        self.evaluation_service = FastAPIEvaluationService(settings)

    def create_batch(self, request: BatchRequest, correlation_id: str) -> BatchResponse:
        """
        Create a new batch for evaluation.

        Args:
            request: Batch request data
            correlation_id: Request correlation ID

        Returns:
            BatchResponse: Batch creation response

        Raises:
            ValueError: If batch creation fails
        """
        try:
            self.logger.info(f"Creating batch with {len(request.documents)} documents")

            # Convert documents to DocumentInput objects
            doc_inputs = []
            for doc in request.documents:
                doc_input = DocumentInput(
                    file_path="",  # No file path for API requests
                    file_name=f"{doc.document_id}.txt",
                    context=doc.context or "",
                    output_text="",
                )
                doc_inputs.append(doc_input)

            # Create batch
            batch_id = self.batch_evaluator.create_batch(doc_inputs)

            self.logger.info(f"Created batch: {batch_id}")

            return BatchResponse(
                batch_id=batch_id,
                status="PENDING",
                total_documents=len(request.documents),
                processed_documents=0,
                failed_documents=0,
                estimated_completion=None,
                results=None,
            )

        except Exception as e:
            self.logger.error(f"Failed to create batch: {str(e)}", exc_info=True)
            raise ValueError(f"Failed to create batch: {str(e)}")

    def get_batch_status(
        self, batch_id: str, correlation_id: str
    ) -> BatchStatusResponse:
        """
        Get the status of a batch evaluation.

        Args:
            batch_id: Batch identifier
            correlation_id: Request correlation ID

        Returns:
            BatchStatusResponse: Batch status information

        Raises:
            ValueError: If batch not found
        """
        try:
            self.logger.info(f"Getting status for batch: {batch_id}")

            # Get batch status from evaluator
            batch_status = self.batch_evaluator.get_batch_status(batch_id)

            if not batch_status:
                raise ValueError(f"Batch {batch_id} not found")

            # Calculate progress percentage
            total = batch_status.get("total_documents", 0)
            processed = batch_status.get("processed_documents", 0)
            progress = (processed / total * 100) if total > 0 else 0.0

            return BatchStatusResponse(
                batch_id=batch_id,
                status=batch_status.get("status", "UNKNOWN"),
                total_documents=total,
                processed_documents=processed,
                failed_documents=batch_status.get("failed_documents", 0),
                progress=progress,
                estimated_completion=batch_status.get("estimated_completion"),
                last_updated=datetime.now().isoformat(),
            )

        except Exception as e:
            self.logger.error(f"Failed to get batch status: {str(e)}", exc_info=True)
            raise ValueError(f"Failed to get batch status: {str(e)}")

    def get_batch_results(
        self, batch_id: str, correlation_id: str
    ) -> BatchResultsResponse:
        """
        Get the results of a completed batch evaluation.

        Args:
            batch_id: Batch identifier
            correlation_id: Request correlation ID

        Returns:
            BatchResultsResponse: Batch results

        Raises:
            ValueError: If batch not found or not completed
        """
        try:
            self.logger.info(f"Getting results for batch: {batch_id}")

            # Get batch results from evaluator
            batch_results = self.batch_evaluator.get_batch_results(batch_id)

            if not batch_results:
                raise ValueError(f"Batch {batch_id} not found or not completed")

            # Convert results to FastAPI format
            results = []
            for result in batch_results.get("results", []):
                # Convert each result to EvaluationResponse format
                # This would need to be implemented based on the actual result structure
                pass  # TODO: Implement result conversion

            # Create summary
            summary = {
                "total_documents": batch_results.get("total_documents", 0),
                "successful_documents": batch_results.get("successful_documents", 0),
                "failed_documents": batch_results.get("failed_documents", 0),
                "total_processing_time": batch_results.get(
                    "total_processing_time", 0.0
                ),
            }

            return BatchResultsResponse(
                batch_id=batch_id,
                status=batch_results.get("status", "COMPLETED"),
                results=results,
                summary=summary,
            )

        except Exception as e:
            self.logger.error(f"Failed to get batch results: {str(e)}", exc_info=True)
            raise ValueError(f"Failed to get batch results: {str(e)}")
