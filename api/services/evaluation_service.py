"""
FastAPI evaluation service.

This service handles single document evaluation requests for the FastAPI application.
It orchestrates the evaluation process by interacting with the DisclosureEvaluator.
"""

import logging
import time
from typing import Optional, Dict, Any
from datetime import datetime

from api.config.settings import FastAPISettings
from api.models.requests import EvaluationRequest
from api.models.responses import (
    EvaluationResponse,
    EvaluationMetadata,
    CriterionResult,
    EvaluationStep,
)

# Import the existing evaluator modules
from evaluator import DisclosureEvaluator


class FastAPIEvaluationService:
    """
    Service for handling evaluation requests in FastAPI.

    This service provides a clean interface between the FastAPI endpoints
    and the existing evaluation logic.
    """

    def __init__(self, settings: FastAPISettings):
        """
        Initialize the FastAPI evaluation service.

        Args:
            settings: FastAPI configuration settings
        """
        self.settings = settings
        self.logger = logging.getLogger(f"{__name__}.FastAPIEvaluationService")

    def evaluate_document(
        self, request: EvaluationRequest, correlation_id: str
    ) -> EvaluationResponse:
        """
        Evaluate a single document for disclosure compliance.

        Args:
            request: Evaluation request data
            correlation_id: Request correlation ID

        Returns:
            EvaluationResponse: Evaluation result

        Raises:
            ValueError: If evaluation fails
        """
        try:
            start_time = time.time()
            self.logger.info(
                f"Starting document evaluation with correlation ID: {correlation_id}"
            )

            # Initialize evaluator with provider
            evaluator = DisclosureEvaluator(provider=request.provider)

            # Perform evaluation
            result = evaluator.evaluate_disclosure(
                input_text=request.document_content,
                context=request.context or "",
                output_text=request.output_text or "",
            )

            # Calculate processing time
            processing_time = time.time() - start_time

            # Convert result to FastAPI response format
            response = self._convert_to_fastapi_response(
                result, processing_time, correlation_id
            )

            self.logger.info(
                f"Document evaluation completed in {processing_time:.2f} seconds"
            )

            return response

        except Exception as e:
            self.logger.error(
                f"Error during document evaluation: {str(e)}", exc_info=True
            )
            raise ValueError(f"Evaluation failed: {str(e)}")

    def _convert_to_fastapi_response(
        self, result: Any, processing_time: float, correlation_id: str
    ) -> EvaluationResponse:
        """
        Convert DisclosureEvaluationResult to FastAPI response format.

        Args:
            result: DisclosureEvaluationResult from evaluator
            processing_time: Processing time in seconds
            correlation_id: Request correlation ID

        Returns:
            EvaluationResponse: FastAPI response format
        """
        # Convert criteria results
        criteria_results = []
        for criterion in result.criterion_evaluations:
            steps = []
            for i, step in enumerate(criterion.steps):
                steps.append(
                    EvaluationStep(
                        step_number=i + 1,
                        step_name=step.step,
                        result=step.result,
                        reasoning=step.reasoning,
                    )
                )

            criteria_results.append(
                CriterionResult(
                    criterion_id=criterion.criterion_id,
                    criterion_name=criterion.criterion_name,
                    score=criterion.score,
                    reasoning=criterion.score_reasoning,
                    steps=steps,
                )
            )

        # Calculate overall score and recommendation
        overall_score = self._calculate_overall_score(result.criterion_evaluations)
        recommendation = self._determine_recommendation(result.criterion_evaluations)

        # Create metadata
        metadata = EvaluationMetadata(
            api_version=self.settings.api_version,
            processing_time=processing_time,
            correlation_id=correlation_id,
            timestamp=datetime.now().isoformat(),
        )

        # Create response
        return EvaluationResponse(
            evaluation_id=result.evaluation_timestamp,
            document_summary=(
                result.input_text[:100] + "..."
                if len(result.input_text) > 100
                else result.input_text
            ),
            criteria_results=criteria_results,
            overall_score=overall_score,
            recommendation=recommendation,
            processing_time=processing_time,
            metadata=metadata,
        )

    def _calculate_overall_score(self, criteria_evaluations: list) -> float:
        """Calculate overall score from criteria evaluations."""
        if not criteria_evaluations:
            return 0.0

        scores = [criterion.score for criterion in criteria_evaluations]
        return sum(scores) / len(scores) / 5.0  # Normalize to 0-1 scale

    def _determine_recommendation(self, criteria_evaluations: list) -> str:
        """Determine recommendation based on criteria evaluations."""
        if not criteria_evaluations:
            return "UNCLEAR"

        # Count scores by category
        high_scores = sum(1 for c in criteria_evaluations if c.score >= 4.0)
        medium_scores = sum(1 for c in criteria_evaluations if 2.0 <= c.score < 4.0)
        low_scores = sum(1 for c in criteria_evaluations if c.score < 2.0)

        total = len(criteria_evaluations)

        if high_scores / total >= 0.6:
            return "NON_DISCLOSE"
        elif low_scores / total >= 0.6:
            return "DISCLOSE"
        elif medium_scores / total >= 0.5:
            return "LIKELY_DISCLOSE"
        else:
            return "UNCLEAR"
