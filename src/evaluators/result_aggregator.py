"""
Aggregates individual criterion evaluations into overall result.
"""

import logging
from typing import List, Dict, Any
from ..models.evaluation import CriterionEvaluation
from ..llm import LLMProvider


class ResultAggregator:
    """Aggregates individual criterion evaluations into overall result"""

    def __init__(self, correlation_id: str, llm_provider: LLMProvider):
        self.correlation_id = correlation_id
        self.llm_provider = llm_provider
        self.logger = logging.getLogger(f"ResultAggregator.{correlation_id}")

    def aggregate_results(
        self,
        criterion_evaluations: List[CriterionEvaluation],
        criteria_config: Dict[str, Any],
    ) -> None:
        """
        Process evaluation results (no longer needed as user makes final judgment).

        Args:
            criterion_evaluations: List of individual criterion evaluations
            criteria_config: Criteria configuration for weights
        """
        try:
            self.logger.info("Starting result aggregation")

            self.logger.info(
                "Result aggregation completed: User will make final judgment"
            )

        except Exception as e:
            self.logger.error(f"Result aggregation failed: {str(e)}")
            raise ValueError(f"Result aggregation failed: {str(e)}")
