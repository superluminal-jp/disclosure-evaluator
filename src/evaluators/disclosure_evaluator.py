"""
Main evaluator orchestrating step-by-step evaluation with comprehensive error handling.
"""

import os
import logging
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any

from ..models.evaluation import (
    EvaluationStep,
    CriterionEvaluation,
    DisclosureEvaluationResult,
)
from ..llm import LLMProvider, create_llm_provider
from ..utils import load_criteria
from .step_evaluator import StepEvaluator
from .criterion_evaluator import CriterionEvaluator
from .result_aggregator import ResultAggregator


class DisclosureEvaluator:
    """Main evaluator orchestrating step-by-step evaluation with comprehensive error handling"""

    def __init__(self, api_key: str = None, provider: str = None, config_manager=None):
        """Initialize evaluator with LLM provider and correlation ID"""
        # Get provider configuration
        if provider is None:
            provider = config_manager.get_current_provider()

        provider_config = config_manager.get_provider_config(provider)

        # Set credentials from environment variables
        if provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable is required")
            provider_config["api_key"] = api_key
        elif provider == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY environment variable is required")
            provider_config["api_key"] = api_key
        elif provider in ["bedrock", "bedrock_nova"]:
            # AWS credentials are handled by boto3 automatically from environment
            # or AWS credentials file
            pass

        # Create LLM provider
        self.llm_provider = create_llm_provider(provider, config_manager)
        self.correlation_id = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.logger = logging.getLogger(f"DisclosureEvaluator.{self.correlation_id}")

        # Initialize evaluators
        self.criterion_evaluator = CriterionEvaluator(
            self.llm_provider, self.correlation_id, config_manager
        )
        self.result_aggregator = ResultAggregator(
            self.correlation_id, self.llm_provider
        )

        # Thread-safe lock for logging
        self._log_lock = threading.Lock()

    def _evaluate_criterion_parallel(
        self, criterion: Dict[str, Any], eval_context: Dict[str, Any]
    ) -> CriterionEvaluation:
        """
        Evaluate a single criterion in parallel execution.

        Args:
            criterion: Criterion configuration
            eval_context: Evaluation context

        Returns:
            CriterionEvaluation: Evaluation result
        """
        try:
            with self._log_lock:
                self.logger.info(
                    f"Starting parallel evaluation of criterion: {criterion['name']}"
                )

            # Create a new criterion evaluator for this thread
            thread_correlation_id = f"{self.correlation_id}_{criterion['id']}"
            thread_evaluator = CriterionEvaluator(
                self.llm_provider,
                thread_correlation_id,
                self.criterion_evaluator.config_manager,
            )

            criterion_eval = thread_evaluator.evaluate_criterion(
                criterion, eval_context
            )

            with self._log_lock:
                self.logger.info(
                    f"Criterion {criterion['name']} completed successfully in parallel"
                )

            return criterion_eval

        except Exception as e:
            with self._log_lock:
                self.logger.error(
                    f"Criterion {criterion['name']} parallel evaluation failed: {str(e)}"
                )

            # Create fallback evaluation
            return CriterionEvaluation(
                criterion_id=criterion["id"],
                criterion_name=criterion["name"],
                article=criterion["article"],
                steps=[
                    EvaluationStep(
                        step="エラー",
                        result="NO",
                        reasoning=f"並列評価エラー: {str(e)}",
                    )
                ],
                score=3,
                score_reasoning=f"並列評価エラーのため中間スコアを設定: {str(e)}",
            )

    def _evaluate_criteria_parallel(
        self, criteria: List[Dict[str, Any]], eval_context: Dict[str, Any]
    ) -> List[CriterionEvaluation]:
        """
        Evaluate multiple criteria in parallel using ThreadPoolExecutor.

        Args:
            criteria: List of criterion configurations
            eval_context: Evaluation context

        Returns:
            List[CriterionEvaluation]: List of evaluation results
        """
        self.logger.info(f"Starting parallel evaluation of {len(criteria)} criteria")

        criterion_evaluations = []

        # Use ThreadPoolExecutor for parallel execution
        eval_config = self.criterion_evaluator.config_manager.get_evaluation_config()
        parallel_config = eval_config.get("parallel", {})

        if not parallel_config.get("enabled", True):
            # Fallback to sequential evaluation
            return self._evaluate_criteria_sequential(criteria, eval_context)

        max_workers = min(parallel_config.get("max_workers", 3), len(criteria))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all evaluation tasks with explicit mapping
            future_to_criterion = {}
            for criterion in criteria:
                future = executor.submit(
                    self._evaluate_criterion_parallel, criterion, eval_context
                )
                future_to_criterion[future] = criterion

            # Collect results as they complete
            for future in as_completed(future_to_criterion):
                criterion = future_to_criterion[future]
                try:
                    result = future.result()
                    # Verify the result matches the expected criterion
                    if result.criterion_id != criterion["id"]:
                        self.logger.warning(
                            f"Criterion ID mismatch: expected {criterion['id']}, got {result.criterion_id}"
                        )
                    criterion_evaluations.append(result)
                    self.logger.info(
                        f"Parallel evaluation completed for: {criterion['name']} (ID: {criterion['id']})"
                    )
                except Exception as e:
                    self.logger.error(
                        f"Parallel evaluation failed for {criterion['name']} (ID: {criterion['id']}): {str(e)}"
                    )
                    # Add fallback evaluation with correct criterion ID
                    fallback_eval = CriterionEvaluation(
                        criterion_id=criterion["id"],
                        criterion_name=criterion["name"],
                        article=criterion["article"],
                        steps=[
                            EvaluationStep(
                                step="エラー",
                                result="NO",
                                reasoning=f"並列実行エラー: {str(e)}",
                            )
                        ],
                        score=3,
                        score_reasoning=f"並列実行エラーのため中間スコアを設定: {str(e)}",
                    )
                    criterion_evaluations.append(fallback_eval)

        # Sort results by original criteria order
        criterion_evaluations.sort(
            key=lambda x: next(
                (i for i, c in enumerate(criteria) if c["id"] == x.criterion_id), 0
            )
        )

        # Verify all criteria were evaluated
        evaluated_ids = {c.criterion_id for c in criterion_evaluations}
        expected_ids = {c["id"] for c in criteria}

        if evaluated_ids != expected_ids:
            missing_ids = expected_ids - evaluated_ids
            extra_ids = evaluated_ids - expected_ids
            self.logger.error(
                f"Evaluation mismatch: missing {missing_ids}, extra {extra_ids}"
            )
        else:
            self.logger.info("All criteria successfully evaluated and mapped")

        self.logger.info(
            f"Parallel evaluation completed: {len(criterion_evaluations)} criteria evaluated"
        )
        return criterion_evaluations

    def evaluate_disclosure(
        self, input_text: str, context: str = "", output_text: str = ""
    ) -> DisclosureEvaluationResult:
        """
        Evaluate disclosure using step-by-step approach with comprehensive error handling.

        Args:
            input_text: Information to be evaluated
            context: Additional context
            output_text: Output text if provided

        Returns:
            DisclosureEvaluationResult: Complete evaluation result

        Raises:
            ValueError: If evaluation fails
            Exception: For unexpected errors
        """
        try:
            self.logger.info(
                f"Starting disclosure evaluation with correlation ID: {self.correlation_id}"
            )

            # Load criteria configuration
            criteria = load_criteria()

            # Prepare evaluation context
            eval_context = {
                "input_text": input_text,
                "context": context,
                "output_text": output_text,
            }

            # Evaluate criteria in parallel
            criterion_evaluations = self._evaluate_criteria_parallel(
                criteria["criteria"], eval_context
            )

            # Process evaluation results
            self.result_aggregator.aggregate_results(criterion_evaluations, criteria)

            # Create final result
            result = DisclosureEvaluationResult(
                input_text=input_text,
                context=context,
                output_text=output_text,
                criterion_evaluations=criterion_evaluations,
                evaluation_timestamp=datetime.now().isoformat(),
            )

            self.logger.info(
                f"Disclosure evaluation completed successfully: {len(criterion_evaluations)} criteria evaluated"
            )

            # Save evaluation result to JSON file
            self._save_evaluation_result(result)

            # Save evaluation result to CSV file
            self._save_evaluation_result_csv(result)

            return result

        except Exception as e:
            self.logger.error(f"Disclosure evaluation failed: {str(e)}")
            raise ValueError(f"Disclosure evaluation failed: {str(e)}")

    def _save_evaluation_result(self, result: DisclosureEvaluationResult) -> str:
        """
        Save evaluation result to timestamped JSON file.

        Args:
            result: The evaluation result to save

        Returns:
            str: Path to the saved file
        """
        try:
            # Get output configuration
            output_config = self.criterion_evaluator.config_manager.get_output_config()
            timestamp_format = output_config.get("timestamp_format", "%Y%m%d_%H%M%S")
            encoding = output_config.get("encoding", "utf-8")

            # Generate timestamped output filename
            timestamp = datetime.now().strftime(timestamp_format)
            outputs_dir = self.criterion_evaluator.config_manager.get(
                "logging.directories.outputs", "outputs"
            )
            output_filename = f"{outputs_dir}/evaluation_result_{timestamp}.json"

            # Convert result to JSON with proper formatting
            result_json = result.model_dump_json(indent=2)

            # Save to file
            with open(output_filename, "w", encoding=encoding) as f:
                f.write(result_json)

            self.logger.info(f"Evaluation result saved to: {output_filename}")
            return output_filename

        except Exception as e:
            self.logger.error(f"Failed to save evaluation result: {str(e)}")
            # Don't raise exception as this is not critical for evaluation
            return ""

    def _save_evaluation_result_csv(self, result: DisclosureEvaluationResult) -> str:
        """
        Save evaluation result to CSV format.

        Args:
            result: DisclosureEvaluationResult to save

        Returns:
            str: Path to saved CSV file, empty string if failed
        """
        try:
            import csv

            output_config = self.criterion_evaluator.config_manager.get_output_config()
            csv_config = output_config.get("csv", {})

            # Check if CSV output is enabled
            if not csv_config.get("enabled", True):
                self.logger.info("CSV output is disabled in configuration")
                return ""

            timestamp_format = output_config.get("timestamp_format", "%Y%m%d_%H%M%S")
            encoding = output_config.get("encoding", "utf-8")
            delimiter = csv_config.get("delimiter", ",")
            quote_char = csv_config.get("quote_char", '"')
            include_headers = csv_config.get("include_headers", True)

            timestamp = datetime.now().strftime(timestamp_format)
            outputs_dir = self.criterion_evaluator.config_manager.get(
                "logging.directories.outputs", "outputs"
            )
            csv_filename = f"{outputs_dir}/evaluation_result_{timestamp}.csv"

            with open(csv_filename, "w", newline="", encoding=encoding) as csvfile:
                writer = csv.writer(csvfile, delimiter=delimiter, quotechar=quote_char)

                # Write header if enabled
                if include_headers:
                    writer.writerow(
                        [
                            "Criterion ID",
                            "Criterion Name",
                            "Article",
                            "Score",
                            "Score Reasoning",
                            "Step",
                            "Step Result",
                            "Step Reasoning",
                        ]
                    )

                # Write data rows
                for criterion in result.criterion_evaluations:
                    for step in criterion.steps:
                        writer.writerow(
                            [
                                criterion.criterion_id,
                                criterion.criterion_name,
                                criterion.article,
                                criterion.score,
                                criterion.score_reasoning,
                                step.step,
                                step.result,
                                step.reasoning,
                            ]
                        )

            self.logger.info(f"Evaluation result saved to CSV: {csv_filename}")
            return csv_filename

        except Exception as e:
            self.logger.error(f"Failed to save evaluation result to CSV: {str(e)}")
            return ""


def evaluate_disclosure(
    input_text,
    context="",
    output_text="",
    api_key=None,
    provider=None,
    config_manager=None,
):
    """
    Legacy function for backward compatibility.
    Creates a new DisclosureEvaluator instance and evaluates disclosure.
    """
    evaluator = DisclosureEvaluator(api_key, provider, config_manager)
    return evaluator.evaluate_disclosure(input_text, context, output_text)
