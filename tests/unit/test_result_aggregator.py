"""
Unit tests for ResultAggregator class.
Tests lines 1281-1326: ResultAggregator class
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from main import (
    ResultAggregator,
    CriterionEvaluation,
    EvaluationStep,
    DisclosureEvaluationResult,
)


class TestResultAggregator:
    """Test ResultAggregator class."""

    def test_result_aggregator_init(self, mock_llm_response):
        """Test ResultAggregator initialization."""
        correlation_id = "test_correlation_001"
        mock_llm_provider = Mock()

        with patch("main.logging.getLogger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            aggregator = ResultAggregator(correlation_id, mock_llm_provider)

            assert aggregator.correlation_id == correlation_id
            assert aggregator.llm_provider == mock_llm_provider
            assert aggregator.logger == mock_logger
            mock_get_logger.assert_called_once_with(
                f"ResultAggregator.{correlation_id}"
            )

    def test_aggregate_results_success(self, mock_llm_response):
        """Test successful result aggregation."""
        correlation_id = "test_correlation_001"
        mock_llm_provider = Mock()
        mock_llm_provider.generate_response.return_value = "総合判断: 開示が適切"

        # Create sample criterion evaluations
        criterion_evaluations = [
            CriterionEvaluation(
                criterion_id="article_5_1",
                criterion_name="個人情報保護",
                article="第5条第1号",
                steps=[
                    EvaluationStep(step="ステップ1", result="YES", reasoning="理由1"),
                    EvaluationStep(step="ステップ2", result="NO", reasoning="理由2"),
                ],
                score=4,
                score_reasoning="スコア理由1",
            ),
            CriterionEvaluation(
                criterion_id="article_5_2",
                criterion_name="法人等情報保護",
                article="第5条第2号",
                steps=[
                    EvaluationStep(step="ステップ1", result="NO", reasoning="理由1"),
                    EvaluationStep(step="ステップ2", result="NO", reasoning="理由2"),
                ],
                score=5,
                score_reasoning="スコア理由2",
            ),
        ]

        context = {
            "input_text": "テスト入力",
            "context": "テストコンテキスト",
            "output_text": "テスト出力",
        }

        with patch("main.logging.getLogger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            with patch("main.config_manager") as mock_config:
                mock_config.get_prompt.return_value = "システムプロンプト"

                aggregator = ResultAggregator(correlation_id, mock_llm_provider)
                result = aggregator.aggregate_results(criterion_evaluations, context)

                assert isinstance(result, DisclosureEvaluationResult)
                assert result.input_text == context["input_text"]
                assert result.context == context["context"]
                assert result.output_text == context["output_text"]
                assert len(result.criterion_evaluations) == 2
                assert result.criterion_evaluations[0].criterion_id == "article_5_1"
                assert result.criterion_evaluations[1].criterion_id == "article_5_2"
                assert result.evaluation_timestamp is not None

                # Verify LLM provider was called for overall evaluation
                mock_llm_provider.generate_response.assert_called_once()
                call_args = mock_llm_provider.generate_response.call_args[0][0]
                assert call_args[0]["role"] == "system"
                assert call_args[1]["role"] == "user"
                assert "総合評価" in call_args[1]["content"]

                # Verify logging
                mock_logger.info.assert_called()
                mock_logger.info.assert_any_call("Starting result aggregation")
                mock_logger.info.assert_any_call(
                    "Result aggregation completed successfully"
                )

    def test_aggregate_results_exception(self, mock_llm_response):
        """Test result aggregation with exception."""
        correlation_id = "test_correlation_001"
        mock_llm_provider = Mock()
        mock_llm_provider.generate_response.side_effect = Exception("LLM API Error")

        criterion_evaluations = [
            CriterionEvaluation(
                criterion_id="article_5_1",
                criterion_name="個人情報保護",
                article="第5条第1号",
                steps=[
                    EvaluationStep(step="ステップ1", result="YES", reasoning="理由1")
                ],
                score=3,
                score_reasoning="スコア理由",
            )
        ]

        context = {"input_text": "テスト入力"}

        with patch("main.logging.getLogger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            with patch("main.config_manager") as mock_config:
                mock_config.get_prompt.return_value = "システムプロンプト"

                aggregator = ResultAggregator(correlation_id, mock_llm_provider)

                with pytest.raises(
                    Exception, match="Result aggregation failed: LLM API Error"
                ):
                    aggregator.aggregate_results(criterion_evaluations, context)

                # Verify error logging
                mock_logger.error.assert_called_with(
                    "Result aggregation failed: LLM API Error"
                )

    def test_aggregate_results_empty_criteria(self, mock_llm_response):
        """Test result aggregation with empty criteria list."""
        correlation_id = "test_correlation_001"
        mock_llm_provider = Mock()
        mock_llm_provider.generate_response.return_value = "総合判断: 開示が適切"

        criterion_evaluations = []
        context = {"input_text": "テスト入力"}

        with patch("main.logging.getLogger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            with patch("main.config_manager") as mock_config:
                mock_config.get_prompt.return_value = "システムプロンプト"

                aggregator = ResultAggregator(correlation_id, mock_llm_provider)
                result = aggregator.aggregate_results(criterion_evaluations, context)

                assert isinstance(result, DisclosureEvaluationResult)
                assert len(result.criterion_evaluations) == 0
                assert result.input_text == context["input_text"]

    def test_aggregate_results_single_criterion(self, mock_llm_response):
        """Test result aggregation with single criterion."""
        correlation_id = "test_correlation_001"
        mock_llm_provider = Mock()
        mock_llm_provider.generate_response.return_value = "総合判断: 不開示が適切"

        criterion_evaluations = [
            CriterionEvaluation(
                criterion_id="article_5_1",
                criterion_name="個人情報保護",
                article="第5条第1号",
                steps=[
                    EvaluationStep(step="ステップ1", result="YES", reasoning="理由1"),
                    EvaluationStep(step="ステップ2", result="YES", reasoning="理由2"),
                ],
                score=1,
                score_reasoning="スコア理由",
            )
        ]

        context = {"input_text": "テスト入力"}

        with patch("main.logging.getLogger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            with patch("main.config_manager") as mock_config:
                mock_config.get_prompt.return_value = "システムプロンプト"

                aggregator = ResultAggregator(correlation_id, mock_llm_provider)
                result = aggregator.aggregate_results(criterion_evaluations, context)

                assert isinstance(result, DisclosureEvaluationResult)
                assert len(result.criterion_evaluations) == 1
                assert result.criterion_evaluations[0].score == 1

    def test_aggregate_results_high_scores(self, mock_llm_response):
        """Test result aggregation with high scores (disclosure likely)."""
        correlation_id = "test_correlation_001"
        mock_llm_provider = Mock()
        mock_llm_provider.generate_response.return_value = "総合判断: 開示が適切"

        criterion_evaluations = [
            CriterionEvaluation(
                criterion_id="article_5_1",
                criterion_name="個人情報保護",
                article="第5条第1号",
                steps=[
                    EvaluationStep(step="ステップ1", result="NO", reasoning="理由1")
                ],
                score=5,
                score_reasoning="スコア理由1",
            ),
            CriterionEvaluation(
                criterion_id="article_5_2",
                criterion_name="法人等情報保護",
                article="第5条第2号",
                steps=[
                    EvaluationStep(step="ステップ1", result="NO", reasoning="理由1")
                ],
                score=5,
                score_reasoning="スコア理由2",
            ),
        ]

        context = {"input_text": "テスト入力"}

        with patch("main.logging.getLogger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            with patch("main.config_manager") as mock_config:
                mock_config.get_prompt.return_value = "システムプロンプト"

                aggregator = ResultAggregator(correlation_id, mock_llm_provider)
                result = aggregator.aggregate_results(criterion_evaluations, context)

                assert isinstance(result, DisclosureEvaluationResult)
                assert len(result.criterion_evaluations) == 2
                assert all(
                    criterion.score == 5 for criterion in result.criterion_evaluations
                )

    def test_aggregate_results_low_scores(self, mock_llm_response):
        """Test result aggregation with low scores (non-disclosure likely)."""
        correlation_id = "test_correlation_001"
        mock_llm_provider = Mock()
        mock_llm_provider.generate_response.return_value = "総合判断: 不開示が適切"

        criterion_evaluations = [
            CriterionEvaluation(
                criterion_id="article_5_1",
                criterion_name="個人情報保護",
                article="第5条第1号",
                steps=[
                    EvaluationStep(step="ステップ1", result="YES", reasoning="理由1"),
                    EvaluationStep(step="ステップ2", result="YES", reasoning="理由2"),
                ],
                score=1,
                score_reasoning="スコア理由1",
            ),
            CriterionEvaluation(
                criterion_id="article_5_2",
                criterion_name="法人等情報保護",
                article="第5条第2号",
                steps=[
                    EvaluationStep(step="ステップ1", result="YES", reasoning="理由1"),
                    EvaluationStep(step="ステップ2", result="YES", reasoning="理由2"),
                ],
                score=2,
                score_reasoning="スコア理由2",
            ),
        ]

        context = {"input_text": "テスト入力"}

        with patch("main.logging.getLogger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            with patch("main.config_manager") as mock_config:
                mock_config.get_prompt.return_value = "システムプロンプト"

                aggregator = ResultAggregator(correlation_id, mock_llm_provider)
                result = aggregator.aggregate_results(criterion_evaluations, context)

                assert isinstance(result, DisclosureEvaluationResult)
                assert len(result.criterion_evaluations) == 2
                assert all(
                    criterion.score <= 2 for criterion in result.criterion_evaluations
                )

    def test_aggregate_results_mixed_scores(self, mock_llm_response):
        """Test result aggregation with mixed scores."""
        correlation_id = "test_correlation_001"
        mock_llm_provider = Mock()
        mock_llm_provider.generate_response.return_value = "総合判断: 慎重な検討が必要"

        criterion_evaluations = [
            CriterionEvaluation(
                criterion_id="article_5_1",
                criterion_name="個人情報保護",
                article="第5条第1号",
                steps=[
                    EvaluationStep(step="ステップ1", result="YES", reasoning="理由1")
                ],
                score=2,
                score_reasoning="スコア理由1",
            ),
            CriterionEvaluation(
                criterion_id="article_5_2",
                criterion_name="法人等情報保護",
                article="第5条第2号",
                steps=[
                    EvaluationStep(step="ステップ1", result="NO", reasoning="理由1")
                ],
                score=4,
                score_reasoning="スコア理由2",
            ),
            CriterionEvaluation(
                criterion_id="article_5_3",
                criterion_name="国家安全保障",
                article="第5条第3号",
                steps=[
                    EvaluationStep(step="ステップ1", result="NO", reasoning="理由1")
                ],
                score=5,
                score_reasoning="スコア理由3",
            ),
        ]

        context = {"input_text": "テスト入力"}

        with patch("main.logging.getLogger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            with patch("main.config_manager") as mock_config:
                mock_config.get_prompt.return_value = "システムプロンプト"

                aggregator = ResultAggregator(correlation_id, mock_llm_provider)
                result = aggregator.aggregate_results(criterion_evaluations, context)

                assert isinstance(result, DisclosureEvaluationResult)
                assert len(result.criterion_evaluations) == 3
                scores = [criterion.score for criterion in result.criterion_evaluations]
                assert 2 in scores
                assert 4 in scores
                assert 5 in scores

    def test_aggregate_results_timestamp_format(self, mock_llm_response):
        """Test that evaluation timestamp is properly formatted."""
        correlation_id = "test_correlation_001"
        mock_llm_provider = Mock()
        mock_llm_provider.generate_response.return_value = "総合判断: 開示が適切"

        criterion_evaluations = [
            CriterionEvaluation(
                criterion_id="article_5_1",
                criterion_name="個人情報保護",
                article="第5条第1号",
                steps=[
                    EvaluationStep(step="ステップ1", result="NO", reasoning="理由1")
                ],
                score=5,
                score_reasoning="スコア理由",
            )
        ]

        context = {"input_text": "テスト入力"}

        with patch("main.logging.getLogger"):
            with patch("main.config_manager") as mock_config:
                mock_config.get_prompt.return_value = "システムプロンプト"

                with patch("main.datetime") as mock_datetime:
                    mock_datetime.now.return_value.isoformat.return_value = (
                        "2025-01-01T12:00:00"
                    )

                    aggregator = ResultAggregator(correlation_id, mock_llm_provider)
                    result = aggregator.aggregate_results(
                        criterion_evaluations, context
                    )

                    assert result.evaluation_timestamp == "2025-01-01T12:00:00"

    def test_aggregate_results_prompt_construction(self, mock_llm_response):
        """Test that the prompt is constructed correctly with criterion details."""
        correlation_id = "test_correlation_001"
        mock_llm_provider = Mock()
        mock_llm_provider.generate_response.return_value = "総合判断: 開示が適切"

        criterion_evaluations = [
            CriterionEvaluation(
                criterion_id="article_5_1",
                criterion_name="個人情報保護",
                article="第5条第1号",
                steps=[
                    EvaluationStep(step="ステップ1", result="YES", reasoning="理由1"),
                    EvaluationStep(step="ステップ2", result="NO", reasoning="理由2"),
                ],
                score=3,
                score_reasoning="スコア理由",
            )
        ]

        context = {
            "input_text": "テスト入力",
            "context": "テストコンテキスト",
            "output_text": "テスト出力",
        }

        with patch("main.logging.getLogger"):
            with patch("main.config_manager") as mock_config:
                mock_config.get_prompt.return_value = "システムプロンプト"

                aggregator = ResultAggregator(correlation_id, mock_llm_provider)
                aggregator.aggregate_results(criterion_evaluations, context)

                # Verify the prompt contains criterion details
                call_args = mock_llm_provider.generate_response.call_args[0][0]
                prompt_content = call_args[1]["content"]

                assert "個人情報保護" in prompt_content
                assert "第5条第1号" in prompt_content
                assert "スコア: 3" in prompt_content
                assert "ステップ1: YES" in prompt_content
                assert "ステップ2: NO" in prompt_content
                assert "テスト入力" in prompt_content
                assert "テストコンテキスト" in prompt_content
                assert "テスト出力" in prompt_content
