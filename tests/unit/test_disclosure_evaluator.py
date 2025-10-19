"""
Unit tests for DisclosureEvaluator main class.
Tests lines 1328-1695: DisclosureEvaluator including parallel and sequential evaluation
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.evaluators import DisclosureEvaluator, CriterionEvaluator, ResultAggregator
from src.models import CriterionEvaluation, EvaluationStep, DisclosureEvaluationResult


class TestDisclosureEvaluator:
    """Test DisclosureEvaluator class."""

    def test_disclosure_evaluator_init_openai(self, mock_env_vars, mock_openai_client):
        """Test DisclosureEvaluator initialization with OpenAI provider."""
        with patch("src.llm.factory.create_llm_provider") as mock_create_provider:
            mock_provider = Mock()
            mock_create_provider.return_value = mock_provider

            with patch("src.config.manager.ConfigManager") as mock_config_class:
                mock_config = Mock()
                mock_config.get_current_provider.return_value = "openai"
                mock_config.get_provider_config.return_value = {"api_key": "test-key"}
                mock_config_class.return_value = mock_config

                with patch("main.CriterionEvaluator") as mock_criterion_class:
                    with patch("main.ResultAggregator") as mock_aggregator_class:
                        with patch("main.logging.getLogger"):
                            evaluator = DisclosureEvaluator(provider="openai")

                            assert evaluator.llm_provider == mock_provider
                            assert evaluator.correlation_id.startswith("eval_")
                            assert evaluator.criterion_evaluator is not None
                            assert evaluator.result_aggregator is not None
                            mock_create_provider.assert_called_once_with("openai")

    def test_disclosure_evaluator_init_anthropic(
        self, mock_env_vars, mock_anthropic_client
    ):
        """Test DisclosureEvaluator initialization with Anthropic provider."""
        with patch("src.llm.factory.create_llm_provider") as mock_create_provider:
            mock_provider = Mock()
            mock_create_provider.return_value = mock_provider

            with patch("src.config.manager.ConfigManager") as mock_config_class:
                mock_config = Mock()
                mock_config.get_current_provider.return_value = "anthropic"
                mock_config.get_provider_config.return_value = {"api_key": "test-key"}
                mock_config_class.return_value = mock_config

                with patch("main.CriterionEvaluator") as mock_criterion_class:
                    with patch("main.ResultAggregator") as mock_aggregator_class:
                        with patch("main.logging.getLogger"):
                            evaluator = DisclosureEvaluator(provider="anthropic")

                            assert evaluator.llm_provider == mock_provider
                            mock_create_provider.assert_called_once_with("anthropic")

    def test_disclosure_evaluator_init_bedrock(
        self, mock_env_vars, mock_bedrock_client
    ):
        """Test DisclosureEvaluator initialization with Bedrock provider."""
        with patch("src.llm.factory.create_llm_provider") as mock_create_provider:
            mock_provider = Mock()
            mock_create_provider.return_value = mock_provider

            with patch("src.config.manager.ConfigManager") as mock_config_class:
                mock_config = Mock()
                mock_config.get_current_provider.return_value = "bedrock"
                mock_config.get_provider_config.return_value = {}
                mock_config_class.return_value = mock_config

                with patch("main.CriterionEvaluator") as mock_criterion_class:
                    with patch("main.ResultAggregator") as mock_aggregator_class:
                        with patch("main.logging.getLogger"):
                            evaluator = DisclosureEvaluator(provider="bedrock")

                            assert evaluator.llm_provider == mock_provider
                            mock_create_provider.assert_called_once_with("bedrock")

    def test_disclosure_evaluator_init_missing_openai_key(self, mock_anthropic_client):
        """Test DisclosureEvaluator initialization with missing OpenAI API key."""
        with patch.dict("os.environ", {}, clear=True):
            with patch("src.config.manager.ConfigManager") as mock_config_class:
                mock_config = Mock()
                mock_config.get_current_provider.return_value = "openai"
                mock_config.get_provider_config.return_value = {}
                mock_config_class.return_value = mock_config

                with pytest.raises(
                    ValueError, match="OPENAI_API_KEY environment variable is required"
                ):
                    DisclosureEvaluator(provider="openai")

    def test_disclosure_evaluator_init_missing_anthropic_key(self, mock_openai_client):
        """Test DisclosureEvaluator initialization with missing Anthropic API key."""
        with patch.dict("os.environ", {}, clear=True):
            with patch("src.config.manager.ConfigManager") as mock_config_class:
                mock_config = Mock()
                mock_config.get_current_provider.return_value = "anthropic"
                mock_config.get_provider_config.return_value = {}
                mock_config_class.return_value = mock_config

                with pytest.raises(
                    ValueError,
                    match="ANTHROPIC_API_KEY environment variable is required",
                ):
                    DisclosureEvaluator(provider="anthropic")

    def test_disclosure_evaluator_init_default_provider(
        self, mock_env_vars, mock_openai_client
    ):
        """Test DisclosureEvaluator initialization with default provider."""
        with patch("src.llm.factory.create_llm_provider") as mock_create_provider:
            mock_provider = Mock()
            mock_create_provider.return_value = mock_provider

            with patch("src.config.manager.ConfigManager") as mock_config_class:
                mock_config = Mock()
                mock_config.get_current_provider.return_value = "openai"
                mock_config.get_provider_config.return_value = {"api_key": "test-key"}
                mock_config_class.return_value = mock_config

                with patch("main.CriterionEvaluator") as mock_criterion_class:
                    with patch("main.ResultAggregator") as mock_aggregator_class:
                        with patch("main.logging.getLogger"):
                            evaluator = DisclosureEvaluator()  # No provider specified

                            assert evaluator.llm_provider == mock_provider
                            mock_create_provider.assert_called_once_with("openai")

    def test_evaluate_criterion_parallel_success(
        self, mock_env_vars, mock_openai_client
    ):
        """Test successful parallel criterion evaluation."""
        mock_llm_provider = Mock()

        criterion = {
            "id": "article_5_1",
            "name": "個人情報保護",
            "article": "第5条第1号",
            "evaluation_steps": ["ステップ1", "ステップ2"],
        }

        eval_context = {"input_text": "テスト入力"}

        # Mock criterion evaluator
        mock_criterion_evaluator = Mock()
        mock_criterion_evaluation = CriterionEvaluation(
            criterion_id="article_5_1",
            criterion_name="個人情報保護",
            article="第5条第1号",
            steps=[EvaluationStep(step="ステップ1", result="YES", reasoning="理由1")],
            score=3,
            score_reasoning="スコア理由",
        )
        mock_criterion_evaluator.evaluate_criterion.return_value = (
            mock_criterion_evaluation
        )

        with patch(
            "src.llm.factory.create_llm_provider", return_value=mock_llm_provider
        ):
            with patch("main.ConfigManager"):
                with patch(
                    "main.CriterionEvaluator", return_value=mock_criterion_evaluator
                ):
                    with patch("main.ResultAggregator"):
                        with patch("main.logging.getLogger") as mock_get_logger:
                            mock_logger = Mock()
                            mock_get_logger.return_value = mock_logger

                            evaluator = DisclosureEvaluator(provider="openai")
                            result = evaluator._evaluate_criterion_parallel(
                                criterion, eval_context
                            )

                            assert result == mock_criterion_evaluation
                            mock_criterion_evaluator.evaluate_criterion.assert_called_once_with(
                                criterion, eval_context
                            )
                            mock_logger.info.assert_called()

    def test_evaluate_criterion_parallel_exception(
        self, mock_env_vars, mock_openai_client
    ):
        """Test parallel criterion evaluation with exception."""
        mock_llm_provider = Mock()

        criterion = {"id": "article_5_1", "name": "テスト", "article": "テスト条項"}
        eval_context = {"input_text": "テスト入力"}

        # Mock criterion evaluator to raise exception
        mock_criterion_evaluator = Mock()
        mock_criterion_evaluator.evaluate_criterion.side_effect = Exception(
            "Evaluation failed"
        )

        with patch(
            "src.llm.factory.create_llm_provider", return_value=mock_llm_provider
        ):
            with patch("main.ConfigManager"):
                with patch(
                    "main.CriterionEvaluator", return_value=mock_criterion_evaluator
                ):
                    with patch("main.ResultAggregator"):
                        with patch("main.logging.getLogger") as mock_get_logger:
                            mock_logger = Mock()
                            mock_get_logger.return_value = mock_logger

                            evaluator = DisclosureEvaluator(provider="openai")
                            result = evaluator._evaluate_criterion_parallel(
                                criterion, eval_context
                            )

                            # Should return fallback evaluation
                            assert isinstance(result, CriterionEvaluation)
                            assert result.criterion_id == "article_5_1"
                            assert result.score == 3
                            assert "並列評価エラー" in result.score_reasoning
                            mock_logger.error.assert_called()

    def test_evaluate_criteria_parallel_success(
        self, mock_env_vars, mock_openai_client
    ):
        """Test successful parallel criteria evaluation."""
        mock_llm_provider = Mock()

        criteria = [
            {"id": "article_5_1", "name": "個人情報保護", "article": "第5条第1号"},
            {"id": "article_5_2", "name": "法人等情報保護", "article": "第5条第2号"},
        ]

        eval_context = {"input_text": "テスト入力"}

        # Mock criterion evaluations
        mock_criterion_evaluations = [
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
        ]

        with patch(
            "src.llm.factory.create_llm_provider", return_value=mock_llm_provider
        ):
            with patch("src.config.manager.ConfigManager") as mock_config_class:
                mock_config = Mock()
                mock_config.get_evaluation_config.return_value = {
                    "parallel": {"enabled": True, "max_workers": 3}
                }
                mock_config_class.return_value = mock_config

                with patch("main.CriterionEvaluator") as mock_criterion_class:
                    mock_criterion_evaluator = Mock()
                    mock_criterion_evaluator.evaluate_criterion.side_effect = (
                        mock_criterion_evaluations
                    )
                    mock_criterion_class.return_value = mock_criterion_evaluator

                    with patch("main.ResultAggregator"):
                        with patch("main.logging.getLogger"):
                            with patch(
                                "main.ThreadPoolExecutor"
                            ) as mock_executor_class:
                                mock_executor = Mock()
                                mock_executor.__enter__ = Mock(
                                    return_value=mock_executor
                                )
                                mock_executor.__exit__ = Mock()
                                mock_executor.submit.return_value = Mock()
                                mock_executor_class.return_value = mock_executor

                                with patch("main.as_completed") as mock_as_completed:
                                    mock_future1 = Mock()
                                    mock_future1.result.return_value = (
                                        mock_criterion_evaluations[0]
                                    )
                                    mock_future2 = Mock()
                                    mock_future2.result.return_value = (
                                        mock_criterion_evaluations[1]
                                    )
                                    mock_as_completed.return_value = [
                                        mock_future1,
                                        mock_future2,
                                    ]

                                    evaluator = DisclosureEvaluator(provider="openai")
                                    results = evaluator._evaluate_criteria_parallel(
                                        criteria, eval_context
                                    )

                                    assert len(results) == 2
                                    assert results[0].criterion_id == "article_5_1"
                                    assert results[1].criterion_id == "article_5_2"

    def test_evaluate_criteria_parallel_disabled(
        self, mock_env_vars, mock_openai_client
    ):
        """Test parallel criteria evaluation when parallel processing is disabled."""
        mock_llm_provider = Mock()

        criteria = [{"id": "article_5_1", "name": "テスト", "article": "テスト条項"}]
        eval_context = {"input_text": "テスト入力"}

        with patch(
            "src.llm.factory.create_llm_provider", return_value=mock_llm_provider
        ):
            with patch("src.config.manager.ConfigManager") as mock_config_class:
                mock_config = Mock()
                mock_config.get_evaluation_config.return_value = {
                    "parallel": {"enabled": False}
                }
                mock_config_class.return_value = mock_config

                with patch("main.CriterionEvaluator"):
                    with patch("main.ResultAggregator"):
                        with patch("main.logging.getLogger"):
                            with patch.object(
                                DisclosureEvaluator, "_evaluate_criteria_sequential"
                            ) as mock_sequential:
                                mock_sequential.return_value = []

                                evaluator = DisclosureEvaluator(provider="openai")
                                results = evaluator._evaluate_criteria_parallel(
                                    criteria, eval_context
                                )

                                mock_sequential.assert_called_once_with(
                                    criteria, eval_context
                                )

    def test_evaluate_criteria_parallel_exception(
        self, mock_env_vars, mock_openai_client
    ):
        """Test parallel criteria evaluation with exception."""
        mock_llm_provider = Mock()

        criteria = [{"id": "article_5_1", "name": "テスト", "article": "テスト条項"}]
        eval_context = {"input_text": "テスト入力"}

        with patch(
            "src.llm.factory.create_llm_provider", return_value=mock_llm_provider
        ):
            with patch("src.config.manager.ConfigManager") as mock_config_class:
                mock_config = Mock()
                mock_config.get_evaluation_config.return_value = {
                    "parallel": {"enabled": True, "max_workers": 3}
                }
                mock_config_class.return_value = mock_config

                with patch("main.CriterionEvaluator"):
                    with patch("main.ResultAggregator"):
                        with patch("main.logging.getLogger") as mock_get_logger:
                            mock_logger = Mock()
                            mock_get_logger.return_value = mock_logger

                            with patch(
                                "main.ThreadPoolExecutor"
                            ) as mock_executor_class:
                                mock_executor = Mock()
                                mock_executor.__enter__ = Mock(
                                    return_value=mock_executor
                                )
                                mock_executor.__exit__ = Mock()
                                mock_executor.submit.side_effect = Exception(
                                    "ThreadPool error"
                                )
                                mock_executor_class.return_value = mock_executor

                                evaluator = DisclosureEvaluator(provider="openai")

                                with pytest.raises(
                                    Exception,
                                    match="Parallel criteria evaluation failed: ThreadPool error",
                                ):
                                    evaluator._evaluate_criteria_parallel(
                                        criteria, eval_context
                                    )

                                mock_logger.error.assert_called()

    def test_evaluate_criteria_sequential_success(
        self, mock_env_vars, mock_openai_client
    ):
        """Test successful sequential criteria evaluation."""
        mock_llm_provider = Mock()

        criteria = [
            {"id": "article_5_1", "name": "個人情報保護", "article": "第5条第1号"},
            {"id": "article_5_2", "name": "法人等情報保護", "article": "第5条第2号"},
        ]

        eval_context = {"input_text": "テスト入力"}

        # Mock criterion evaluations
        mock_criterion_evaluations = [
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
        ]

        with patch(
            "src.llm.factory.create_llm_provider", return_value=mock_llm_provider
        ):
            with patch("main.ConfigManager"):
                with patch("main.CriterionEvaluator") as mock_criterion_class:
                    mock_criterion_evaluator = Mock()
                    mock_criterion_evaluator.evaluate_criterion.side_effect = (
                        mock_criterion_evaluations
                    )
                    mock_criterion_class.return_value = mock_criterion_evaluator

                    with patch("main.ResultAggregator"):
                        with patch("main.logging.getLogger"):
                            evaluator = DisclosureEvaluator(provider="openai")
                            results = evaluator._evaluate_criteria_sequential(
                                criteria, eval_context
                            )

                            assert len(results) == 2
                            assert results[0].criterion_id == "article_5_1"
                            assert results[1].criterion_id == "article_5_2"
                            assert (
                                mock_criterion_evaluator.evaluate_criterion.call_count
                                == 2
                            )

    def test_evaluate_criteria_sequential_with_error(
        self, mock_env_vars, mock_openai_client
    ):
        """Test sequential criteria evaluation with error handling."""
        mock_llm_provider = Mock()

        criteria = [
            {"id": "article_5_1", "name": "個人情報保護", "article": "第5条第1号"},
            {"id": "article_5_2", "name": "法人等情報保護", "article": "第5条第2号"},
        ]

        eval_context = {"input_text": "テスト入力"}

        # Mock criterion evaluator to fail on second criterion
        mock_criterion_evaluator = Mock()
        mock_criterion_evaluation = CriterionEvaluation(
            criterion_id="article_5_1",
            criterion_name="個人情報保護",
            article="第5条第1号",
            steps=[EvaluationStep(step="ステップ1", result="YES", reasoning="理由1")],
            score=2,
            score_reasoning="スコア理由1",
        )
        mock_criterion_evaluator.evaluate_criterion.side_effect = [
            mock_criterion_evaluation,
            Exception("Second criterion failed"),
        ]

        with patch(
            "src.llm.factory.create_llm_provider", return_value=mock_llm_provider
        ):
            with patch("main.ConfigManager"):
                with patch(
                    "main.CriterionEvaluator", return_value=mock_criterion_evaluator
                ):
                    with patch("main.ResultAggregator"):
                        with patch("main.logging.getLogger") as mock_get_logger:
                            mock_logger = Mock()
                            mock_get_logger.return_value = mock_logger

                            evaluator = DisclosureEvaluator(provider="openai")

                            with pytest.raises(
                                Exception,
                                match="Sequential criteria evaluation failed: Second criterion failed",
                            ):
                                evaluator._evaluate_criteria_sequential(
                                    criteria, eval_context
                                )

                            mock_logger.error.assert_called()

    def test_evaluate_disclosure_success(
        self, mock_env_vars, mock_openai_client, sample_criteria
    ):
        """Test successful disclosure evaluation."""
        mock_llm_provider = Mock()

        input_text = "テスト入力"
        context = "テストコンテキスト"
        output_text = "テスト出力"

        # Mock criterion evaluations
        mock_criterion_evaluations = [
            CriterionEvaluation(
                criterion_id="article_5_1",
                criterion_name="個人情報保護",
                article="第5条第1号",
                steps=[
                    EvaluationStep(step="ステップ1", result="YES", reasoning="理由1")
                ],
                score=2,
                score_reasoning="スコア理由1",
            )
        ]

        # Mock result aggregation
        mock_result = DisclosureEvaluationResult(
            input_text=input_text,
            context=context,
            output_text=output_text,
            criterion_evaluations=mock_criterion_evaluations,
            evaluation_timestamp="2025-01-01T12:00:00",
        )

        with patch(
            "src.llm.factory.create_llm_provider", return_value=mock_llm_provider
        ):
            with patch("main.ConfigManager"):
                with patch("main.CriterionEvaluator"):
                    with patch("main.ResultAggregator") as mock_aggregator_class:
                        mock_aggregator = Mock()
                        mock_aggregator.aggregate_results.return_value = mock_result
                        mock_aggregator_class.return_value = mock_aggregator

                        with patch("main.logging.getLogger"):
                            with patch(
                                "main.load_criteria", return_value=sample_criteria
                            ):
                                evaluator = DisclosureEvaluator(provider="openai")
                                result = evaluator.evaluate_disclosure(
                                    input_text, context, output_text
                                )

                                assert result == mock_result
                                mock_aggregator.aggregate_results.assert_called_once()

    def test_evaluate_disclosure_exception(
        self, mock_env_vars, mock_openai_client, sample_criteria
    ):
        """Test disclosure evaluation with exception."""
        mock_llm_provider = Mock()

        input_text = "テスト入力"
        context = "テストコンテキスト"
        output_text = "テスト出力"

        with patch(
            "src.llm.factory.create_llm_provider", return_value=mock_llm_provider
        ):
            with patch("main.ConfigManager"):
                with patch("main.CriterionEvaluator"):
                    with patch("main.ResultAggregator"):
                        with patch("main.logging.getLogger") as mock_get_logger:
                            mock_logger = Mock()
                            mock_get_logger.return_value = mock_logger

                            with patch(
                                "main.load_criteria",
                                side_effect=Exception("Criteria loading failed"),
                            ):
                                evaluator = DisclosureEvaluator(provider="openai")

                                with pytest.raises(
                                    Exception,
                                    match="Disclosure evaluation failed: Criteria loading failed",
                                ):
                                    evaluator.evaluate_disclosure(
                                        input_text, context, output_text
                                    )

                                mock_logger.error.assert_called()

    def test_evaluate_disclosure_legacy_function(
        self, mock_env_vars, mock_openai_client, sample_criteria
    ):
        """Test evaluate_disclosure legacy function."""
        from main import evaluate_disclosure

        input_text = "テスト入力"
        context = "テストコンテキスト"
        output_text = "テスト出力"

        with patch("main.DisclosureEvaluator") as mock_evaluator_class:
            mock_evaluator = Mock()
            mock_result = DisclosureEvaluationResult(
                input_text=input_text,
                context=context,
                output_text=output_text,
                criterion_evaluations=[],
                evaluation_timestamp="2025-01-01T12:00:00",
            )
            mock_evaluator.evaluate_disclosure.return_value = mock_result
            mock_evaluator_class.return_value = mock_evaluator

            result = evaluate_disclosure(
                input_text, context, output_text, provider="openai"
            )

            assert result == mock_result
            mock_evaluator_class.assert_called_once_with(None, "openai")
            mock_evaluator.evaluate_disclosure.assert_called_once_with(
                input_text, context, output_text
            )

    def test_format_structured_output_json(self, sample_document_result):
        """Test format_structured_output with JSON format."""
        from main import format_structured_output

        result = DisclosureEvaluationResult(
            input_text="テスト入力",
            context="テストコンテキスト",
            output_text="テスト出力",
            criterion_evaluations=[],
            evaluation_timestamp="2025-01-01T12:00:00",
        )

        formatted = format_structured_output(result, "json")

        assert isinstance(formatted, str)
        assert "テスト入力" in formatted
        assert "テストコンテキスト" in formatted
        assert "テスト出力" in formatted

    def test_format_structured_output_summary(self, sample_document_result):
        """Test format_structured_output with summary format."""
        from main import format_structured_output

        criterion_evaluation = CriterionEvaluation(
            criterion_id="article_5_1",
            criterion_name="個人情報保護",
            article="第5条第1号",
            steps=[EvaluationStep(step="ステップ1", result="YES", reasoning="理由1")],
            score=2,
            score_reasoning="スコア理由",
        )

        result = DisclosureEvaluationResult(
            input_text="テスト入力",
            context="テストコンテキスト",
            output_text="テスト出力",
            criterion_evaluations=[criterion_evaluation],
            evaluation_timestamp="2025-01-01T12:00:00",
        )

        formatted = format_structured_output(result, "summary")

        assert isinstance(formatted, str)
        assert "情報公開法評価結果" in formatted
        assert "テスト入力" in formatted
        assert "個人情報保護" in formatted
        assert "スコア: 2/5" in formatted
        assert "⚠️" in formatted  # Low score warning

    def test_format_structured_output_unsupported_format(self, sample_document_result):
        """Test format_structured_output with unsupported format."""
        from main import format_structured_output

        result = DisclosureEvaluationResult(
            input_text="テスト入力",
            context="テストコンテキスト",
            output_text="テスト出力",
            criterion_evaluations=[],
            evaluation_timestamp="2025-01-01T12:00:00",
        )

        with pytest.raises(ValueError, match="Unsupported format type: invalid"):
            format_structured_output(result, "invalid")

    def test_format_structured_output_high_score_no_warning(
        self, sample_document_result
    ):
        """Test format_structured_output with high score (no warning)."""
        from main import format_structured_output

        criterion_evaluation = CriterionEvaluation(
            criterion_id="article_5_1",
            criterion_name="個人情報保護",
            article="第5条第1号",
            steps=[EvaluationStep(step="ステップ1", result="NO", reasoning="理由1")],
            score=5,
            score_reasoning="スコア理由",
        )

        result = DisclosureEvaluationResult(
            input_text="テスト入力",
            context="テストコンテキスト",
            output_text="テスト出力",
            criterion_evaluations=[criterion_evaluation],
            evaluation_timestamp="2025-01-01T12:00:00",
        )

        formatted = format_structured_output(result, "summary")

        assert "スコア: 5/5" in formatted
        assert "⚠️" not in formatted  # No warning for high score
