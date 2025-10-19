"""
Unit tests for CriterionEvaluator class.
Tests lines 827-1279: CriterionEvaluator including score calculation and reasoning generation
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from main import CriterionEvaluator, EvaluationStep, CriterionEvaluation, StepEvaluator


class TestCriterionEvaluator:
    """Test CriterionEvaluator class."""

    def test_criterion_evaluator_init(self, mock_llm_response):
        """Test CriterionEvaluator initialization."""
        mock_llm_provider = Mock()
        correlation_id = "test_correlation_001"

        with patch("main.logging.getLogger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            with patch("main.StepEvaluator") as mock_step_evaluator_class:
                mock_step_evaluator = Mock()
                mock_step_evaluator_class.return_value = mock_step_evaluator

                evaluator = CriterionEvaluator(mock_llm_provider, correlation_id)

                assert evaluator.llm_provider == mock_llm_provider
                assert evaluator.correlation_id == correlation_id
                assert evaluator.step_evaluator == mock_step_evaluator
                assert evaluator.logger == mock_logger
                mock_step_evaluator_class.assert_called_once_with(
                    mock_llm_provider, correlation_id
                )
                mock_get_logger.assert_called_once_with(
                    f"CriterionEvaluator.{correlation_id}"
                )

    def test_criterion_evaluator_evaluate_criterion_success(
        self, sample_criterion, mock_llm_response
    ):
        """Test successful criterion evaluation."""
        mock_llm_provider = Mock()
        mock_llm_provider.generate_response.return_value = "スコア理由: テスト理由"

        context = {
            "input_text": "テスト入力",
            "context": "テストコンテキスト",
            "output_text": "テスト出力",
        }

        # Mock step evaluator
        mock_step_evaluator = Mock()
        mock_step_evaluator.evaluate_step.side_effect = [
            EvaluationStep(step="ステップ1", result="YES", reasoning="理由1"),
            EvaluationStep(step="ステップ2", result="NO", reasoning="理由2"),
        ]

        with patch("main.logging.getLogger"):
            with patch("main.StepEvaluator", return_value=mock_step_evaluator):
                with patch("main.config_manager") as mock_config:
                    mock_config.get_prompt.return_value = "システムプロンプト"

                    evaluator = CriterionEvaluator(
                        mock_llm_provider, "test_correlation"
                    )
                    result = evaluator.evaluate_criterion(sample_criterion, context)

                    assert isinstance(result, CriterionEvaluation)
                    assert result.criterion_id == sample_criterion["id"]
                    assert result.criterion_name == sample_criterion["name"]
                    assert result.article == sample_criterion["article"]
                    assert len(result.steps) == 2
                    assert result.score == 4  # 1 YES out of 2 steps = score 4
                    assert result.score_reasoning == "テスト理由"

                    # Verify step evaluator was called for each step
                    assert mock_step_evaluator.evaluate_step.call_count == 2

    def test_criterion_evaluator_evaluate_criterion_exception(
        self, sample_criterion, mock_llm_response
    ):
        """Test criterion evaluation with exception."""
        mock_llm_provider = Mock()

        context = {"input_text": "テスト入力"}

        # Mock step evaluator to raise exception
        mock_step_evaluator = Mock()
        mock_step_evaluator.evaluate_step.side_effect = Exception(
            "Step evaluation failed"
        )

        with patch("main.logging.getLogger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            with patch("main.StepEvaluator", return_value=mock_step_evaluator):
                with patch("main.config_manager") as mock_config:
                    mock_config.get_prompt.return_value = "システムプロンプト"

                    evaluator = CriterionEvaluator(
                        mock_llm_provider, "test_correlation"
                    )

                    with pytest.raises(Exception, match="Step evaluation failed"):
                        evaluator.evaluate_criterion(sample_criterion, context)

                    mock_logger.error.assert_called()

    def test_criterion_evaluator_evaluate_steps_sequential_success(
        self, sample_criterion, mock_llm_response
    ):
        """Test successful sequential step evaluation."""
        mock_llm_provider = Mock()

        context = {"input_text": "テスト入力"}

        # Mock step evaluator
        mock_step_evaluator = Mock()
        mock_step_evaluator.evaluate_step.side_effect = [
            EvaluationStep(step="ステップ1", result="YES", reasoning="理由1"),
            EvaluationStep(step="ステップ2", result="NO", reasoning="理由2"),
        ]

        with patch("main.logging.getLogger"):
            with patch("main.StepEvaluator", return_value=mock_step_evaluator):
                with patch("main.config_manager") as mock_config:
                    mock_config.get_prompt.return_value = "システムプロンプト"

                    evaluator = CriterionEvaluator(
                        mock_llm_provider, "test_correlation"
                    )
                    steps = evaluator._evaluate_steps_sequential(
                        sample_criterion, context
                    )

                    assert len(steps) == 2
                    assert steps[0].result == "YES"
                    assert steps[1].result == "NO"
                    assert mock_step_evaluator.evaluate_step.call_count == 2

    def test_criterion_evaluator_evaluate_steps_sequential_with_error(
        self, sample_criterion, mock_llm_response
    ):
        """Test sequential step evaluation with error handling."""
        mock_llm_provider = Mock()

        context = {"input_text": "テスト入力"}

        # Mock step evaluator to fail on second step
        mock_step_evaluator = Mock()
        mock_step_evaluator.evaluate_step.side_effect = [
            EvaluationStep(step="ステップ1", result="YES", reasoning="理由1"),
            Exception("Step evaluation failed"),
        ]

        with patch("main.logging.getLogger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            with patch("main.StepEvaluator", return_value=mock_step_evaluator):
                with patch("main.config_manager") as mock_config:
                    mock_config.get_prompt.return_value = "システムプロンプト"

                    evaluator = CriterionEvaluator(
                        mock_llm_provider, "test_correlation"
                    )
                    steps = evaluator._evaluate_steps_sequential(
                        sample_criterion, context
                    )

                    assert len(steps) == 2
                    assert steps[0].result == "YES"
                    assert steps[1].result == "NO"  # Fallback step
                    assert "順次ステップ評価エラー" in steps[1].reasoning
                    mock_logger.error.assert_called()

    def test_calculate_score_all_no(self, mock_llm_response):
        """Test score calculation when all steps are NO."""
        mock_llm_provider = Mock()

        steps = [
            EvaluationStep(step="ステップ1", result="NO", reasoning="理由1"),
            EvaluationStep(step="ステップ2", result="NO", reasoning="理由2"),
            EvaluationStep(step="ステップ3", result="NO", reasoning="理由3"),
        ]

        criterion = {"id": "test", "name": "テスト", "article": "テスト条項"}

        with patch("main.logging.getLogger"):
            evaluator = CriterionEvaluator(mock_llm_provider, "test_correlation")
            score, reasoning = evaluator._calculate_score(steps, criterion)

            assert score == 5  # No criteria met - disclose
            assert "テスト理由" in reasoning or "スコア計算エラー" in reasoning

    def test_calculate_score_all_yes(self, mock_llm_response):
        """Test score calculation when all steps are YES."""
        mock_llm_provider = Mock()

        steps = [
            EvaluationStep(step="ステップ1", result="YES", reasoning="理由1"),
            EvaluationStep(step="ステップ2", result="YES", reasoning="理由2"),
            EvaluationStep(step="ステップ3", result="YES", reasoning="理由3"),
        ]

        criterion = {"id": "test", "name": "テスト", "article": "テスト条項"}

        with patch("main.logging.getLogger"):
            evaluator = CriterionEvaluator(mock_llm_provider, "test_correlation")
            score, reasoning = evaluator._calculate_score(steps, criterion)

            assert score == 1  # All criteria met - non-disclosure
            assert "テスト理由" in reasoning or "スコア計算エラー" in reasoning

    def test_calculate_score_70_percent_yes(self, mock_llm_response):
        """Test score calculation when 70% of steps are YES."""
        mock_llm_provider = Mock()

        steps = [
            EvaluationStep(step="ステップ1", result="YES", reasoning="理由1"),
            EvaluationStep(step="ステップ2", result="YES", reasoning="理由2"),
            EvaluationStep(step="ステップ3", result="NO", reasoning="理由3"),
        ]  # 2 out of 3 = 66.7% < 70%, so score should be 3

        criterion = {"id": "test", "name": "テスト", "article": "テスト条項"}

        with patch("main.logging.getLogger"):
            evaluator = CriterionEvaluator(mock_llm_provider, "test_correlation")
            score, reasoning = evaluator._calculate_score(steps, criterion)

            assert score == 3  # Some criteria met - unclear
            assert "テスト理由" in reasoning or "スコア計算エラー" in reasoning

    def test_calculate_score_30_percent_yes(self, mock_llm_response):
        """Test score calculation when 30% of steps are YES."""
        mock_llm_provider = Mock()

        steps = [
            EvaluationStep(step="ステップ1", result="YES", reasoning="理由1"),
            EvaluationStep(step="ステップ2", result="NO", reasoning="理由2"),
            EvaluationStep(step="ステップ3", result="NO", reasoning="理由3"),
            EvaluationStep(step="ステップ4", result="NO", reasoning="理由4"),
        ]  # 1 out of 4 = 25% < 30%, so score should be 4

        criterion = {"id": "test", "name": "テスト", "article": "テスト条項"}

        with patch("main.logging.getLogger"):
            evaluator = CriterionEvaluator(mock_llm_provider, "test_correlation")
            score, reasoning = evaluator._calculate_score(steps, criterion)

            assert score == 4  # Few criteria met - likely disclosure
            assert "テスト理由" in reasoning or "スコア計算エラー" in reasoning

    def test_calculate_score_exception_handling(self, mock_llm_response):
        """Test score calculation exception handling."""
        mock_llm_provider = Mock()

        steps = [EvaluationStep(step="ステップ1", result="YES", reasoning="理由1")]

        criterion = {"id": "test", "name": "テスト", "article": "テスト条項"}

        with patch("main.logging.getLogger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            # Mock _generate_score_reasoning to raise exception
            with patch.object(
                CriterionEvaluator,
                "_generate_score_reasoning",
                side_effect=Exception("Reasoning failed"),
            ):
                evaluator = CriterionEvaluator(mock_llm_provider, "test_correlation")
                score, reasoning = evaluator._calculate_score(steps, criterion)

                assert score == 3  # Fallback score
                assert "スコア計算エラー" in reasoning
                mock_logger.error.assert_called()

    def test_generate_score_reasoning_success(self, mock_llm_response):
        """Test successful score reasoning generation."""
        mock_llm_provider = Mock()
        mock_llm_provider.generate_response.return_value = "スコア理由: テスト理由"

        steps = [
            EvaluationStep(step="ステップ1", result="YES", reasoning="理由1"),
            EvaluationStep(step="ステップ2", result="NO", reasoning="理由2"),
        ]

        criterion = {"id": "test", "name": "テスト", "article": "テスト条項"}

        with patch("main.logging.getLogger"):
            with patch("main.config_manager") as mock_config:
                mock_config.get_prompt.return_value = "システムプロンプト"

                evaluator = CriterionEvaluator(mock_llm_provider, "test_correlation")
                reasoning = evaluator._generate_score_reasoning(steps, criterion, 3)

                assert reasoning == "テスト理由"
                mock_llm_provider.generate_response.assert_called_once()

    def test_generate_score_reasoning_exception_fallback_score_1_2(
        self, mock_llm_response
    ):
        """Test score reasoning exception fallback for scores 1-2."""
        mock_llm_provider = Mock()
        mock_llm_provider.generate_response.side_effect = Exception("LLM Error")

        steps = [EvaluationStep(step="ステップ1", result="YES", reasoning="理由1")]
        criterion = {"id": "test", "name": "テスト", "article": "テスト条項"}

        with patch("main.logging.getLogger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            with patch("main.config_manager") as mock_config:
                mock_config.get_prompt.return_value = "システムプロンプト"

                evaluator = CriterionEvaluator(mock_llm_provider, "test_correlation")

                # Test score 1
                reasoning = evaluator._generate_score_reasoning(steps, criterion, 1)
                assert "不開示事由に該当するため不開示が適切" in reasoning
                assert "1点のスコアが算出された" in reasoning

                # Test score 2
                reasoning = evaluator._generate_score_reasoning(steps, criterion, 2)
                assert "不開示事由に該当するため不開示が適切" in reasoning
                assert "2点のスコアが算出された" in reasoning

                mock_logger.error.assert_called()

    def test_generate_score_reasoning_exception_fallback_score_3(
        self, mock_llm_response
    ):
        """Test score reasoning exception fallback for score 3."""
        mock_llm_provider = Mock()
        mock_llm_provider.generate_response.side_effect = Exception("LLM Error")

        steps = [EvaluationStep(step="ステップ1", result="YES", reasoning="理由1")]
        criterion = {"id": "test", "name": "テスト", "article": "テスト条項"}

        with patch("main.logging.getLogger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            with patch("main.config_manager") as mock_config:
                mock_config.get_prompt.return_value = "システムプロンプト"

                evaluator = CriterionEvaluator(mock_llm_provider, "test_correlation")
                reasoning = evaluator._generate_score_reasoning(steps, criterion, 3)

                assert "慎重な法的検討が必要" in reasoning
                assert "3点のスコアが算出された" in reasoning
                mock_logger.error.assert_called()

    def test_generate_score_reasoning_exception_fallback_score_4_5(
        self, mock_llm_response
    ):
        """Test score reasoning exception fallback for scores 4-5."""
        mock_llm_provider = Mock()
        mock_llm_provider.generate_response.side_effect = Exception("LLM Error")

        steps = [EvaluationStep(step="ステップ1", result="NO", reasoning="理由1")]
        criterion = {"id": "test", "name": "テスト", "article": "テスト条項"}

        with patch("main.logging.getLogger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            with patch("main.config_manager") as mock_config:
                mock_config.get_prompt.return_value = "システムプロンプト"

                evaluator = CriterionEvaluator(mock_llm_provider, "test_correlation")

                # Test score 4
                reasoning = evaluator._generate_score_reasoning(steps, criterion, 4)
                assert "不開示事由に該当しないため開示が適切" in reasoning
                assert "4点のスコアが算出された" in reasoning

                # Test score 5
                reasoning = evaluator._generate_score_reasoning(steps, criterion, 5)
                assert "不開示事由に該当しないため開示が適切" in reasoning
                assert "5点のスコアが算出された" in reasoning

                mock_logger.error.assert_called()

    def test_prepare_step_analysis(self, mock_llm_response):
        """Test step analysis preparation."""
        mock_llm_provider = Mock()

        steps = [
            EvaluationStep(step="ステップ1", result="YES", reasoning="理由1"),
            EvaluationStep(step="ステップ2", result="NO", reasoning="理由2"),
            EvaluationStep(step="ステップ3", result="YES", reasoning="理由3"),
        ]

        criterion = {"id": "test", "name": "テスト", "article": "テスト条項"}

        with patch("main.logging.getLogger"):
            evaluator = CriterionEvaluator(mock_llm_provider, "test_correlation")
            analysis = evaluator._prepare_step_analysis(steps, criterion, 3)

            assert "評価結果サマリー" in analysis
            assert "総段階数: 3" in analysis
            assert "YES評価: 2段階" in analysis
            assert "NO評価: 1段階" in analysis
            assert "YES率: 66.7%" in analysis
            assert "各段階の詳細評価" in analysis
            assert "段階1: ✅ YES" in analysis
            assert "段階2: ❌ NO" in analysis
            assert "段階3: ✅ YES" in analysis
            assert "法的根拠分析" in analysis
            assert "該当条文: テスト条項" in analysis

    def test_parse_reasoning_response_success(self, mock_llm_response):
        """Test successful reasoning response parsing."""
        mock_llm_provider = Mock()

        response = "スコア理由: テスト理由"

        with patch("main.logging.getLogger"):
            evaluator = CriterionEvaluator(mock_llm_provider, "test_correlation")
            reasoning = evaluator._parse_reasoning_response(response)

            assert reasoning == "テスト理由"

    def test_parse_reasoning_response_fallback(self, mock_llm_response):
        """Test reasoning response parsing fallback."""
        mock_llm_provider = Mock()

        response = "これは長い説明文で、明確な理由が含まれています。"

        with patch("main.logging.getLogger"):
            evaluator = CriterionEvaluator(mock_llm_provider, "test_correlation")
            reasoning = evaluator._parse_reasoning_response(response)

            assert reasoning == response  # Should return full content as fallback

    def test_parse_reasoning_response_various_formats(self, mock_llm_response):
        """Test reasoning response parsing with various formats."""
        mock_llm_provider = Mock()

        test_cases = [
            ("スコア理由: テスト理由", "テスト理由"),
            ("Reasoning: Test reasoning", "Test reasoning"),
            ("根拠: テスト根拠", "テスト根拠"),
            ("説明: テスト説明", "テスト説明"),
        ]

        with patch("main.logging.getLogger"):
            evaluator = CriterionEvaluator(mock_llm_provider, "test_correlation")

            for response, expected in test_cases:
                reasoning = evaluator._parse_reasoning_response(response)
                assert reasoning == expected

    def test_parse_reasoning_response_missing_colon(self, mock_llm_response):
        """Test reasoning response parsing when colon is missing."""
        mock_llm_provider = Mock()

        response = "スコア理由 テスト理由"  # Missing colon

        with patch("main.logging.getLogger"):
            evaluator = CriterionEvaluator(mock_llm_provider, "test_correlation")
            reasoning = evaluator._parse_reasoning_response(response)

            assert reasoning == response  # Should return full content as fallback

    def test_parse_reasoning_response_exception_handling(self, mock_llm_response):
        """Test reasoning response parsing exception handling."""
        mock_llm_provider = Mock()

        with patch("main.logging.getLogger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            evaluator = CriterionEvaluator(mock_llm_provider, "test_correlation")

            # Test with None response
            reasoning = evaluator._parse_reasoning_response(None)
            assert reasoning == ""  # Should return empty string
            mock_logger.error.assert_called()
