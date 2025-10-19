"""
Unit tests for StepEvaluator class.
Tests lines 700-825: StepEvaluator class including parsing logic
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from main import StepEvaluator, EvaluationStep, ConfigManager


class TestStepEvaluator:
    """Test StepEvaluator class."""

    def test_step_evaluator_init(self, mock_llm_response):
        """Test StepEvaluator initialization."""
        mock_llm_provider = Mock()
        correlation_id = "test_correlation_001"

        with patch("main.logging.getLogger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            evaluator = StepEvaluator(mock_llm_provider, correlation_id)

            assert evaluator.llm_provider == mock_llm_provider
            assert evaluator.correlation_id == correlation_id
            assert evaluator.logger == mock_logger
            mock_get_logger.assert_called_once_with(f"StepEvaluator.{correlation_id}")

    def test_step_evaluator_evaluate_step_success(self, mock_llm_response):
        """Test successful step evaluation."""
        mock_llm_provider = Mock()
        mock_llm_provider.generate_response.return_value = "結果: YES\n理由: テスト理由"

        step_prompt = "個人に関する情報かどうかを評価してください。"
        step_name = "個人情報確認"
        context = {
            "input_text": "田中太郎の個人情報",
            "context": "情報公開請求",
            "output_text": "開示判断",
        }

        with patch("main.logging.getLogger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            with patch("main.config_manager") as mock_config:
                mock_config.get_prompt.return_value = "システムプロンプト"

                evaluator = StepEvaluator(mock_llm_provider, "test_correlation")
                result = evaluator.evaluate_step(step_prompt, step_name, context)

                assert isinstance(result, EvaluationStep)
                assert result.step == step_name
                assert result.result == "YES"
                assert result.reasoning == "テスト理由"

                # Verify LLM provider was called with correct messages
                mock_llm_provider.generate_response.assert_called_once()
                call_args = mock_llm_provider.generate_response.call_args[0][0]
                assert call_args[0]["role"] == "system"
                assert call_args[1]["role"] == "user"
                assert step_prompt in call_args[1]["content"]
                assert context["input_text"] in call_args[1]["content"]

                # Verify logging
                mock_logger.info.assert_called()
                mock_logger.info.assert_any_call(
                    f"Starting evaluation step: {step_name}"
                )
                mock_logger.info.assert_any_call(f"Step {step_name} completed: YES")

    def test_step_evaluator_evaluate_step_exception(self, mock_llm_response):
        """Test step evaluation with exception."""
        mock_llm_provider = Mock()
        mock_llm_provider.generate_response.side_effect = Exception("LLM API Error")

        step_prompt = "テストプロンプト"
        step_name = "テストステップ"
        context = {"input_text": "テスト入力"}

        with patch("main.logging.getLogger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            with patch("main.config_manager") as mock_config:
                mock_config.get_prompt.return_value = "システムプロンプト"

                evaluator = StepEvaluator(mock_llm_provider, "test_correlation")

                with pytest.raises(
                    ValueError,
                    match=f"Step evaluation failed for {step_name}: LLM API Error",
                ):
                    evaluator.evaluate_step(step_prompt, step_name, context)

                # Verify error logging
                mock_logger.error.assert_called_with(
                    f"Step evaluation failed for {step_name}: LLM API Error"
                )

    def test_parse_step_response_yes_variations(self, mock_llm_response):
        """Test parsing YES responses with various formats."""
        mock_llm_provider = Mock()

        test_cases = [
            "結果: YES\n理由: テスト理由",
            "Result: YES\n理由: テスト理由",
            "判定: YES\n理由: テスト理由",
            "判断: YES\n理由: テスト理由",
            "結果: はい\n理由: テスト理由",
            "結果: 該当\n理由: テスト理由",
        ]

        with patch("main.logging.getLogger"):
            evaluator = StepEvaluator(mock_llm_provider, "test_correlation")

            for response in test_cases:
                result = evaluator._parse_step_response(response, "テストステップ")
                assert result.result == "YES"
                assert result.reasoning == "テスト理由"

    def test_parse_step_response_no_variations(self, mock_llm_response):
        """Test parsing NO responses with various formats."""
        mock_llm_provider = Mock()

        test_cases = [
            "結果: NO\n理由: テスト理由",
            "Result: NO\n理由: テスト理由",
            "判定: NO\n理由: テスト理由",
            "判断: NO\n理由: テスト理由",
            "結果: いいえ\n理由: テスト理由",
            "結果: 非該当\n理由: テスト理由",
        ]

        with patch("main.logging.getLogger"):
            evaluator = StepEvaluator(mock_llm_provider, "test_correlation")

            for response in test_cases:
                result = evaluator._parse_step_response(response, "テストステップ")
                assert result.result == "NO"
                assert result.reasoning == "テスト理由"

    def test_parse_step_response_reasoning_variations(self, mock_llm_response):
        """Test parsing reasoning with various formats."""
        mock_llm_provider = Mock()

        test_cases = [
            "結果: YES\n理由: テスト理由",
            "結果: YES\nReasoning: Test reasoning",
            "結果: YES\n根拠: テスト根拠",
            "結果: YES\n説明: テスト説明",
        ]

        with patch("main.logging.getLogger"):
            evaluator = StepEvaluator(mock_llm_provider, "test_correlation")

            for response in test_cases:
                result = evaluator._parse_step_response(response, "テストステップ")
                assert result.result == "YES"
                # Extract reasoning after colon
                expected_reasoning = response.split("\n")[1].split(":", 1)[1].strip()
                assert result.reasoning == expected_reasoning

    def test_parse_step_response_fallback_reasoning(self, mock_llm_response):
        """Test fallback reasoning when parsing fails."""
        mock_llm_provider = Mock()

        # Response without clear reasoning pattern
        response = "これは長い説明文で、明確な理由が含まれています。"

        with patch("main.logging.getLogger"):
            evaluator = StepEvaluator(mock_llm_provider, "test_correlation")
            result = evaluator._parse_step_response(response, "テストステップ")

            assert result.result == "NO"  # Default to NO
            assert (
                result.reasoning == "これは長い説明文で、明確な理由が含まれています。"
            )

    def test_parse_step_response_meaningful_lines_fallback(self, mock_llm_response):
        """Test fallback to meaningful lines when reasoning parsing fails."""
        mock_llm_provider = Mock()

        # Response with multiple lines, some meaningful
        response = """# Header
        短い行
        これは非常に長い意味のある説明文で、詳細な理由が含まれています。
        # Another header
        別の短い行"""

        with patch("main.logging.getLogger"):
            evaluator = StepEvaluator(mock_llm_provider, "test_correlation")
            result = evaluator._parse_step_response(response, "テストステップ")

            assert result.result == "NO"  # Default to NO
            # Should use the longest meaningful line
            assert (
                "これは非常に長い意味のある説明文で、詳細な理由が含まれています。"
                in result.reasoning
            )

    def test_parse_step_response_exception_handling(self, mock_llm_response):
        """Test exception handling in response parsing."""
        mock_llm_provider = Mock()

        with patch("main.logging.getLogger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            evaluator = StepEvaluator(mock_llm_provider, "test_correlation")

            # Test with None response (should not happen but test robustness)
            result = evaluator._parse_step_response(None, "テストステップ")

            assert result.result == "NO"
            assert (
                result.reasoning
                == "解析エラー: 'NoneType' object has no attribute 'strip'"
            )
            mock_logger.error.assert_called()

    def test_parse_step_response_empty_response(self, mock_llm_response):
        """Test parsing empty response."""
        mock_llm_provider = Mock()

        with patch("main.logging.getLogger"):
            evaluator = StepEvaluator(mock_llm_provider, "test_correlation")
            result = evaluator._parse_step_response("", "テストステップ")

            assert result.result == "NO"
            assert result.reasoning == "解析に失敗しました"

    def test_parse_step_response_whitespace_handling(self, mock_llm_response):
        """Test parsing response with whitespace."""
        mock_llm_provider = Mock()

        response = "  結果: YES  \n  理由: テスト理由  "

        with patch("main.logging.getLogger"):
            evaluator = StepEvaluator(mock_llm_provider, "test_correlation")
            result = evaluator._parse_step_response(response, "テストステップ")

            assert result.result == "YES"
            assert result.reasoning == "テスト理由"  # Should be trimmed

    def test_parse_step_response_multiple_yes_no_patterns(self, mock_llm_response):
        """Test parsing response with multiple YES/NO patterns."""
        mock_llm_provider = Mock()

        # Response with multiple patterns - should use the first one found
        response = "結果: NO\n理由: 最初の理由\n結果: YES\n理由: 二番目の理由"

        with patch("main.logging.getLogger"):
            evaluator = StepEvaluator(mock_llm_provider, "test_correlation")
            result = evaluator._parse_step_response(response, "テストステップ")

            assert result.result == "NO"  # First pattern found
            assert result.reasoning == "最初の理由"

    def test_parse_step_response_case_insensitive(self, mock_llm_response):
        """Test parsing response with case insensitive matching."""
        mock_llm_provider = Mock()

        test_cases = [
            "結果: yes\n理由: テスト理由",
            "結果: no\n理由: テスト理由",
            "結果: Yes\n理由: テスト理由",
            "結果: No\n理由: テスト理由",
        ]

        with patch("main.logging.getLogger"):
            evaluator = StepEvaluator(mock_llm_provider, "test_correlation")

            for i, response in enumerate(test_cases):
                result = evaluator._parse_step_response(response, "テストステップ")
                expected_result = "YES" if i % 2 == 0 else "NO"
                assert result.result == expected_result
                assert result.reasoning == "テスト理由"

    def test_parse_step_response_japanese_keywords(self, mock_llm_response):
        """Test parsing response with Japanese keywords."""
        mock_llm_provider = Mock()

        test_cases = [
            ("結果: はい\n理由: テスト理由", "YES"),
            ("結果: いいえ\n理由: テスト理由", "NO"),
            ("結果: 該当\n理由: テスト理由", "YES"),
            ("結果: 非該当\n理由: テスト理由", "NO"),
        ]

        with patch("main.logging.getLogger"):
            evaluator = StepEvaluator(mock_llm_provider, "test_correlation")

            for response, expected_result in test_cases:
                result = evaluator._parse_step_response(response, "テストステップ")
                assert result.result == expected_result
                assert result.reasoning == "テスト理由"

    def test_parse_step_response_missing_colon_in_reasoning(self, mock_llm_response):
        """Test parsing response where reasoning line doesn't have colon."""
        mock_llm_provider = Mock()

        response = "結果: YES\n理由 テスト理由"  # Missing colon after "理由"

        with patch("main.logging.getLogger"):
            evaluator = StepEvaluator(mock_llm_provider, "test_correlation")
            result = evaluator._parse_step_response(response, "テストステップ")

            assert result.result == "YES"
            assert result.reasoning == "理由 テスト理由"  # Should use the whole line

    def test_parse_step_response_complex_reasoning(self, mock_llm_response):
        """Test parsing response with complex reasoning."""
        mock_llm_provider = Mock()

        response = """結果: YES
理由: この文書には個人情報が含まれており、情報公開法第5条第1号の不開示事由に該当する可能性が高い。
詳細な分析により、氏名、住所等の個人識別情報が明確に記載されているため、開示することにより個人の権利利益を害するおそれがあると判断される。"""

        with patch("main.logging.getLogger"):
            evaluator = StepEvaluator(mock_llm_provider, "test_correlation")
            result = evaluator._parse_step_response(response, "テストステップ")

            assert result.result == "YES"
            assert "個人情報が含まれており" in result.reasoning
            assert "情報公開法第5条第1号" in result.reasoning
