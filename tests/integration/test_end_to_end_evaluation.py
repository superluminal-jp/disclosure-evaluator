"""
Integration tests for complete evaluation flow.
Tests the full evaluation pipeline from input to output
"""

import json
import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from src.evaluators import DisclosureEvaluator, evaluate_disclosure
from src.utils import format_structured_output
from src.models import DisclosureEvaluationResult, CriterionEvaluation, EvaluationStep


class TestEndToEndEvaluation:
    """Test complete evaluation flow."""

    def test_end_to_end_evaluation_openai(
        self, mock_env_vars, mock_openai_client, sample_criteria
    ):
        """Test complete evaluation flow with OpenAI provider."""
        input_text = "個人情報: 田中太郎、住所: 東京都渋谷区恵比寿1-1-1"
        context = "情報公開請求に対する検討"
        output_text = "開示判断の結果"

        # Mock OpenAI responses
        mock_responses = [
            "結果: YES\n理由: 個人情報保護規定に該当する可能性が高い",
            "結果: NO\n理由: 法人等情報保護規定に該当しない",
            "スコア理由: 各段階の評価結果を総合的に分析した結果、不開示事由に該当するため不開示が適切",
            "スコア理由: 各段階の評価結果を総合的に分析した結果、不開示事由に該当しないため開示が適切",
        ]

        with patch("main.load_criteria", return_value=sample_criteria):
            with patch("main.OpenAIProvider") as mock_provider_class:
                mock_provider = Mock()
                mock_provider.generate_response.side_effect = mock_responses
                mock_provider_class.return_value = mock_provider

                with patch("main.ConfigManager") as mock_config_class:
                    mock_config = Mock()
                    mock_config.get_current_provider.return_value = "openai"
                    mock_config.get_provider_config.return_value = {
                        "model": "gpt-4",
                        "temperature": 0.1,
                        "max_tokens": 2000,
                    }
                    mock_config.get_prompt.return_value = "システムプロンプト"
                    mock_config.get_evaluation_config.return_value = {
                        "parallel": {"enabled": True, "max_workers": 3}
                    }
                    mock_config.get_output_config.return_value = {"format": "json"}
                    mock_config_class.return_value = mock_config

                    with patch("main.logging.getLogger"):
                        result = evaluate_disclosure(
                            input_text, context, output_text, provider="openai"
                        )

                        assert isinstance(result, DisclosureEvaluationResult)
                        assert result.input_text == input_text
                        assert result.context == context
                        assert result.output_text == output_text
                        assert len(result.criterion_evaluations) == 2
                        assert result.evaluation_timestamp is not None

                        # Verify criterion evaluations
                        criterion_ids = [
                            c.criterion_id for c in result.criterion_evaluations
                        ]
                        assert "article_5_1" in criterion_ids
                        assert "article_5_2" in criterion_ids

    def test_end_to_end_evaluation_anthropic(
        self, mock_env_vars, mock_anthropic_client, sample_criteria
    ):
        """Test complete evaluation flow with Anthropic provider."""
        input_text = "法人等情報: 株式会社テスト、代表者: 山田花子"
        context = "情報公開請求に対する検討"
        output_text = "開示判断の結果"

        # Mock Anthropic responses
        mock_responses = [
            "結果: NO\n理由: 個人情報保護規定に該当しない",
            "結果: YES\n理由: 法人等情報保護規定に該当する可能性が高い",
            "スコア理由: 各段階の評価結果を総合的に分析した結果、不開示事由に該当しないため開示が適切",
            "スコア理由: 各段階の評価結果を総合的に分析した結果、不開示事由に該当するため不開示が適切",
        ]

        with patch("main.load_criteria", return_value=sample_criteria):
            with patch("main.AnthropicProvider") as mock_provider_class:
                mock_provider = Mock()
                mock_provider.generate_response.side_effect = mock_responses
                mock_provider_class.return_value = mock_provider

                with patch("main.ConfigManager") as mock_config_class:
                    mock_config = Mock()
                    mock_config.get_current_provider.return_value = "anthropic"
                    mock_config.get_provider_config.return_value = {
                        "model": "claude-3",
                        "temperature": 0.1,
                        "max_tokens": 2000,
                    }
                    mock_config.get_prompt.return_value = "システムプロンプト"
                    mock_config.get_evaluation_config.return_value = {
                        "parallel": {"enabled": True, "max_workers": 3}
                    }
                    mock_config.get_output_config.return_value = {"format": "json"}
                    mock_config_class.return_value = mock_config

                    with patch("main.logging.getLogger"):
                        result = evaluate_disclosure(
                            input_text, context, output_text, provider="anthropic"
                        )

                        assert isinstance(result, DisclosureEvaluationResult)
                        assert result.input_text == input_text
                        assert result.context == context
                        assert result.output_text == output_text
                        assert len(result.criterion_evaluations) == 2

    def test_end_to_end_evaluation_bedrock(
        self, mock_env_vars, mock_bedrock_client, sample_criteria
    ):
        """Test complete evaluation flow with Bedrock provider."""
        input_text = "公共情報: 公共事業の入札結果"
        context = "情報公開請求に対する検討"
        output_text = "開示判断の結果"

        # Mock Bedrock responses
        mock_responses = [
            "結果: NO\n理由: 個人情報保護規定に該当しない",
            "結果: NO\n理由: 法人等情報保護規定に該当しない",
            "スコア理由: 各段階の評価結果を総合的に分析した結果、不開示事由に該当しないため開示が適切",
            "スコア理由: 各段階の評価結果を総合的に分析した結果、不開示事由に該当しないため開示が適切",
        ]

        with patch("main.load_criteria", return_value=sample_criteria):
            with patch("main.BedrockAnthropicProvider") as mock_provider_class:
                mock_provider = Mock()
                mock_provider.generate_response.side_effect = mock_responses
                mock_provider_class.return_value = mock_provider

                with patch("main.ConfigManager") as mock_config_class:
                    mock_config = Mock()
                    mock_config.get_current_provider.return_value = "bedrock"
                    mock_config.get_provider_config.return_value = {
                        "model": "claude-3",
                        "temperature": 0.1,
                        "max_tokens": 2000,
                    }
                    mock_config.get_prompt.return_value = "システムプロンプト"
                    mock_config.get_evaluation_config.return_value = {
                        "parallel": {"enabled": True, "max_workers": 3}
                    }
                    mock_config.get_output_config.return_value = {"format": "json"}
                    mock_config_class.return_value = mock_config

                    with patch("main.logging.getLogger"):
                        result = evaluate_disclosure(
                            input_text, context, output_text, provider="bedrock"
                        )

                        assert isinstance(result, DisclosureEvaluationResult)
                        assert result.input_text == input_text
                        assert result.context == context
                        assert result.output_text == output_text
                        assert len(result.criterion_evaluations) == 2

    def test_end_to_end_evaluation_sequential_processing(
        self, mock_env_vars, mock_openai_client, sample_criteria
    ):
        """Test complete evaluation flow with sequential processing."""
        input_text = "テスト入力"
        context = "テストコンテキスト"
        output_text = "テスト出力"

        # Mock OpenAI responses
        mock_responses = [
            "結果: YES\n理由: 個人情報保護規定に該当する可能性が高い",
            "結果: NO\n理由: 法人等情報保護規定に該当しない",
            "スコア理由: 各段階の評価結果を総合的に分析した結果、不開示事由に該当するため不開示が適切",
            "スコア理由: 各段階の評価結果を総合的に分析した結果、不開示事由に該当しないため開示が適切",
        ]

        with patch("main.load_criteria", return_value=sample_criteria):
            with patch("main.OpenAIProvider") as mock_provider_class:
                mock_provider = Mock()
                mock_provider.generate_response.side_effect = mock_responses
                mock_provider_class.return_value = mock_provider

                with patch("main.ConfigManager") as mock_config_class:
                    mock_config = Mock()
                    mock_config.get_current_provider.return_value = "openai"
                    mock_config.get_provider_config.return_value = {
                        "model": "gpt-4",
                        "temperature": 0.1,
                        "max_tokens": 2000,
                    }
                    mock_config.get_prompt.return_value = "システムプロンプト"
                    mock_config.get_evaluation_config.return_value = {
                        "parallel": {"enabled": False}  # Sequential processing
                    }
                    mock_config.get_output_config.return_value = {"format": "json"}
                    mock_config_class.return_value = mock_config

                    with patch("main.logging.getLogger"):
                        result = evaluate_disclosure(
                            input_text, context, output_text, provider="openai"
                        )

                        assert isinstance(result, DisclosureEvaluationResult)
                        assert result.input_text == input_text
                        assert result.context == context
                        assert result.output_text == output_text
                        assert len(result.criterion_evaluations) == 2

    def test_end_to_end_evaluation_json_output(
        self, mock_env_vars, mock_openai_client, sample_criteria
    ):
        """Test complete evaluation flow with JSON output format."""
        input_text = "テスト入力"
        context = "テストコンテキスト"
        output_text = "テスト出力"

        # Mock OpenAI responses
        mock_responses = [
            "結果: YES\n理由: 個人情報保護規定に該当する可能性が高い",
            "結果: NO\n理由: 法人等情報保護規定に該当しない",
            "スコア理由: 各段階の評価結果を総合的に分析した結果、不開示事由に該当するため不開示が適切",
            "スコア理由: 各段階の評価結果を総合的に分析した結果、不開示事由に該当しないため開示が適切",
        ]

        with patch("main.load_criteria", return_value=sample_criteria):
            with patch("main.OpenAIProvider") as mock_provider_class:
                mock_provider = Mock()
                mock_provider.generate_response.side_effect = mock_responses
                mock_provider_class.return_value = mock_provider

                with patch("main.ConfigManager") as mock_config_class:
                    mock_config = Mock()
                    mock_config.get_current_provider.return_value = "openai"
                    mock_config.get_provider_config.return_value = {
                        "model": "gpt-4",
                        "temperature": 0.1,
                        "max_tokens": 2000,
                    }
                    mock_config.get_prompt.return_value = "システムプロンプト"
                    mock_config.get_evaluation_config.return_value = {
                        "parallel": {"enabled": True, "max_workers": 3}
                    }
                    mock_config.get_output_config.return_value = {"format": "json"}
                    mock_config_class.return_value = mock_config

                    with patch("main.logging.getLogger"):
                        result = evaluate_disclosure(
                            input_text, context, output_text, provider="openai"
                        )

                        # Test JSON output formatting
                        json_output = format_structured_output(result, "json")

                        assert isinstance(json_output, str)
                        json_data = json.loads(json_output)
                        assert json_data["input_text"] == input_text
                        assert json_data["context"] == context
                        assert json_data["output_text"] == output_text
                        assert "criterion_evaluations" in json_data
                        assert "evaluation_timestamp" in json_data

    def test_end_to_end_evaluation_summary_output(
        self, mock_env_vars, mock_openai_client, sample_criteria
    ):
        """Test complete evaluation flow with summary output format."""
        input_text = "テスト入力"
        context = "テストコンテキスト"
        output_text = "テスト出力"

        # Mock OpenAI responses
        mock_responses = [
            "結果: YES\n理由: 個人情報保護規定に該当する可能性が高い",
            "結果: NO\n理由: 法人等情報保護規定に該当しない",
            "スコア理由: 各段階の評価結果を総合的に分析した結果、不開示事由に該当するため不開示が適切",
            "スコア理由: 各段階の評価結果を総合的に分析した結果、不開示事由に該当しないため開示が適切",
        ]

        with patch("main.load_criteria", return_value=sample_criteria):
            with patch("main.OpenAIProvider") as mock_provider_class:
                mock_provider = Mock()
                mock_provider.generate_response.side_effect = mock_responses
                mock_provider_class.return_value = mock_provider

                with patch("main.ConfigManager") as mock_config_class:
                    mock_config = Mock()
                    mock_config.get_current_provider.return_value = "openai"
                    mock_config.get_provider_config.return_value = {
                        "model": "gpt-4",
                        "temperature": 0.1,
                        "max_tokens": 2000,
                    }
                    mock_config.get_prompt.return_value = "システムプロンプト"
                    mock_config.get_evaluation_config.return_value = {
                        "parallel": {"enabled": True, "max_workers": 3}
                    }
                    mock_config.get_output_config.return_value = {"format": "summary"}
                    mock_config_class.return_value = mock_config

                    with patch("main.logging.getLogger"):
                        result = evaluate_disclosure(
                            input_text, context, output_text, provider="openai"
                        )

                        # Test summary output formatting
                        summary_output = format_structured_output(result, "summary")

                        assert isinstance(summary_output, str)
                        assert "情報公開法評価結果" in summary_output
                        assert input_text in summary_output
                        assert context in summary_output
                        assert output_text in summary_output
                        assert "個人情報保護" in summary_output
                        assert "法人等情報保護" in summary_output
                        assert "スコア" in summary_output

    def test_end_to_end_evaluation_with_unicode_content(
        self, mock_env_vars, mock_openai_client, sample_criteria
    ):
        """Test complete evaluation flow with Unicode content."""
        input_text = (
            "個人情報: 田中太郎、住所: 東京都渋谷区恵比寿1-1-1、電話: 03-1234-5678"
        )
        context = "情報公開請求に対する検討"
        output_text = "開示判断の結果"

        # Mock OpenAI responses
        mock_responses = [
            "結果: YES\n理由: 個人情報保護規定に該当する可能性が高い",
            "結果: NO\n理由: 法人等情報保護規定に該当しない",
            "スコア理由: 各段階の評価結果を総合的に分析した結果、不開示事由に該当するため不開示が適切",
            "スコア理由: 各段階の評価結果を総合的に分析した結果、不開示事由に該当しないため開示が適切",
        ]

        with patch("main.load_criteria", return_value=sample_criteria):
            with patch("main.OpenAIProvider") as mock_provider_class:
                mock_provider = Mock()
                mock_provider.generate_response.side_effect = mock_responses
                mock_provider_class.return_value = mock_provider

                with patch("main.ConfigManager") as mock_config_class:
                    mock_config = Mock()
                    mock_config.get_current_provider.return_value = "openai"
                    mock_config.get_provider_config.return_value = {
                        "model": "gpt-4",
                        "temperature": 0.1,
                        "max_tokens": 2000,
                    }
                    mock_config.get_prompt.return_value = "システムプロンプト"
                    mock_config.get_evaluation_config.return_value = {
                        "parallel": {"enabled": True, "max_workers": 3}
                    }
                    mock_config.get_output_config.return_value = {"format": "json"}
                    mock_config_class.return_value = mock_config

                    with patch("main.logging.getLogger"):
                        result = evaluate_disclosure(
                            input_text, context, output_text, provider="openai"
                        )

                        assert isinstance(result, DisclosureEvaluationResult)
                        assert result.input_text == input_text
                        assert result.context == context
                        assert result.output_text == output_text

                        # Test Unicode handling in output
                        json_output = format_structured_output(result, "json")
                        assert "田中太郎" in json_output
                        assert "東京都渋谷区" in json_output
                        assert "03-1234-5678" in json_output

    def test_end_to_end_evaluation_error_handling(
        self, mock_env_vars, mock_openai_client, sample_criteria
    ):
        """Test complete evaluation flow with error handling."""
        input_text = "テスト入力"
        context = "テストコンテキスト"
        output_text = "テスト出力"

        with patch("main.load_criteria", return_value=sample_criteria):
            with patch("main.OpenAIProvider") as mock_provider_class:
                mock_provider = Mock()
                mock_provider.generate_response.side_effect = Exception("API Error")
                mock_provider_class.return_value = mock_provider

                with patch("main.ConfigManager") as mock_config_class:
                    mock_config = Mock()
                    mock_config.get_current_provider.return_value = "openai"
                    mock_config.get_provider_config.return_value = {
                        "model": "gpt-4",
                        "temperature": 0.1,
                        "max_tokens": 2000,
                    }
                    mock_config.get_prompt.return_value = "システムプロンプト"
                    mock_config.get_evaluation_config.return_value = {
                        "parallel": {"enabled": True, "max_workers": 3}
                    }
                    mock_config.get_output_config.return_value = {"format": "json"}
                    mock_config_class.return_value = mock_config

                    with patch("main.logging.getLogger"):
                        # The evaluation should complete successfully with error handling
                        result = evaluate_disclosure(
                            input_text, context, output_text, provider="openai"
                        )

                        # Verify the result was created despite the API error
                        assert result is not None
                        assert isinstance(result, DisclosureEvaluationResult)
                        assert result.input_text == input_text
                        assert result.context == context

    def test_end_to_end_evaluation_criteria_loading_error(
        self, mock_env_vars, mock_openai_client
    ):
        """Test complete evaluation flow with criteria loading error."""
        input_text = "テスト入力"
        context = "テストコンテキスト"
        output_text = "テスト出力"

        with patch(
            "main.load_criteria", side_effect=Exception("Criteria loading failed")
        ):
            with patch("main.logging.getLogger"):
                with pytest.raises(Exception, match="Criteria loading failed"):
                    evaluate_disclosure(
                        input_text, context, output_text, provider="openai"
                    )

    def test_end_to_end_evaluation_missing_api_key(self, mock_anthropic_client):
        """Test complete evaluation flow with missing API key."""
        input_text = "テスト入力"
        context = "テストコンテキスト"
        output_text = "テスト出力"

        with patch.dict("os.environ", {}, clear=True):
            with patch("main.ConfigManager") as mock_config_class:
                mock_config = Mock()
                mock_config.get_current_provider.return_value = "openai"
                mock_config.get_provider_config.return_value = {}
                mock_config_class.return_value = mock_config

                with patch("main.logging.getLogger"):
                    with pytest.raises(
                        ValueError,
                        match="OPENAI_API_KEY environment variable is required",
                    ):
                        evaluate_disclosure(
                            input_text, context, output_text, provider="openai"
                        )

    def test_end_to_end_evaluation_large_input(
        self, mock_env_vars, mock_openai_client, sample_criteria
    ):
        """Test complete evaluation flow with large input."""
        # Create large input text
        large_input = "これは非常に長いテスト文書です。" * 1000  # ~10KB of text
        context = "テストコンテキスト"
        output_text = "テスト出力"

        # Mock OpenAI responses
        mock_responses = [
            "結果: NO\n理由: 個人情報保護規定に該当しない",
            "結果: NO\n理由: 法人等情報保護規定に該当しない",
            "スコア理由: 各段階の評価結果を総合的に分析した結果、不開示事由に該当しないため開示が適切",
            "スコア理由: 各段階の評価結果を総合的に分析した結果、不開示事由に該当しないため開示が適切",
        ]

        with patch("main.load_criteria", return_value=sample_criteria):
            with patch("main.OpenAIProvider") as mock_provider_class:
                mock_provider = Mock()
                mock_provider.generate_response.side_effect = mock_responses
                mock_provider_class.return_value = mock_provider

                with patch("main.ConfigManager") as mock_config_class:
                    mock_config = Mock()
                    mock_config.get_current_provider.return_value = "openai"
                    mock_config.get_provider_config.return_value = {
                        "model": "gpt-4",
                        "temperature": 0.1,
                        "max_tokens": 2000,
                    }
                    mock_config.get_prompt.return_value = "システムプロンプト"
                    mock_config.get_evaluation_config.return_value = {
                        "parallel": {"enabled": True, "max_workers": 3}
                    }
                    mock_config.get_output_config.return_value = {"format": "json"}
                    mock_config_class.return_value = mock_config

                    with patch("main.logging.getLogger"):
                        result = evaluate_disclosure(
                            large_input, context, output_text, provider="openai"
                        )

                        assert isinstance(result, DisclosureEvaluationResult)
                        assert result.input_text == large_input
                        assert result.context == context
                        assert result.output_text == output_text
                        assert len(result.criterion_evaluations) == 2
