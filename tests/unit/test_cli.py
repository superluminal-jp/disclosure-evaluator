"""
Unit tests for CLI command handlers.
Tests lines 2486-2872: main(), handle_* functions
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
from io import StringIO

from main import (
    main,
    print_usage,
    handle_single_document_evaluation,
    handle_batch_command,
    handle_batch_status_command,
    handle_batch_results_command,
    handle_resume_batch_command,
    handle_retry_documents_command,
    evaluate_disclosure,
    format_structured_output,
)


class TestMainFunction:
    """Test main() function."""

    def test_main_no_arguments(self, mock_sys_exit):
        """Test main() with no arguments."""
        with patch("sys.argv", ["main.py"]):
            with patch("main.print_usage") as mock_print_usage:
                main()

                mock_print_usage.assert_called_once()
                mock_sys_exit.assert_called_once_with(1)

    def test_main_batch_command(self, mock_sys_exit):
        """Test main() with batch command."""
        with patch("sys.argv", ["main.py", "--batch", "--folder", "/test/path"]):
            with patch("main.handle_batch_command") as mock_handle_batch:
                main()

                mock_handle_batch.assert_called_once()

    def test_main_batch_status_command(self, mock_sys_exit):
        """Test main() with batch-status command."""
        with patch("sys.argv", ["main.py", "--batch-status", "batch_001"]):
            with patch("main.handle_batch_status_command") as mock_handle_status:
                main()

                mock_handle_status.assert_called_once()

    def test_main_batch_results_command(self, mock_sys_exit):
        """Test main() with batch-results command."""
        with patch("sys.argv", ["main.py", "--batch-results", "batch_001"]):
            with patch("main.handle_batch_results_command") as mock_handle_results:
                main()

                mock_handle_results.assert_called_once()

    def test_main_resume_batch_command(self, mock_sys_exit):
        """Test main() with resume-batch command."""
        with patch("sys.argv", ["main.py", "--resume-batch", "batch_001"]):
            with patch("main.handle_resume_batch_command") as mock_handle_resume:
                main()

                mock_handle_resume.assert_called_once()

    def test_main_retry_documents_command(self, mock_sys_exit):
        """Test main() with retry-documents command."""
        with patch(
            "sys.argv", ["main.py", "--retry-documents", "batch_001", "doc_001,doc_002"]
        ):
            with patch("main.handle_retry_documents_command") as mock_handle_retry:
                main()

                mock_handle_retry.assert_called_once()

    def test_main_single_document_evaluation(self, mock_sys_exit):
        """Test main() with single document evaluation."""
        with patch(
            "sys.argv", ["main.py", "test_input", "test_context", "test_output"]
        ):
            with patch("main.handle_single_document_evaluation") as mock_handle_single:
                main()

                mock_handle_single.assert_called_once()


class TestPrintUsage:
    """Test print_usage() function."""

    def test_print_usage_output(self, mock_print):
        """Test print_usage() output."""
        print_usage()

        # Verify print was called
        mock_print.assert_called()

        # Check that usage information was printed
        call_args = mock_print.call_args[0][0]
        assert "Usage: python evaluator.py" in call_args
        assert "Single Document Evaluation:" in call_args
        assert "Batch Processing:" in call_args
        assert "Examples:" in call_args


class TestHandleSingleDocumentEvaluation:
    """Test handle_single_document_evaluation() function."""

    def test_handle_single_document_evaluation_success(self, mock_sys_argv, mock_print):
        """Test successful single document evaluation."""
        with patch(
            "sys.argv", ["main.py", "test_input", "test_context", "test_output"]
        ):
            with patch("main.evaluate_disclosure") as mock_evaluate:
                mock_result = Mock()
                mock_evaluate.return_value = mock_result

                with patch("main.format_structured_output") as mock_format:
                    mock_format.return_value = "Formatted output"

                    handle_single_document_evaluation()

                    mock_evaluate.assert_called_once_with(
                        "test_input", "test_context", "test_output", provider=None
                    )
                    mock_format.assert_called_once_with(mock_result, "json")
                    mock_print.assert_called_with("Formatted output")

    def test_handle_single_document_evaluation_with_format(
        self, mock_sys_argv, mock_print
    ):
        """Test single document evaluation with format argument."""
        with patch(
            "sys.argv",
            [
                "main.py",
                "test_input",
                "test_context",
                "test_output",
                "--format",
                "summary",
            ],
        ):
            with patch("main.evaluate_disclosure") as mock_evaluate:
                mock_result = Mock()
                mock_evaluate.return_value = mock_result

                with patch("main.format_structured_output") as mock_format:
                    mock_format.return_value = "Formatted output"

                    handle_single_document_evaluation()

                    mock_evaluate.assert_called_once_with(
                        "test_input", "test_context", "test_output", provider=None
                    )
                    mock_format.assert_called_once_with(mock_result, "summary")

    def test_handle_single_document_evaluation_with_provider(
        self, mock_sys_argv, mock_print
    ):
        """Test single document evaluation with provider argument."""
        with patch(
            "sys.argv",
            [
                "main.py",
                "test_input",
                "test_context",
                "test_output",
                "--provider",
                "anthropic",
            ],
        ):
            with patch("main.evaluate_disclosure") as mock_evaluate:
                mock_result = Mock()
                mock_evaluate.return_value = mock_result

                with patch("main.format_structured_output") as mock_format:
                    mock_format.return_value = "Formatted output"

                    handle_single_document_evaluation()

                    mock_evaluate.assert_called_once_with(
                        "test_input",
                        "test_context",
                        "test_output",
                        provider="anthropic",
                    )

    def test_handle_single_document_evaluation_exception(
        self, mock_sys_argv, mock_print, mock_sys_exit
    ):
        """Test single document evaluation with exception."""
        with patch(
            "sys.argv", ["main.py", "test_input", "test_context", "test_output"]
        ):
            with patch(
                "main.evaluate_disclosure", side_effect=Exception("Evaluation failed")
            ):
                handle_single_document_evaluation()

                mock_print.assert_called_with(
                    "Error: Evaluation failed", file=sys.stderr
                )
                mock_sys_exit.assert_called_once_with(1)

    def test_handle_single_document_evaluation_minimal_args(
        self, mock_sys_argv, mock_print
    ):
        """Test single document evaluation with minimal arguments."""
        with patch("sys.argv", ["main.py", "test_input"]):
            with patch("main.evaluate_disclosure") as mock_evaluate:
                mock_result = Mock()
                mock_evaluate.return_value = mock_result

                with patch("main.format_structured_output") as mock_format:
                    mock_format.return_value = "Formatted output"

                    handle_single_document_evaluation()

                    mock_evaluate.assert_called_once_with(
                        "test_input", "", "", provider=None
                    )

    def test_handle_single_document_evaluation_format_index_error(
        self, mock_sys_argv, mock_print
    ):
        """Test single document evaluation with format argument index error."""
        with patch("sys.argv", ["main.py", "test_input", "--format"]):
            with patch("main.evaluate_disclosure") as mock_evaluate:
                mock_result = Mock()
                mock_evaluate.return_value = mock_result

                with patch("main.format_structured_output") as mock_format:
                    mock_format.return_value = "Formatted output"

                    handle_single_document_evaluation()

                    # Should use default format when index error occurs
                    mock_format.assert_called_once_with(mock_result, "json")

    def test_handle_single_document_evaluation_provider_index_error(
        self, mock_sys_argv, mock_print
    ):
        """Test single document evaluation with provider argument index error."""
        with patch("sys.argv", ["main.py", "test_input", "--provider"]):
            with patch("main.evaluate_disclosure") as mock_evaluate:
                mock_result = Mock()
                mock_evaluate.return_value = mock_result

                with patch("main.format_structured_output") as mock_format:
                    mock_format.return_value = "Formatted output"

                    handle_single_document_evaluation()

                    # Should use default provider when index error occurs
                    mock_evaluate.assert_called_once_with(
                        "test_input", "", "", provider=None
                    )


class TestHandleBatchCommand:
    """Test handle_batch_command() function."""

    def test_handle_batch_command_folder_success(self, mock_sys_argv, mock_print):
        """Test successful batch command with folder."""
        with patch("sys.argv", ["main.py", "--batch", "--folder", "/test/path"]):
            with patch("main.BatchEvaluator") as mock_evaluator_class:
                mock_evaluator = Mock()
                mock_evaluator.create_batch_from_folder.return_value = "batch_001"
                mock_evaluator.start_batch.return_value = None
                mock_evaluator_class.return_value = mock_evaluator

                handle_batch_command()

                mock_evaluator.create_batch_from_folder.assert_called_once()
                mock_evaluator.start_batch.assert_called_once_with("batch_001")
                mock_print.assert_called()

    def test_handle_batch_command_documents_success(self, mock_sys_argv, mock_print):
        """Test successful batch command with documents."""
        with patch(
            "sys.argv", ["main.py", "--batch", "--documents", "doc1.txt,doc2.txt"]
        ):
            with patch("main.BatchEvaluator") as mock_evaluator_class:
                mock_evaluator = Mock()
                mock_evaluator.create_batch.return_value = "batch_001"
                mock_evaluator.start_batch.return_value = None
                mock_evaluator_class.return_value = mock_evaluator

                with patch("main.DocumentInput") as mock_doc_input:
                    handle_batch_command()

                    mock_evaluator.create_batch.assert_called_once()
                    mock_evaluator.start_batch.assert_called_once_with("batch_001")
                    mock_print.assert_called()

    def test_handle_batch_command_no_folder_or_documents(
        self, mock_sys_argv, mock_print, mock_sys_exit
    ):
        """Test batch command with neither folder nor documents."""
        with patch("sys.argv", ["main.py", "--batch"]):
            handle_batch_command()

            mock_print.assert_called_with(
                "Error: Must specify either --folder or --documents", file=sys.stderr
            )
            mock_sys_exit.assert_called_once_with(1)

    def test_handle_batch_command_with_options(self, mock_sys_argv, mock_print):
        """Test batch command with various options."""
        with patch(
            "sys.argv",
            [
                "main.py",
                "--batch",
                "--folder",
                "/test/path",
                "--recursive",
                "--file-types",
                "text/plain,application/pdf",
                "--exclude",
                "*.tmp,*.log",
                "--max-workers",
                "10",
                "--timeout",
                "600",
                "--retry-attempts",
                "5",
                "--file-size-limit",
                "104857600",
                "--context",
                "Test context",
                "--output-formats",
                "json,csv",
            ],
        ):
            with patch("main.BatchEvaluator") as mock_evaluator_class:
                mock_evaluator = Mock()
                mock_evaluator.create_batch_from_folder.return_value = "batch_001"
                mock_evaluator.start_batch.return_value = None
                mock_evaluator_class.return_value = mock_evaluator

                handle_batch_command()

                # Verify BatchConfiguration was created with custom values
                mock_evaluator_class.assert_called_once()
                config = mock_evaluator_class.call_args[0][0]
                assert config.max_concurrent_workers == 10
                assert config.timeout_seconds == 600
                assert config.max_retry_attempts == 5
                assert config.file_size_limit == 104857600
                assert config.output_formats == ["json", "csv"]

    def test_handle_batch_command_exception(
        self, mock_sys_argv, mock_print, mock_sys_exit
    ):
        """Test batch command with exception."""
        with patch("sys.argv", ["main.py", "--batch", "--folder", "/test/path"]):
            with patch(
                "main.BatchEvaluator", side_effect=Exception("Batch creation failed")
            ):
                handle_batch_command()

                mock_print.assert_called_with(
                    "Error: Batch creation failed", file=sys.stderr
                )
                mock_sys_exit.assert_called_once_with(1)

    def test_handle_batch_command_missing_folder_argument(
        self, mock_sys_argv, mock_print, mock_sys_exit
    ):
        """Test batch command with missing folder argument."""
        with patch("sys.argv", ["main.py", "--batch", "--folder"]):
            with patch("main.BatchEvaluator") as mock_evaluator_class:
                mock_evaluator = Mock()
                mock_evaluator.create_batch_from_folder.return_value = "batch_001"
                mock_evaluator.start_batch.return_value = None
                mock_evaluator_class.return_value = mock_evaluator

                handle_batch_command()

                # Should use None as folder path
                mock_evaluator.create_batch_from_folder.assert_called_once()

    def test_handle_batch_command_missing_documents_argument(
        self, mock_sys_argv, mock_print, mock_sys_exit
    ):
        """Test batch command with missing documents argument."""
        with patch("sys.argv", ["main.py", "--batch", "--documents"]):
            with patch("main.BatchEvaluator") as mock_evaluator_class:
                mock_evaluator = Mock()
                mock_evaluator.create_batch.return_value = "batch_001"
                mock_evaluator.start_batch.return_value = None
                mock_evaluator_class.return_value = mock_evaluator

                handle_batch_command()

                # Should use empty list as documents
                mock_evaluator.create_batch.assert_called_once()


class TestHandleBatchStatusCommand:
    """Test handle_batch_status_command() function."""

    def test_handle_batch_status_command_success(self, mock_sys_argv, mock_print):
        """Test successful batch status command."""
        with patch("sys.argv", ["main.py", "--batch-status", "batch_001"]):
            with patch("main.BatchEvaluator") as mock_evaluator_class:
                mock_evaluator = Mock()
                mock_batch = Mock()
                mock_batch.status.value = "processing"
                mock_batch.total_documents = 10
                mock_batch.processed_documents = 5
                mock_batch.successful_documents = 4
                mock_batch.failed_documents = 1
                mock_evaluator.get_batch.return_value = mock_batch
                mock_evaluator.get_batch_progress.return_value = Mock()
                mock_evaluator_class.return_value = mock_evaluator

                handle_batch_status_command()

                mock_evaluator.get_batch.assert_called_once_with("batch_001")
                mock_evaluator.get_batch_progress.assert_called_once_with("batch_001")
                mock_print.assert_called()

    def test_handle_batch_status_command_batch_not_found(
        self, mock_sys_argv, mock_print, mock_sys_exit
    ):
        """Test batch status command with batch not found."""
        with patch("sys.argv", ["main.py", "--batch-status", "nonexistent_batch"]):
            with patch("main.BatchEvaluator") as mock_evaluator_class:
                mock_evaluator = Mock()
                mock_evaluator.get_batch.return_value = None
                mock_evaluator_class.return_value = mock_evaluator

                handle_batch_status_command()

                mock_print.assert_called_with(
                    "Error: Batch nonexistent_batch not found", file=sys.stderr
                )
                mock_sys_exit.assert_called_once_with(1)

    def test_handle_batch_status_command_missing_batch_id(
        self, mock_sys_argv, mock_print, mock_sys_exit
    ):
        """Test batch status command with missing batch ID."""
        with patch("sys.argv", ["main.py", "--batch-status"]):
            handle_batch_status_command()

            mock_print.assert_called_with(
                "Error: --batch-status requires a batch_id", file=sys.stderr
            )
            mock_sys_exit.assert_called_once_with(1)

    def test_handle_batch_status_command_exception(
        self, mock_sys_argv, mock_print, mock_sys_exit
    ):
        """Test batch status command with exception."""
        with patch("sys.argv", ["main.py", "--batch-status", "batch_001"]):
            with patch(
                "main.BatchEvaluator", side_effect=Exception("Status check failed")
            ):
                handle_batch_status_command()

                mock_print.assert_called_with(
                    "Error: Status check failed", file=sys.stderr
                )
                mock_sys_exit.assert_called_once_with(1)


class TestHandleBatchResultsCommand:
    """Test handle_batch_results_command() function."""

    def test_handle_batch_results_command_success(self, mock_sys_argv, mock_print):
        """Test successful batch results command."""
        with patch("sys.argv", ["main.py", "--batch-results", "batch_001"]):
            with patch("main.BatchEvaluator") as mock_evaluator_class:
                mock_evaluator = Mock()
                mock_result = Mock()
                mock_evaluator.get_batch_results.return_value = mock_result
                mock_evaluator_class.return_value = mock_evaluator

                handle_batch_results_command()

                mock_evaluator.get_batch_results.assert_called_once_with(
                    "batch_001", "json"
                )
                mock_print.assert_called()

    def test_handle_batch_results_command_with_format(self, mock_sys_argv, mock_print):
        """Test batch results command with format argument."""
        with patch(
            "sys.argv", ["main.py", "--batch-results", "batch_001", "--format", "csv"]
        ):
            with patch("main.BatchEvaluator") as mock_evaluator_class:
                mock_evaluator = Mock()
                mock_result = Mock()
                mock_evaluator.get_batch_results.return_value = mock_result
                mock_evaluator_class.return_value = mock_evaluator

                handle_batch_results_command()

                mock_evaluator.get_batch_results.assert_called_once_with(
                    "batch_001", "csv"
                )

    def test_handle_batch_results_command_missing_batch_id(
        self, mock_sys_argv, mock_print, mock_sys_exit
    ):
        """Test batch results command with missing batch ID."""
        with patch("sys.argv", ["main.py", "--batch-results"]):
            handle_batch_results_command()

            mock_print.assert_called_with(
                "Error: --batch-results requires a batch_id", file=sys.stderr
            )
            mock_sys_exit.assert_called_once_with(1)

    def test_handle_batch_results_command_exception(
        self, mock_sys_argv, mock_print, mock_sys_exit
    ):
        """Test batch results command with exception."""
        with patch("sys.argv", ["main.py", "--batch-results", "batch_001"]):
            with patch(
                "main.BatchEvaluator", side_effect=Exception("Results retrieval failed")
            ):
                handle_batch_results_command()

                mock_print.assert_called_with(
                    "Error: Results retrieval failed", file=sys.stderr
                )
                mock_sys_exit.assert_called_once_with(1)


class TestHandleResumeBatchCommand:
    """Test handle_resume_batch_command() function."""

    def test_handle_resume_batch_command_success(self, mock_sys_argv, mock_print):
        """Test successful resume batch command."""
        with patch("sys.argv", ["main.py", "--resume-batch", "batch_001"]):
            with patch("main.BatchEvaluator") as mock_evaluator_class:
                mock_evaluator = Mock()
                mock_evaluator.resume_batch.return_value = None
                mock_evaluator_class.return_value = mock_evaluator

                handle_resume_batch_command()

                mock_evaluator.resume_batch.assert_called_once_with("batch_001")
                mock_print.assert_called()

    def test_handle_resume_batch_command_missing_batch_id(
        self, mock_sys_argv, mock_print, mock_sys_exit
    ):
        """Test resume batch command with missing batch ID."""
        with patch("sys.argv", ["main.py", "--resume-batch"]):
            handle_resume_batch_command()

            mock_print.assert_called_with(
                "Error: --resume-batch requires a batch_id", file=sys.stderr
            )
            mock_sys_exit.assert_called_once_with(1)

    def test_handle_resume_batch_command_exception(
        self, mock_sys_argv, mock_print, mock_sys_exit
    ):
        """Test resume batch command with exception."""
        with patch("sys.argv", ["main.py", "--resume-batch", "batch_001"]):
            with patch("main.BatchEvaluator", side_effect=Exception("Resume failed")):
                handle_resume_batch_command()

                mock_print.assert_called_with("Error: Resume failed", file=sys.stderr)
                mock_sys_exit.assert_called_once_with(1)


class TestHandleRetryDocumentsCommand:
    """Test handle_retry_documents_command() function."""

    def test_handle_retry_documents_command_success(self, mock_sys_argv, mock_print):
        """Test successful retry documents command."""
        with patch(
            "sys.argv", ["main.py", "--retry-documents", "batch_001", "doc_001,doc_002"]
        ):
            with patch("main.BatchEvaluator") as mock_evaluator_class:
                mock_evaluator = Mock()
                mock_evaluator.retry_failed_documents.return_value = None
                mock_evaluator_class.return_value = mock_evaluator

                handle_retry_documents_command()

                mock_evaluator.retry_failed_documents.assert_called_once_with(
                    "batch_001", ["doc_001", "doc_002"]
                )
                mock_print.assert_called()

    def test_handle_retry_documents_command_missing_batch_id(
        self, mock_sys_argv, mock_print, mock_sys_exit
    ):
        """Test retry documents command with missing batch ID."""
        with patch("sys.argv", ["main.py", "--retry-documents"]):
            handle_retry_documents_command()

            mock_print.assert_called_with(
                "Error: --retry-documents requires a batch_id", file=sys.stderr
            )
            mock_sys_exit.assert_called_once_with(1)

    def test_handle_retry_documents_command_missing_document_ids(
        self, mock_sys_argv, mock_print, mock_sys_exit
    ):
        """Test retry documents command with missing document IDs."""
        with patch("sys.argv", ["main.py", "--retry-documents", "batch_001"]):
            handle_retry_documents_command()

            mock_print.assert_called_with(
                "Error: --retry-documents requires document IDs", file=sys.stderr
            )
            mock_sys_exit.assert_called_once_with(1)

    def test_handle_retry_documents_command_exception(
        self, mock_sys_argv, mock_print, mock_sys_exit
    ):
        """Test retry documents command with exception."""
        with patch(
            "sys.argv", ["main.py", "--retry-documents", "batch_001", "doc_001"]
        ):
            with patch("main.BatchEvaluator", side_effect=Exception("Retry failed")):
                handle_retry_documents_command()

                mock_print.assert_called_with("Error: Retry failed", file=sys.stderr)
                mock_sys_exit.assert_called_once_with(1)


class TestEvaluateDisclosureLegacyFunction:
    """Test evaluate_disclosure() legacy function."""

    def test_evaluate_disclosure_legacy_success(
        self, mock_env_vars, mock_openai_client
    ):
        """Test successful evaluate_disclosure legacy function."""
        with patch("main.DisclosureEvaluator") as mock_evaluator_class:
            mock_evaluator = Mock()
            mock_result = Mock()
            mock_evaluator.evaluate_disclosure.return_value = mock_result
            mock_evaluator_class.return_value = mock_evaluator

            result = evaluate_disclosure(
                "test_input", "test_context", "test_output", provider="openai"
            )

            assert result == mock_result
            mock_evaluator_class.assert_called_once_with(None, "openai")
            mock_evaluator.evaluate_disclosure.assert_called_once_with(
                "test_input", "test_context", "test_output"
            )

    def test_evaluate_disclosure_legacy_default_provider(
        self, mock_env_vars, mock_openai_client
    ):
        """Test evaluate_disclosure legacy function with default provider."""
        with patch("main.DisclosureEvaluator") as mock_evaluator_class:
            mock_evaluator = Mock()
            mock_result = Mock()
            mock_evaluator.evaluate_disclosure.return_value = mock_result
            mock_evaluator_class.return_value = mock_evaluator

            result = evaluate_disclosure("test_input", "test_context", "test_output")

            assert result == mock_result
            mock_evaluator_class.assert_called_once_with(None, None)
            mock_evaluator.evaluate_disclosure.assert_called_once_with(
                "test_input", "test_context", "test_output"
            )


class TestFormatStructuredOutput:
    """Test format_structured_output() function."""

    def test_format_structured_output_json(self, sample_document_result):
        """Test format_structured_output with JSON format."""
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
        from main import CriterionEvaluation, EvaluationStep

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
        from main import CriterionEvaluation, EvaluationStep

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
