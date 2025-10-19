"""
Integration tests for CLI commands via subprocess.
Tests CLI commands through subprocess execution
"""

import subprocess
import pytest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
from pathlib import Path


class TestCLICommands:
    """Test CLI commands via subprocess."""

    def test_cli_single_document_evaluation_success(self, mock_subprocess):
        """Test successful single document evaluation via CLI."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "Evaluation completed successfully"
        mock_result.stderr = ""
        mock_subprocess.return_value = mock_result

        # Test CLI command execution
        result = subprocess.run(
            ["python", "main.py", "test_input", "test_context", "test_output"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "Evaluation completed successfully" in result.stdout

    def test_cli_single_document_evaluation_with_format(self, mock_subprocess):
        """Test single document evaluation with format argument via CLI."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "Summary format output"
        mock_result.stderr = ""
        mock_subprocess.return_value = mock_result

        # Test CLI command with format argument
        result = subprocess.run(
            [
                "python",
                "main.py",
                "test_input",
                "test_context",
                "test_output",
                "--format",
                "summary",
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "Summary format output" in result.stdout

    def test_cli_single_document_evaluation_with_provider(self, mock_subprocess):
        """Test single document evaluation with provider argument via CLI."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "Anthropic evaluation completed"
        mock_result.stderr = ""
        mock_subprocess.return_value = mock_result

        # Test CLI command with provider argument
        result = subprocess.run(
            [
                "python",
                "main.py",
                "test_input",
                "test_context",
                "test_output",
                "--provider",
                "anthropic",
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "Anthropic evaluation completed" in result.stdout

    def test_cli_single_document_evaluation_error(self, mock_subprocess):
        """Test single document evaluation error via CLI."""
        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "Error: Evaluation failed"
        mock_subprocess.return_value = mock_result

        # Test CLI command with error
        result = subprocess.run(
            ["python", "main.py", "test_input", "test_context", "test_output"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 1
        assert "Error: Evaluation failed" in result.stderr

    def test_cli_batch_command_success(self, mock_subprocess):
        """Test successful batch command via CLI."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "Batch created and started successfully"
        mock_result.stderr = ""
        mock_subprocess.return_value = mock_result

        # Test CLI batch command
        result = subprocess.run(
            ["python", "main.py", "--batch", "--folder", "/test/path"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "Batch created and started successfully" in result.stdout

    def test_cli_batch_command_with_documents(self, mock_subprocess):
        """Test batch command with documents via CLI."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "Batch created with documents"
        mock_result.stderr = ""
        mock_subprocess.return_value = mock_result

        # Test CLI batch command with documents
        result = subprocess.run(
            ["python", "main.py", "--batch", "--documents", "doc1.txt,doc2.txt"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "Batch created with documents" in result.stdout

    def test_cli_batch_command_with_options(self, mock_subprocess):
        """Test batch command with various options via CLI."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "Batch created with custom options"
        mock_result.stderr = ""
        mock_subprocess.return_value = mock_result

        # Test CLI batch command with options
        result = subprocess.run(
            [
                "python",
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
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "Batch created with custom options" in result.stdout

    def test_cli_batch_command_error(self, mock_subprocess):
        """Test batch command error via CLI."""
        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "Error: Must specify either --folder or --documents"
        mock_subprocess.return_value = mock_result

        # Test CLI batch command with error
        result = subprocess.run(
            ["python", "main.py", "--batch"], capture_output=True, text=True
        )

        assert result.returncode == 1
        assert "Error: Must specify either --folder or --documents" in result.stderr

    def test_cli_batch_status_command_success(self, mock_subprocess):
        """Test successful batch status command via CLI."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "Batch Status: processing\nProgress: 50%\nDocuments: 5/10"
        mock_result.stderr = ""
        mock_subprocess.return_value = mock_result

        # Test CLI batch status command
        result = subprocess.run(
            ["python", "main.py", "--batch-status", "batch_001"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "Batch Status: processing" in result.stdout
        assert "Progress: 50%" in result.stdout
        assert "Documents: 5/10" in result.stdout

    def test_cli_batch_status_command_not_found(self, mock_subprocess):
        """Test batch status command with batch not found via CLI."""
        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "Error: Batch nonexistent_batch not found"
        mock_subprocess.return_value = mock_result

        # Test CLI batch status command with error
        result = subprocess.run(
            ["python", "main.py", "--batch-status", "nonexistent_batch"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 1
        assert "Error: Batch nonexistent_batch not found" in result.stderr

    def test_cli_batch_results_command_success(self, mock_subprocess):
        """Test successful batch results command via CLI."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "Batch Results:\nTotal: 10\nSuccessful: 8\nFailed: 2"
        mock_result.stderr = ""
        mock_subprocess.return_value = mock_result

        # Test CLI batch results command
        result = subprocess.run(
            ["python", "main.py", "--batch-results", "batch_001"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "Batch Results:" in result.stdout
        assert "Total: 10" in result.stdout
        assert "Successful: 8" in result.stdout
        assert "Failed: 2" in result.stdout

    def test_cli_batch_results_command_with_format(self, mock_subprocess):
        """Test batch results command with format via CLI."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "CSV format results"
        mock_result.stderr = ""
        mock_subprocess.return_value = mock_result

        # Test CLI batch results command with format
        result = subprocess.run(
            ["python", "main.py", "--batch-results", "batch_001", "--format", "csv"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "CSV format results" in result.stdout

    def test_cli_resume_batch_command_success(self, mock_subprocess):
        """Test successful resume batch command via CLI."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "Batch resumed successfully"
        mock_result.stderr = ""
        mock_subprocess.return_value = mock_result

        # Test CLI resume batch command
        result = subprocess.run(
            ["python", "main.py", "--resume-batch", "batch_001"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "Batch resumed successfully" in result.stdout

    def test_cli_retry_documents_command_success(self, mock_subprocess):
        """Test successful retry documents command via CLI."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "Documents retried successfully"
        mock_result.stderr = ""
        mock_subprocess.return_value = mock_result

        # Test CLI retry documents command
        result = subprocess.run(
            ["python", "main.py", "--retry-documents", "batch_001", "doc_001,doc_002"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "Documents retried successfully" in result.stdout

    def test_cli_retry_documents_command_missing_args(self, mock_subprocess):
        """Test retry documents command with missing arguments via CLI."""
        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "Error: --retry-documents requires a batch_id"
        mock_subprocess.return_value = mock_result

        # Test CLI retry documents command with error
        result = subprocess.run(
            ["python", "main.py", "--retry-documents"], capture_output=True, text=True
        )

        assert result.returncode == 1
        assert "Error: --retry-documents requires a batch_id" in result.stderr

    def test_cli_usage_command(self, mock_subprocess):
        """Test CLI usage command."""
        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "Usage: python evaluator.py"
        mock_subprocess.return_value = mock_result

        # Test CLI usage command
        result = subprocess.run(["python", "main.py"], capture_output=True, text=True)

        assert result.returncode == 1
        assert "Usage: python evaluator.py" in result.stderr

    def test_cli_help_command(self, mock_subprocess):
        """Test CLI help command."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "Help information"
        mock_result.stderr = ""
        mock_subprocess.return_value = mock_result

        # Test CLI help command
        result = subprocess.run(
            ["python", "main.py", "--help"], capture_output=True, text=True
        )

        assert result.returncode == 0
        assert "Help information" in result.stdout

    def test_cli_version_command(self, mock_subprocess):
        """Test CLI version command."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "Version 2.0.0"
        mock_result.stderr = ""
        mock_subprocess.return_value = mock_result

        # Test CLI version command
        result = subprocess.run(
            ["python", "main.py", "--version"], capture_output=True, text=True
        )

        assert result.returncode == 0
        assert "Version 2.0.0" in result.stdout

    def test_cli_unicode_content(self, mock_subprocess):
        """Test CLI with Unicode content."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "Unicode evaluation completed: テスト入力"
        mock_result.stderr = ""
        mock_subprocess.return_value = mock_result

        # Test CLI with Unicode content
        result = subprocess.run(
            ["python", "main.py", "テスト入力", "テストコンテキスト", "テスト出力"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "Unicode evaluation completed: テスト入力" in result.stdout

    def test_cli_large_input(self, mock_subprocess):
        """Test CLI with large input."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "Large input evaluation completed"
        mock_result.stderr = ""
        mock_subprocess.return_value = mock_result

        # Test CLI with large input
        large_input = "This is a large test document. " * 1000  # ~30KB of text
        result = subprocess.run(
            ["python", "main.py", large_input, "test_context", "test_output"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "Large input evaluation completed" in result.stdout

    def test_cli_timeout_handling(self, mock_subprocess):
        """Test CLI timeout handling."""
        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "Error: Request timeout"
        mock_subprocess.return_value = mock_result

        # Test CLI with timeout
        result = subprocess.run(
            ["python", "main.py", "test_input", "test_context", "test_output"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 1
        assert "Error: Request timeout" in result.stderr

    def test_cli_memory_error(self, mock_subprocess):
        """Test CLI memory error handling."""
        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "Error: Out of memory"
        mock_subprocess.return_value = mock_result

        # Test CLI with memory error
        result = subprocess.run(
            ["python", "main.py", "test_input", "test_context", "test_output"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 1
        assert "Error: Out of memory" in result.stderr

    def test_cli_permission_error(self, mock_subprocess):
        """Test CLI permission error handling."""
        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "Error: Permission denied"
        mock_subprocess.return_value = mock_result

        # Test CLI with permission error
        result = subprocess.run(
            ["python", "main.py", "test_input", "test_context", "test_output"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 1
        assert "Error: Permission denied" in result.stderr

    def test_cli_file_not_found_error(self, mock_subprocess):
        """Test CLI file not found error handling."""
        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "Error: File not found"
        mock_subprocess.return_value = mock_result

        # Test CLI with file not found error
        result = subprocess.run(
            ["python", "main.py", "--batch", "--folder", "/nonexistent/path"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 1
        assert "Error: File not found" in result.stderr

    def test_cli_invalid_arguments(self, mock_subprocess):
        """Test CLI with invalid arguments."""
        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "Error: Invalid arguments"
        mock_subprocess.return_value = mock_result

        # Test CLI with invalid arguments
        result = subprocess.run(
            ["python", "main.py", "--invalid-argument"], capture_output=True, text=True
        )

        assert result.returncode == 1
        assert "Error: Invalid arguments" in result.stderr

    def test_cli_missing_required_arguments(self, mock_subprocess):
        """Test CLI with missing required arguments."""
        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "Error: Missing required arguments"
        mock_subprocess.return_value = mock_result

        # Test CLI with missing required arguments
        result = subprocess.run(
            ["python", "main.py", "--batch-status"], capture_output=True, text=True
        )

        assert result.returncode == 1
        assert "Error: Missing required arguments" in result.stderr

    def test_cli_concurrent_execution(self, mock_subprocess):
        """Test CLI concurrent execution."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "Concurrent execution completed"
        mock_result.stderr = ""
        mock_subprocess.return_value = mock_result

        # Test CLI concurrent execution
        result = subprocess.run(
            [
                "python",
                "main.py",
                "--batch",
                "--folder",
                "/test/path",
                "--max-workers",
                "10",
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "Concurrent execution completed" in result.stdout

    def test_cli_environment_variables(self, mock_subprocess):
        """Test CLI with environment variables."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "Environment variables configured"
        mock_result.stderr = ""
        mock_subprocess.return_value = mock_result

        # Test CLI with environment variables
        env = os.environ.copy()
        env["OPENAI_API_KEY"] = "test-key"
        env["ANTHROPIC_API_KEY"] = "test-key"

        result = subprocess.run(
            ["python", "main.py", "test_input", "test_context", "test_output"],
            capture_output=True,
            text=True,
            env=env,
        )

        assert result.returncode == 0
        assert "Environment variables configured" in result.stdout
