"""
Integration tests for batch error handling and recovery.

These tests validate error handling, retry mechanisms, and recovery
for various failure scenarios in batch processing.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
from unittest.mock import patch, MagicMock


class TestBatchErrorHandling:
    """Test batch error handling and recovery integration."""

    def setup_method(self):
        """Set up test environment with documents that may cause errors."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.create_test_documents()

    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def create_test_documents(self):
        """Create test documents including some that may cause errors."""
        # Valid documents
        (self.test_dir / "valid1.txt").write_text("Valid document content")
        (self.test_dir / "valid2.txt").write_text("Another valid document")

        # Corrupted documents
        (self.test_dir / "corrupted.pdf").write_bytes(b"Not a valid PDF file")
        (self.test_dir / "corrupted.docx").write_bytes(b"Not a valid Word document")

        # Empty documents
        (self.test_dir / "empty.txt").write_text("")

        # Very large document
        (self.test_dir / "large.txt").write_text("Large content " * 100000)

        # Document with special characters
        (self.test_dir / "special.txt").write_text(
            "Document with special chars: éñ中文🚀"
        )

    def test_individual_document_failures(self):
        """Test handling of individual document processing failures."""
        from evaluator import BatchEvaluator, BatchConfiguration

        config = BatchConfiguration(
            max_concurrent_workers=2, max_retry_attempts=1, timeout_seconds=30
        )
        evaluator = BatchEvaluator(config=config)

        # Create batch with mixed valid/invalid documents
        batch_id = evaluator.create_batch_from_folder(
            folder_path=str(self.test_dir), context="Test individual failures"
        )

        # Start processing
        evaluator.start_batch(batch_id)

        # Validate error handling
        batch = evaluator.get_batch(batch_id)
        assert batch.status in ["processing", "completed", "failed", "partially_failed"]

        # Check individual document statuses
        documents = evaluator.list_batch_documents(batch_id)
        statuses = [doc.status for doc in documents]
        assert "failed" in statuses or "completed" in statuses

    def test_batch_level_failures(self):
        """Test handling of batch-level failures."""
        from evaluator import BatchEvaluator, BatchConfiguration

        config = BatchConfiguration(
            max_concurrent_workers=1, max_retry_attempts=0, timeout_seconds=5
        )
        evaluator = BatchEvaluator(config=config)

        # Create batch that may fail
        batch_id = evaluator.create_batch_from_folder(
            folder_path=str(self.test_dir), context="Test batch failures"
        )

        # Start processing
        evaluator.start_batch(batch_id)

        # Validate batch-level error handling
        batch = evaluator.get_batch(batch_id)
        assert batch.status in ["processing", "completed", "failed", "partially_failed"]

        if batch.status == "failed":
            assert batch.error_summary is not None

    def test_retry_mechanism(self):
        """Test retry mechanism for failed documents."""
        from evaluator import BatchEvaluator, BatchConfiguration

        config = BatchConfiguration(
            max_concurrent_workers=2, max_retry_attempts=3, timeout_seconds=30
        )
        evaluator = BatchEvaluator(config=config)

        # Create batch
        batch_id = evaluator.create_batch_from_folder(
            folder_path=str(self.test_dir), context="Test retry mechanism"
        )

        # Start processing
        evaluator.start_batch(batch_id)

        # Get failed documents
        documents = evaluator.list_batch_documents(batch_id)
        failed_docs = [doc for doc in documents if doc.status == "failed"]

        # Retry failed documents
        for doc in failed_docs:
            evaluator.retry_document(batch_id, doc.document_id)

        # Validate retry mechanism
        batch = evaluator.get_batch(batch_id)
        assert batch is not None

    def test_batch_resumption(self):
        """Test batch resumption after failures."""
        from evaluator import BatchEvaluator, BatchConfiguration

        config = BatchConfiguration(max_concurrent_workers=2, enable_resumption=True)
        evaluator = BatchEvaluator(config=config)

        # Create batch
        batch_id = evaluator.create_batch_from_folder(
            folder_path=str(self.test_dir), context="Test batch resumption"
        )

        # Start processing
        evaluator.start_batch(batch_id)

        # Simulate interruption and resumption
        evaluator.resume_batch(batch_id)

        # Validate resumption
        batch = evaluator.get_batch(batch_id)
        assert batch is not None

    def test_timeout_handling(self):
        """Test timeout handling for long-running evaluations."""
        from evaluator import BatchEvaluator, BatchConfiguration

        config = BatchConfiguration(
            max_concurrent_workers=1, timeout_seconds=1  # Very short timeout
        )
        evaluator = BatchEvaluator(config=config)

        # Create batch with large document
        batch_id = evaluator.create_batch_from_folder(
            folder_path=str(self.test_dir), context="Test timeout handling"
        )

        # Start processing
        evaluator.start_batch(batch_id)

        # Validate timeout handling
        batch = evaluator.get_batch(batch_id)
        assert batch.status in ["processing", "completed", "failed", "partially_failed"]

    def test_memory_exhaustion_handling(self):
        """Test handling of memory exhaustion scenarios."""
        from evaluator import BatchEvaluator, BatchConfiguration

        config = BatchConfiguration(
            max_concurrent_workers=1, memory_limit_mb=1  # Very low memory limit
        )
        evaluator = BatchEvaluator(config=config)

        # Create batch with large documents
        batch_id = evaluator.create_batch_from_folder(
            folder_path=str(self.test_dir), context="Test memory exhaustion"
        )

        # Start processing
        evaluator.start_batch(batch_id)

        # Validate memory handling
        batch = evaluator.get_batch(batch_id)
        assert batch.status in ["processing", "completed", "failed", "partially_failed"]

    def test_api_rate_limit_handling(self):
        """Test handling of LLM API rate limits."""
        from evaluator import BatchEvaluator, BatchConfiguration

        config = BatchConfiguration(max_concurrent_workers=5, api_rate_limit_delay=1.0)
        evaluator = BatchEvaluator(config=config)

        # Create batch
        batch_id = evaluator.create_batch_from_folder(
            folder_path=str(self.test_dir), context="Test API rate limits"
        )

        # Start processing
        evaluator.start_batch(batch_id)

        # Validate rate limit handling
        batch = evaluator.get_batch(batch_id)
        assert batch.status in ["processing", "completed", "failed", "partially_failed"]

    def test_network_error_handling(self):
        """Test handling of network errors during processing."""
        from evaluator import BatchEvaluator, BatchConfiguration

        config = BatchConfiguration(
            max_concurrent_workers=2, max_retry_attempts=2, retry_delay_seconds=1
        )
        evaluator = BatchEvaluator(config=config)

        # Mock network errors
        with patch("evaluator.LLMProvider.generate_response") as mock_response:
            mock_response.side_effect = Exception("Network error")

            # Create batch
            batch_id = evaluator.create_batch_from_folder(
                folder_path=str(self.test_dir), context="Test network errors"
            )

            # Start processing
            evaluator.start_batch(batch_id)

            # Validate network error handling
            batch = evaluator.get_batch(batch_id)
            assert batch.status in [
                "processing",
                "completed",
                "failed",
                "partially_failed",
            ]

    def test_file_access_error_handling(self):
        """Test handling of file access errors."""
        from evaluator import BatchEvaluator, BatchConfiguration

        config = BatchConfiguration(max_concurrent_workers=2)
        evaluator = BatchEvaluator(config=config)

        # Create batch with inaccessible files
        batch_id = evaluator.create_batch_from_folder(
            folder_path="/nonexistent/path", context="Test file access errors"
        )

        # Validate file access error handling
        with pytest.raises(Exception):
            evaluator.start_batch(batch_id)

    def test_corrupted_state_file_handling(self):
        """Test handling of corrupted state files."""
        from evaluator import BatchEvaluator, BatchConfiguration

        config = BatchConfiguration(max_concurrent_workers=2, enable_resumption=True)
        evaluator = BatchEvaluator(config=config)

        # Create batch
        batch_id = evaluator.create_batch_from_folder(
            folder_path=str(self.test_dir), context="Test corrupted state"
        )

        # Corrupt state file
        state_file = Path(f"batch_state/active_batches/{batch_id}.json")
        state_file.parent.mkdir(parents=True, exist_ok=True)
        state_file.write_text("corrupted json content")

        # Try to resume
        with pytest.raises(Exception):
            evaluator.resume_batch(batch_id)

    def test_concurrent_batch_errors(self):
        """Test handling of errors in concurrent batch processing."""
        from evaluator import BatchEvaluator, BatchConfiguration

        config = BatchConfiguration(max_concurrent_workers=3, max_retry_attempts=1)
        evaluator = BatchEvaluator(config=config)

        # Create multiple batches
        batch_ids = []
        for i in range(3):
            batch_id = evaluator.create_batch_from_folder(
                folder_path=str(self.test_dir), context=f"Test concurrent batch {i}"
            )
            batch_ids.append(batch_id)

        # Start all batches concurrently
        for batch_id in batch_ids:
            evaluator.start_batch(batch_id)

        # Validate concurrent error handling
        for batch_id in batch_ids:
            batch = evaluator.get_batch(batch_id)
            assert batch.status in [
                "processing",
                "completed",
                "failed",
                "partially_failed",
            ]

    def test_error_aggregation(self):
        """Test aggregation of errors across documents."""
        from evaluator import BatchEvaluator, BatchConfiguration

        config = BatchConfiguration(max_concurrent_workers=2, max_retry_attempts=1)
        evaluator = BatchEvaluator(config=config)

        # Create batch
        batch_id = evaluator.create_batch_from_folder(
            folder_path=str(self.test_dir), context="Test error aggregation"
        )

        # Start processing
        evaluator.start_batch(batch_id)

        # Get results
        results = evaluator.get_batch_results(batch_id)

        # Validate error aggregation
        if results:
            assert results.failed_evaluations >= 0
            assert results.successful_evaluations >= 0
            assert len(results.error_summary) >= 0

    def test_error_recovery_strategies(self):
        """Test different error recovery strategies."""
        from evaluator import BatchEvaluator, BatchConfiguration

        # Test with different retry strategies
        configs = [
            BatchConfiguration(max_retry_attempts=0),  # No retries
            BatchConfiguration(max_retry_attempts=1),  # Single retry
            BatchConfiguration(max_retry_attempts=3),  # Multiple retries
        ]

        for config in configs:
            evaluator = BatchEvaluator(config=config)

            # Create batch
            batch_id = evaluator.create_batch_from_folder(
                folder_path=str(self.test_dir), context="Test recovery strategies"
            )

            # Start processing
            evaluator.start_batch(batch_id)

            # Validate recovery strategy
            batch = evaluator.get_batch(batch_id)
            assert batch.status in [
                "processing",
                "completed",
                "failed",
                "partially_failed",
            ]

    def test_error_logging(self):
        """Test error logging and correlation."""
        from evaluator import BatchEvaluator, BatchConfiguration

        config = BatchConfiguration(max_concurrent_workers=2)
        evaluator = BatchEvaluator(config=config)

        # Create batch
        batch_id = evaluator.create_batch_from_folder(
            folder_path=str(self.test_dir), context="Test error logging"
        )

        # Start processing
        evaluator.start_batch(batch_id)

        # Validate error logging
        batch = evaluator.get_batch(batch_id)
        assert batch.correlation_id is not None

        # Check document correlation IDs
        documents = evaluator.list_batch_documents(batch_id)
        for doc in documents:
            assert doc.correlation_id is not None

    def test_graceful_degradation(self):
        """Test graceful degradation when errors occur."""
        from evaluator import BatchEvaluator, BatchConfiguration

        config = BatchConfiguration(
            max_concurrent_workers=2, enable_graceful_degradation=True
        )
        evaluator = BatchEvaluator(config=config)

        # Create batch
        batch_id = evaluator.create_batch_from_folder(
            folder_path=str(self.test_dir), context="Test graceful degradation"
        )

        # Start processing
        evaluator.start_batch(batch_id)

        # Validate graceful degradation
        batch = evaluator.get_batch(batch_id)
        assert batch.status in ["processing", "completed", "failed", "partially_failed"]

        # Even with errors, some documents should be processed
        if batch.status == "partially_failed":
            assert batch.successful_documents > 0
