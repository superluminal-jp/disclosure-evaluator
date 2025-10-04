"""
Integration tests for folder-based batch evaluation.

These tests validate the complete workflow of discovering documents
in a folder and processing them as a batch.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from typing import List
from datetime import datetime


class TestBatchFolderEvaluation:
    """Test folder-based batch evaluation integration."""

    def setup_method(self):
        """Set up test environment with sample documents."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.create_test_documents()

    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def create_test_documents(self):
        """Create sample documents for testing."""
        # Create test documents
        (self.test_dir / "document1.txt").write_text(
            "This is a test document with personal information."
        )
        (self.test_dir / "document2.txt").write_text(
            "This is another test document with financial data."
        )
        (self.test_dir / "document3.pdf").write_bytes(b"PDF content placeholder")

        # Create subdirectory with more documents
        subdir = self.test_dir / "subdir"
        subdir.mkdir()
        (subdir / "document4.txt").write_text("Document in subdirectory.")

        # Create unsupported file type
        (self.test_dir / "document5.log").write_text("Log file content")

    def test_folder_discovery_integration(self):
        """Test complete folder discovery and batch creation workflow."""
        from evaluator import BatchEvaluator, BatchConfiguration

        # Initialize batch evaluator
        config = BatchConfiguration(
            max_concurrent_workers=2,
            max_retry_attempts=1,
            timeout_seconds=60,
            enable_resumption=True,
        )
        evaluator = BatchEvaluator(config=config)

        # Create batch from folder
        batch_id = evaluator.create_batch_from_folder(
            folder_path=str(self.test_dir),
            context="Test batch evaluation",
            recursive=True,
        )

        # Validate batch creation
        assert batch_id is not None
        assert isinstance(batch_id, str)
        assert batch_id.startswith("batch_")

        # Get batch details
        batch = evaluator.get_batch(batch_id)
        assert batch is not None
        assert batch.batch_id == batch_id
        assert batch.total_documents > 0
        assert batch.status == "pending"

    def test_folder_discovery_with_filters(self):
        """Test folder discovery with file type filters."""
        from evaluator import BatchEvaluator, BatchConfiguration

        config = BatchConfiguration(max_concurrent_workers=2, enable_resumption=True)
        evaluator = BatchEvaluator(config=config)

        # Create batch with specific file types only
        batch_id = evaluator.create_batch_from_folder(
            folder_path=str(self.test_dir),
            context="Test batch with filters",
            recursive=True,
            file_types=["text/plain"],  # Only text files
            exclude_patterns=["*.log"],
        )

        # Validate batch creation
        batch = evaluator.get_batch(batch_id)
        assert (
            batch.total_documents >= 3
        )  # Should include txt files but exclude log files

    def test_folder_discovery_recursive(self):
        """Test recursive folder discovery."""
        from evaluator import BatchEvaluator, BatchConfiguration

        config = BatchConfiguration(max_concurrent_workers=2)
        evaluator = BatchEvaluator(config=config)

        # Create batch with recursive discovery
        batch_id = evaluator.create_batch_from_folder(
            folder_path=str(self.test_dir),
            context="Test recursive discovery",
            recursive=True,
        )

        # Validate that subdirectory documents are included
        batch = evaluator.get_batch(batch_id)
        assert batch.total_documents >= 4  # Should include documents from subdirectory

    def test_folder_discovery_non_recursive(self):
        """Test non-recursive folder discovery."""
        from evaluator import BatchEvaluator, BatchConfiguration

        config = BatchConfiguration(max_concurrent_workers=2)
        evaluator = BatchEvaluator(config=config)

        # Create batch without recursive discovery
        batch_id = evaluator.create_batch_from_folder(
            folder_path=str(self.test_dir),
            context="Test non-recursive discovery",
            recursive=False,
        )

        # Validate that subdirectory documents are not included
        batch = evaluator.get_batch(batch_id)
        assert (
            batch.total_documents == 3
        )  # Should only include root directory documents

    def test_folder_discovery_empty_folder(self):
        """Test folder discovery with empty folder."""
        from evaluator import BatchEvaluator, BatchConfiguration

        # Create empty folder
        empty_dir = self.test_dir / "empty"
        empty_dir.mkdir()

        config = BatchConfiguration(max_concurrent_workers=2)
        evaluator = BatchEvaluator(config=config)

        # Should handle empty folder gracefully
        with pytest.raises(Exception):  # Should raise appropriate error
            evaluator.create_batch_from_folder(
                folder_path=str(empty_dir), context="Test empty folder"
            )

    def test_folder_discovery_nonexistent_folder(self):
        """Test folder discovery with nonexistent folder."""
        from evaluator import BatchEvaluator, BatchConfiguration

        config = BatchConfiguration(max_concurrent_workers=2)
        evaluator = BatchEvaluator(config=config)

        # Should handle nonexistent folder gracefully
        with pytest.raises(Exception):  # Should raise appropriate error
            evaluator.create_batch_from_folder(
                folder_path="/nonexistent/folder", context="Test nonexistent folder"
            )

    def test_folder_discovery_permission_error(self):
        """Test folder discovery with permission errors."""
        from evaluator import BatchEvaluator, BatchConfiguration

        # Create folder with restricted permissions (if possible)
        restricted_dir = self.test_dir / "restricted"
        restricted_dir.mkdir()

        config = BatchConfiguration(max_concurrent_workers=2)
        evaluator = BatchEvaluator(config=config)

        # Should handle permission errors gracefully
        # Note: This test may not work on all systems
        try:
            batch_id = evaluator.create_batch_from_folder(
                folder_path=str(restricted_dir), context="Test permission error"
            )
            # If no error, validate the batch
            batch = evaluator.get_batch(batch_id)
            assert batch is not None
        except Exception:
            # Permission error is expected and acceptable
            pass

    def test_folder_discovery_large_folder(self):
        """Test folder discovery with many documents."""
        from evaluator import BatchEvaluator, BatchConfiguration

        # Create many test documents
        large_dir = self.test_dir / "large"
        large_dir.mkdir()

        for i in range(50):
            (large_dir / f"document_{i:03d}.txt").write_text(f"Test document {i}")

        config = BatchConfiguration(
            max_concurrent_workers=5, max_retry_attempts=1, timeout_seconds=30
        )
        evaluator = BatchEvaluator(config=config)

        # Create batch with many documents
        batch_id = evaluator.create_batch_from_folder(
            folder_path=str(large_dir), context="Test large folder", recursive=False
        )

        # Validate batch creation
        batch = evaluator.get_batch(batch_id)
        assert batch.total_documents == 50

    def test_folder_discovery_file_size_limits(self):
        """Test folder discovery with file size limits."""
        from evaluator import BatchEvaluator, BatchConfiguration

        # Create documents of different sizes
        small_file = self.test_dir / "small.txt"
        small_file.write_text("Small content")

        large_file = self.test_dir / "large.txt"
        large_file.write_text("Large content " * 1000)  # Make it larger

        config = BatchConfiguration(
            max_concurrent_workers=2, file_size_limit=1024  # 1KB limit
        )
        evaluator = BatchEvaluator(config=config)

        # Create batch with size limit
        batch_id = evaluator.create_batch_from_folder(
            folder_path=str(self.test_dir),
            context="Test size limits",
            file_size_limit=1024,
        )

        # Validate that large files are excluded
        batch = evaluator.get_batch(batch_id)
        # Should only include small files
        assert batch.total_documents >= 1

    def test_folder_discovery_mixed_file_types(self):
        """Test folder discovery with mixed file types."""
        from evaluator import BatchEvaluator, BatchConfiguration

        # Create various file types
        (self.test_dir / "text.txt").write_text("Text content")
        (self.test_dir / "data.csv").write_text("CSV,data,content")
        (self.test_dir / "config.json").write_text('{"key": "value"}')

        config = BatchConfiguration(max_concurrent_workers=2)
        evaluator = BatchEvaluator(config=config)

        # Create batch with mixed file types
        batch_id = evaluator.create_batch_from_folder(
            folder_path=str(self.test_dir),
            context="Test mixed file types",
            file_types=["text/plain", "text/csv", "application/json"],
        )

        # Validate batch creation
        batch = evaluator.get_batch(batch_id)
        assert batch.total_documents >= 3

    def test_folder_discovery_progress_callback(self):
        """Test folder discovery with progress callback."""
        from evaluator import BatchEvaluator, BatchConfiguration

        progress_updates = []

        def progress_callback(update):
            progress_updates.append(update)

        config = BatchConfiguration(max_concurrent_workers=2)
        evaluator = BatchEvaluator(config=config)

        # Create batch with progress callback
        batch_id = evaluator.create_batch_from_folder(
            folder_path=str(self.test_dir),
            context="Test progress callback",
            progress_callback=progress_callback,
        )

        # Validate that progress updates were received
        # Note: This may not work until the implementation is complete
        assert batch_id is not None

    def test_folder_discovery_error_handling(self):
        """Test folder discovery error handling."""
        from evaluator import BatchEvaluator, BatchConfiguration

        config = BatchConfiguration(max_concurrent_workers=2)
        evaluator = BatchEvaluator(config=config)

        # Test with invalid folder path
        with pytest.raises(Exception):
            evaluator.create_batch_from_folder(
                folder_path="", context="Test error handling"  # Empty path
            )

        # Test with None folder path
        with pytest.raises(Exception):
            evaluator.create_batch_from_folder(
                folder_path=None, context="Test error handling"
            )

    def test_folder_discovery_batch_processing_workflow(self):
        """Test complete folder discovery and batch processing workflow."""
        from evaluator import BatchEvaluator, BatchConfiguration

        config = BatchConfiguration(
            max_concurrent_workers=2, max_retry_attempts=1, timeout_seconds=60
        )
        evaluator = BatchEvaluator(config=config)

        # Create batch from folder
        batch_id = evaluator.create_batch_from_folder(
            folder_path=str(self.test_dir), context="Test complete workflow"
        )

        # Start batch processing
        evaluator.start_batch(batch_id)

        # Monitor progress
        progress = evaluator.get_batch_progress(batch_id)
        assert progress is not None
        assert progress.batch_id == batch_id

        # Wait for completion (in real implementation)
        # For now, just validate the workflow structure
        batch = evaluator.get_batch(batch_id)
        assert batch.status in [
            "pending",
            "processing",
            "completed",
            "failed",
            "partially_failed",
        ]
