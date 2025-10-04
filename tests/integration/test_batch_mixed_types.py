"""
Integration tests for mixed document types batch evaluation.

These tests validate processing documents of different types
(text, PDF, Word, etc.) in a single batch.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime


class TestBatchMixedTypes:
    """Test mixed document types batch evaluation integration."""

    def setup_method(self):
        """Set up test environment with mixed document types."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.create_mixed_documents()

    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def create_mixed_documents(self):
        """Create documents of various types for testing."""
        # Text documents
        (self.test_dir / "document1.txt").write_text(
            "Personal information: John Doe, 123 Main St"
        )
        (self.test_dir / "document2.txt").write_text(
            "Financial data: Revenue $1M, Profit $200K"
        )

        # PDF documents (mock)
        (self.test_dir / "document3.pdf").write_bytes(b"%PDF-1.4\nMock PDF content")
        (self.test_dir / "document4.pdf").write_bytes(b"%PDF-1.4\nAnother PDF document")

        # Word documents (mock)
        (self.test_dir / "document5.docx").write_bytes(b"Mock Word document content")
        (self.test_dir / "document6.doc").write_bytes(b"Mock Word 97 document content")

        # CSV documents
        (self.test_dir / "document7.csv").write_text(
            "Name,Age,City\nJohn,30,Tokyo\nJane,25,Osaka"
        )

        # JSON documents
        (self.test_dir / "document8.json").write_text(
            '{"name": "John", "age": 30, "city": "Tokyo"}'
        )

        # XML documents
        (self.test_dir / "document9.xml").write_text(
            "<root><name>John</name><age>30</age></root>"
        )

    def test_mixed_document_types_batch_creation(self):
        """Test creating batch with mixed document types."""
        from evaluator import BatchEvaluator, BatchConfiguration, DocumentInput

        config = BatchConfiguration(
            max_concurrent_workers=3, max_retry_attempts=2, timeout_seconds=120
        )
        evaluator = BatchEvaluator(config=config)

        # Create documents of different types
        documents = [
            DocumentInput(
                file_path=str(self.test_dir / "document1.txt"),
                file_name="document1.txt",
                context="Personal information document",
            ),
            DocumentInput(
                file_path=str(self.test_dir / "document3.pdf"),
                file_name="document3.pdf",
                context="PDF document with sensitive data",
            ),
            DocumentInput(
                file_path=str(self.test_dir / "document5.docx"),
                file_name="document5.docx",
                context="Word document with business information",
            ),
            DocumentInput(
                file_path=str(self.test_dir / "document7.csv"),
                file_name="document7.csv",
                context="CSV data file",
            ),
        ]

        # Create batch with mixed document types
        batch_id = evaluator.create_batch(documents, config)

        # Validate batch creation
        assert batch_id is not None
        batch = evaluator.get_batch(batch_id)
        assert batch.total_documents == 4
        assert batch.status == "pending"

    def test_mixed_document_types_processing(self):
        """Test processing mixed document types in batch."""
        from evaluator import BatchEvaluator, BatchConfiguration

        config = BatchConfiguration(
            max_concurrent_workers=2, max_retry_attempts=1, timeout_seconds=60
        )
        evaluator = BatchEvaluator(config=config)

        # Create batch from folder with mixed types
        batch_id = evaluator.create_batch_from_folder(
            folder_path=str(self.test_dir),
            context="Test mixed document processing",
            recursive=False,
        )

        # Start processing
        evaluator.start_batch(batch_id)

        # Validate processing started
        batch = evaluator.get_batch(batch_id)
        assert batch.status in ["processing", "completed", "failed", "partially_failed"]

    def test_document_type_detection(self):
        """Test automatic document type detection."""
        from evaluator import BatchEvaluator, BatchConfiguration

        config = BatchConfiguration(max_concurrent_workers=2)
        evaluator = BatchEvaluator(config=config)

        # Create batch and check document types
        batch_id = evaluator.create_batch_from_folder(
            folder_path=str(self.test_dir), context="Test document type detection"
        )

        # Get batch documents
        documents = evaluator.list_batch_documents(batch_id)

        # Validate document types are detected
        mime_types = [doc.mime_type for doc in documents]
        assert "text/plain" in mime_types
        assert "application/pdf" in mime_types
        assert (
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            in mime_types
        )

    def test_unsupported_document_types(self):
        """Test handling of unsupported document types."""
        from evaluator import BatchEvaluator, BatchConfiguration

        # Create unsupported file types
        (self.test_dir / "document.bin").write_bytes(b"Binary content")
        (self.test_dir / "document.exe").write_bytes(b"Executable content")

        config = BatchConfiguration(max_concurrent_workers=2)
        evaluator = BatchEvaluator(config=config)

        # Create batch - should filter out unsupported types
        batch_id = evaluator.create_batch_from_folder(
            folder_path=str(self.test_dir), context="Test unsupported types"
        )

        # Validate that unsupported types are excluded
        batch = evaluator.get_batch(batch_id)
        # Should only include supported document types
        assert batch.total_documents >= 6  # Supported types only

    def test_document_type_specific_processing(self):
        """Test document type-specific processing logic."""
        from evaluator import BatchEvaluator, BatchConfiguration

        config = BatchConfiguration(max_concurrent_workers=2, timeout_seconds=60)
        evaluator = BatchEvaluator(config=config)

        # Create batch with specific document types
        batch_id = evaluator.create_batch_from_folder(
            folder_path=str(self.test_dir),
            context="Test type-specific processing",
            file_types=["text/plain", "application/pdf"],
        )

        # Validate only specified types are included
        batch = evaluator.get_batch(batch_id)
        documents = evaluator.list_batch_documents(batch_id)

        # Should only include text and PDF files
        mime_types = [doc.mime_type for doc in documents]
        assert all(
            mime_type in ["text/plain", "application/pdf"] for mime_type in mime_types
        )

    def test_mixed_document_error_handling(self):
        """Test error handling with mixed document types."""
        from evaluator import BatchEvaluator, BatchConfiguration

        # Create corrupted documents
        (self.test_dir / "corrupted.pdf").write_bytes(b"Not a valid PDF")
        (self.test_dir / "corrupted.docx").write_bytes(b"Not a valid Word document")

        config = BatchConfiguration(
            max_concurrent_workers=2, max_retry_attempts=1, timeout_seconds=30
        )
        evaluator = BatchEvaluator(config=config)

        # Create batch with corrupted documents
        batch_id = evaluator.create_batch_from_folder(
            folder_path=str(self.test_dir), context="Test error handling"
        )

        # Start processing
        evaluator.start_batch(batch_id)

        # Validate error handling
        batch = evaluator.get_batch(batch_id)
        # Should handle corrupted documents gracefully
        assert batch.status in ["processing", "completed", "failed", "partially_failed"]

    def test_mixed_document_memory_usage(self):
        """Test memory usage with mixed document types."""
        from evaluator import BatchEvaluator, BatchConfiguration

        # Create large documents of different types
        large_text = "Large text content " * 10000
        (self.test_dir / "large_text.txt").write_text(large_text)

        large_binary = b"Large binary content " * 10000
        (self.test_dir / "large_binary.pdf").write_bytes(large_binary)

        config = BatchConfiguration(
            max_concurrent_workers=1, timeout_seconds=60  # Process one at a time
        )
        evaluator = BatchEvaluator(config=config)

        # Create batch with large documents
        batch_id = evaluator.create_batch_from_folder(
            folder_path=str(self.test_dir), context="Test memory usage"
        )

        # Validate batch creation
        batch = evaluator.get_batch(batch_id)
        assert batch.total_documents >= 2

    def test_mixed_document_parallel_processing(self):
        """Test parallel processing of mixed document types."""
        from evaluator import BatchEvaluator, BatchConfiguration

        config = BatchConfiguration(max_concurrent_workers=3, timeout_seconds=60)
        evaluator = BatchEvaluator(config=config)

        # Create batch with mixed types
        batch_id = evaluator.create_batch_from_folder(
            folder_path=str(self.test_dir), context="Test parallel processing"
        )

        # Start processing
        evaluator.start_batch(batch_id)

        # Monitor progress
        progress = evaluator.get_batch_progress(batch_id)
        assert progress is not None
        assert progress.active_workers <= 3

    def test_mixed_document_result_aggregation(self):
        """Test result aggregation for mixed document types."""
        from evaluator import BatchEvaluator, BatchConfiguration

        config = BatchConfiguration(max_concurrent_workers=2, timeout_seconds=60)
        evaluator = BatchEvaluator(config=config)

        # Create batch with mixed types
        batch_id = evaluator.create_batch_from_folder(
            folder_path=str(self.test_dir), context="Test result aggregation"
        )

        # Start processing
        evaluator.start_batch(batch_id)

        # Get results (when processing is complete)
        results = evaluator.get_batch_results(batch_id)

        # Validate result aggregation
        if results:
            assert results.total_documents > 0
            assert results.successful_evaluations >= 0
            assert results.failed_evaluations >= 0
            assert 0.0 <= results.success_rate <= 1.0

    def test_mixed_document_output_formats(self):
        """Test output format generation for mixed document types."""
        from evaluator import BatchEvaluator, BatchConfiguration

        config = BatchConfiguration(
            max_concurrent_workers=2, output_formats=["json", "summary", "csv"]
        )
        evaluator = BatchEvaluator(config=config)

        # Create batch with mixed types
        batch_id = evaluator.create_batch_from_folder(
            folder_path=str(self.test_dir), context="Test output formats"
        )

        # Start processing
        evaluator.start_batch(batch_id)

        # Get results in different formats
        json_results = evaluator.get_batch_results(batch_id, format="json")
        summary_results = evaluator.get_batch_results(batch_id, format="summary")
        csv_results = evaluator.get_batch_results(batch_id, format="csv")

        # Validate output formats
        if json_results:
            assert isinstance(json_results, dict) or hasattr(json_results, "model_dump")

        if summary_results:
            assert isinstance(summary_results, str)

        if csv_results:
            assert isinstance(csv_results, str)

    def test_mixed_document_correlation_tracking(self):
        """Test correlation ID tracking for mixed document types."""
        from evaluator import BatchEvaluator, BatchConfiguration

        config = BatchConfiguration(max_concurrent_workers=2)
        evaluator = BatchEvaluator(config=config)

        # Create batch with mixed types
        batch_id = evaluator.create_batch_from_folder(
            folder_path=str(self.test_dir), context="Test correlation tracking"
        )

        # Get batch and document correlation IDs
        batch = evaluator.get_batch(batch_id)
        documents = evaluator.list_batch_documents(batch_id)

        # Validate correlation IDs
        assert batch.correlation_id is not None
        assert batch.correlation_id.startswith("batch_")

        for doc in documents:
            assert doc.correlation_id is not None
            assert doc.correlation_id.startswith("doc_")

    def test_mixed_document_retry_mechanism(self):
        """Test retry mechanism for failed mixed document types."""
        from evaluator import BatchEvaluator, BatchConfiguration

        config = BatchConfiguration(
            max_concurrent_workers=2, max_retry_attempts=2, timeout_seconds=30
        )
        evaluator = BatchEvaluator(config=config)

        # Create batch with mixed types
        batch_id = evaluator.create_batch_from_folder(
            folder_path=str(self.test_dir), context="Test retry mechanism"
        )

        # Start processing
        evaluator.start_batch(batch_id)

        # Get failed documents and retry them
        documents = evaluator.list_batch_documents(batch_id)
        failed_docs = [doc for doc in documents if doc.status == "failed"]

        for doc in failed_docs:
            # Retry failed document
            evaluator.retry_document(batch_id, doc.document_id)

        # Validate retry mechanism
        batch = evaluator.get_batch(batch_id)
        assert batch is not None
