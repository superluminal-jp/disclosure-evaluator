"""
Unit tests for parallel document processing service.

These tests validate the parallel processing functionality.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
from evaluator import (
    ParallelDocumentProcessingService,
    BatchConfiguration,
    BatchDocument,
    DocumentStatus,
    DisclosureEvaluationResult,
    CriterionEvaluation,
    EvaluationStep,
)


class TestParallelDocumentProcessingService:
    """Test ParallelDocumentProcessingService"""

    def setup_method(self):
        """Set up test environment"""
        self.test_dir = Path(tempfile.mkdtemp())
        self.config = BatchConfiguration(max_concurrent_workers=2)
        self.llm_provider = MagicMock()
        self.service = ParallelDocumentProcessingService(self.config, self.llm_provider)
        self.create_test_documents()

    def teardown_method(self):
        """Clean up test environment"""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def create_test_documents(self):
        """Create test documents for processing"""
        # Create test documents
        (self.test_dir / "document1.txt").write_text("Test document 1 content")
        (self.test_dir / "document2.txt").write_text("Test document 2 content")
        (self.test_dir / "document3.txt").write_text("Test document 3 content")

        # Create test batch documents
        self.batch_documents = []
        for i in range(3):
            doc = BatchDocument(
                document_id=f"doc_{i:03d}",
                batch_id="test_batch",
                file_path=str(self.test_dir / f"document{i+1}.txt"),
                file_name=f"document{i+1}.txt",
                file_size=100,
                mime_type="text/plain",
                status=DocumentStatus.PENDING,
                correlation_id=f"doc_doc_{i:03d}",
            )
            self.batch_documents.append(doc)

    def test_process_documents_parallel_basic(self):
        """Test basic parallel document processing"""
        # Mock the LLM provider response
        mock_result = DisclosureEvaluationResult(
            input_text="Test document content",
            context="",
            output_text="",
            criterion_evaluations=[
                CriterionEvaluation(
                    criterion_id="article_5_1",
                    criterion_name="Personal Information",
                    article="Article 5(1)",
                    steps=[
                        EvaluationStep(
                            step="Step 1",
                            result="YES",
                            reasoning="Contains personal information",
                        )
                    ],
                    score=3,
                    score_reasoning="Moderate disclosure risk",
                )
            ],
            evaluation_timestamp=datetime.now().isoformat(),
        )

        self.llm_provider.generate_response.return_value = "Mock LLM response"

        # Mock the DisclosureEvaluator
        with patch("evaluator.DisclosureEvaluator") as mock_evaluator_class:
            mock_evaluator = MagicMock()
            mock_evaluator.evaluate_disclosure.return_value = mock_result
            mock_evaluator_class.return_value = mock_evaluator

            # Process documents
            results = self.service.process_documents_parallel(self.batch_documents)

            # Verify results
            assert len(results) == 3
            for result in results:
                assert result.status == DocumentStatus.COMPLETED
                assert result.evaluation_result is not None
                assert result.processing_started_at is not None
                assert result.processing_completed_at is not None

    def test_process_documents_parallel_with_progress_callback(self):
        """Test parallel processing with progress callback"""
        progress_updates = []

        def progress_callback(doc):
            progress_updates.append(doc)

        # Mock the LLM provider response
        mock_result = DisclosureEvaluationResult(
            input_text="Test document content",
            context="",
            output_text="",
            criterion_evaluations=[],
            evaluation_timestamp=datetime.now().isoformat(),
        )

        self.llm_provider.generate_response.return_value = "Mock LLM response"

        # Mock the DisclosureEvaluator
        with patch("evaluator.DisclosureEvaluator") as mock_evaluator_class:
            mock_evaluator = MagicMock()
            mock_evaluator.evaluate_disclosure.return_value = mock_result
            mock_evaluator_class.return_value = mock_evaluator

            # Process documents with progress callback
            results = self.service.process_documents_parallel(
                self.batch_documents, progress_callback=progress_callback
            )

            # Verify results
            assert len(results) == 3
            assert len(progress_updates) == 3

    def test_process_documents_parallel_with_failures(self):
        """Test parallel processing with document failures"""
        # Mock the LLM provider to raise an error
        self.llm_provider.generate_response.side_effect = Exception("LLM API error")

        # Mock the DisclosureEvaluator to raise an error
        with patch("evaluator.DisclosureEvaluator") as mock_evaluator_class:
            mock_evaluator = MagicMock()
            mock_evaluator.evaluate_disclosure.side_effect = Exception(
                "Evaluation failed"
            )
            mock_evaluator_class.return_value = mock_evaluator

            # Process documents
            results = self.service.process_documents_parallel(self.batch_documents)

            # Verify results
            assert len(results) == 3
            for result in results:
                assert result.status == DocumentStatus.FAILED
                assert result.error_message is not None
                assert result.processing_started_at is not None
                assert result.processing_completed_at is not None

    def test_process_documents_parallel_timeout(self):
        """Test parallel processing with timeout"""
        # Create configuration with short timeout
        config = BatchConfiguration(max_concurrent_workers=2, timeout_seconds=30)
        service = ParallelDocumentProcessingService(config, self.llm_provider)

        # Mock the DisclosureEvaluator to simulate timeout
        with patch("evaluator.DisclosureEvaluator") as mock_evaluator_class:
            mock_evaluator = MagicMock()

            def timeout_evaluation(*args, **kwargs):
                # Simulate a timeout by raising a TimeoutError
                raise TimeoutError("Document processing timed out")

            mock_evaluator.evaluate_disclosure.side_effect = timeout_evaluation
            mock_evaluator_class.return_value = mock_evaluator

            # Process documents
            results = service.process_documents_parallel(self.batch_documents)

            # Verify results (should have timeout errors)
            assert len(results) == 3
            for result in results:
                assert result.status == DocumentStatus.FAILED
                assert result.error_message is not None
                assert "timed out" in result.error_message.lower()

    def test_process_documents_parallel_memory_management(self):
        """Test parallel processing memory management"""
        # Create many documents to test memory management
        many_documents = []
        for i in range(10):
            # Create the actual file
            file_path = self.test_dir / f"document{i+1}.txt"
            file_path.write_text(f"Test document {i+1} content")

            doc = BatchDocument(
                document_id=f"doc_{i:03d}",
                batch_id="test_batch",
                file_path=str(file_path),
                file_name=f"document{i+1}.txt",
                file_size=100,
                mime_type="text/plain",
                status=DocumentStatus.PENDING,
                correlation_id=f"doc_doc_{i:03d}",
            )
            many_documents.append(doc)

        # Mock the LLM provider response
        mock_result = DisclosureEvaluationResult(
            input_text="Test document content",
            context="",
            output_text="",
            criterion_evaluations=[],
            evaluation_timestamp=datetime.now().isoformat(),
        )

        self.llm_provider.generate_response.return_value = "Mock LLM response"

        # Mock the DisclosureEvaluator
        with patch("evaluator.DisclosureEvaluator") as mock_evaluator_class:
            mock_evaluator = MagicMock()
            mock_evaluator.evaluate_disclosure.return_value = mock_result
            mock_evaluator_class.return_value = mock_evaluator

            # Process documents
            results = self.service.process_documents_parallel(many_documents)

            # Verify results
            assert len(results) == 10
            for result in results:
                assert result.status == DocumentStatus.COMPLETED

    def test_process_documents_parallel_concurrent_workers(self):
        """Test parallel processing with different worker counts"""
        # Test with different worker counts
        for worker_count in [1, 2, 4]:
            config = BatchConfiguration(max_concurrent_workers=worker_count)
            service = ParallelDocumentProcessingService(config, self.llm_provider)

            # Mock the LLM provider response
            mock_result = DisclosureEvaluationResult(
                input_text="Test document content",
                context="",
                output_text="",
                criterion_evaluations=[],
                evaluation_timestamp=datetime.now().isoformat(),
            )

            self.llm_provider.generate_response.return_value = "Mock LLM response"

            # Mock the DisclosureEvaluator
            with patch("evaluator.DisclosureEvaluator") as mock_evaluator_class:
                mock_evaluator = MagicMock()
                mock_evaluator.evaluate_disclosure.return_value = mock_result
                mock_evaluator_class.return_value = mock_evaluator

                # Process documents
                results = service.process_documents_parallel(self.batch_documents)

                # Verify results
                assert len(results) == 3
                for result in results:
                    assert result.status == DocumentStatus.COMPLETED

    def test_process_documents_parallel_mixed_success_failure(self):
        """Test parallel processing with mixed success and failure"""

        # Mock the LLM provider to fail for some documents
        def mock_generate_response(*args, **kwargs):
            # Fail for document 2, succeed for others
            if "document2.txt" in str(args):
                raise Exception("Document 2 processing failed")
            return "Mock LLM response"

        self.llm_provider.generate_response.side_effect = mock_generate_response

        # Mock the DisclosureEvaluator
        with patch("evaluator.DisclosureEvaluator") as mock_evaluator_class:
            mock_evaluator = MagicMock()

            def mock_evaluate_disclosure(*args, **kwargs):
                # Check if this is document 2 by looking at the input_text content
                input_text = kwargs.get("input_text", "")
                if "Test document 2 content" in input_text:
                    raise Exception("Document 2 evaluation failed")
                return DisclosureEvaluationResult(
                    input_text="Test document content",
                    context="",
                    output_text="",
                    criterion_evaluations=[],
                    evaluation_timestamp=datetime.now().isoformat(),
                )

            mock_evaluator.evaluate_disclosure.side_effect = mock_evaluate_disclosure
            mock_evaluator_class.return_value = mock_evaluator

            # Process documents
            results = self.service.process_documents_parallel(self.batch_documents)

            # Verify results
            assert len(results) == 3

            # Check success/failure distribution
            successful = [r for r in results if r.status == DocumentStatus.COMPLETED]
            failed = [r for r in results if r.status == DocumentStatus.FAILED]

            assert len(successful) == 2
            assert len(failed) == 1

            # Check that failed document has error message
            failed_doc = failed[0]
            assert failed_doc.error_message is not None
            assert "Document 2" in failed_doc.error_message

    def test_process_documents_parallel_empty_list(self):
        """Test parallel processing with empty document list"""
        results = self.service.process_documents_parallel([])
        assert len(results) == 0

    def test_process_documents_parallel_single_document(self):
        """Test parallel processing with single document"""
        single_doc = [self.batch_documents[0]]

        # Mock the LLM provider response
        mock_result = DisclosureEvaluationResult(
            input_text="Test document content",
            context="",
            output_text="",
            criterion_evaluations=[],
            evaluation_timestamp=datetime.now().isoformat(),
        )

        self.llm_provider.generate_response.return_value = "Mock LLM response"

        # Mock the DisclosureEvaluator
        with patch("evaluator.DisclosureEvaluator") as mock_evaluator_class:
            mock_evaluator = MagicMock()
            mock_evaluator.evaluate_disclosure.return_value = mock_result
            mock_evaluator_class.return_value = mock_evaluator

            # Process single document
            results = self.service.process_documents_parallel(single_doc)

            # Verify results
            assert len(results) == 1
            assert results[0].status == DocumentStatus.COMPLETED

    def test_process_documents_parallel_large_documents(self):
        """Test parallel processing with large documents"""
        # Create large document
        large_file = self.test_dir / "large_document.txt"
        large_content = "Large content " * 10000
        large_file.write_text(large_content)

        large_doc = BatchDocument(
            document_id="doc_large",
            batch_id="test_batch",
            file_path=str(large_file),
            file_name="large_document.txt",
            file_size=len(large_content.encode()),
            mime_type="text/plain",
            status=DocumentStatus.PENDING,
            correlation_id="doc_doc_large",
        )

        # Mock the LLM provider response
        mock_result = DisclosureEvaluationResult(
            input_text=large_content,
            context="",
            output_text="",
            criterion_evaluations=[],
            evaluation_timestamp=datetime.now().isoformat(),
        )

        self.llm_provider.generate_response.return_value = "Mock LLM response"

        # Mock the DisclosureEvaluator
        with patch("evaluator.DisclosureEvaluator") as mock_evaluator_class:
            mock_evaluator = MagicMock()
            mock_evaluator.evaluate_disclosure.return_value = mock_result
            mock_evaluator_class.return_value = mock_evaluator

            # Process large document
            results = self.service.process_documents_parallel([large_doc])

            # Verify results
            assert len(results) == 1
            assert results[0].status == DocumentStatus.COMPLETED
            assert results[0].evaluation_result is not None

    def test_process_documents_parallel_retry_mechanism(self):
        """Test parallel processing retry mechanism"""
        # Create configuration with retry attempts
        config = BatchConfiguration(max_concurrent_workers=2, max_retry_attempts=2)
        service = ParallelDocumentProcessingService(config, self.llm_provider)

        # Mock the DisclosureEvaluator with retry logic
        with patch("evaluator.DisclosureEvaluator") as mock_evaluator_class:
            mock_evaluator = MagicMock()

            # Track attempts per document
            attempt_counts = {}

            def mock_evaluate_disclosure(*args, **kwargs):
                # Get document ID from the file path to track attempts per document
                input_text = kwargs.get("input_text", "")
                doc_key = (
                    input_text  # Use content as key since it's unique per document
                )

                if doc_key not in attempt_counts:
                    attempt_counts[doc_key] = 0
                attempt_counts[doc_key] += 1

                # Fail first two times, succeed on third
                if attempt_counts[doc_key] <= 2:
                    raise Exception("Temporary failure")

                return DisclosureEvaluationResult(
                    input_text="Test document content",
                    context="",
                    output_text="",
                    criterion_evaluations=[],
                    evaluation_timestamp=datetime.now().isoformat(),
                )

            mock_evaluator.evaluate_disclosure.side_effect = mock_evaluate_disclosure
            mock_evaluator_class.return_value = mock_evaluator

            # Process documents
            results = service.process_documents_parallel(self.batch_documents)

            # Verify results
            assert len(results) == 3
            for result in results:
                assert result.status == DocumentStatus.COMPLETED
