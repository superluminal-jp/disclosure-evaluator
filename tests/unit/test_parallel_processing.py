"""
Unit tests for ParallelDocumentProcessingService class.
Tests lines 1874-1988: ParallelDocumentProcessingService with thread safety verification
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

from main import (
    ParallelDocumentProcessingService,
    BatchConfiguration,
    BatchDocument,
    DocumentStatus,
    DisclosureEvaluationResult,
    EvaluationStep,
    CriterionEvaluation,
)


class TestParallelDocumentProcessingService:
    """Test ParallelDocumentProcessingService class."""

    def test_parallel_document_processing_service_init(
        self, test_batch_configuration, mock_openai_client
    ):
        """Test ParallelDocumentProcessingService initialization."""
        mock_llm_provider = Mock()

        with patch("main.logging.getLogger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            service = ParallelDocumentProcessingService(
                test_batch_configuration, mock_llm_provider
            )

            assert service.config == test_batch_configuration
            assert service.llm_provider == mock_llm_provider
            assert service.logger == mock_logger
            mock_get_logger.assert_called_once_with("ParallelDocumentProcessingService")

    def test_process_documents_parallel_success(
        self, test_batch_configuration, mock_openai_client
    ):
        """Test successful parallel document processing."""
        mock_llm_provider = Mock()

        # Create test documents
        documents = [
            BatchDocument(
                document_id="doc_001",
                batch_id="batch_001",
                file_path="/test/path/doc1.txt",
                file_name="doc1.txt",
                file_size=1024,
                mime_type="text/plain",
                status=DocumentStatus.PENDING,
                correlation_id="correlation_001",
                context="Test context 1",
                output_text="Test output 1",
            ),
            BatchDocument(
                document_id="doc_002",
                batch_id="batch_001",
                file_path="/test/path/doc2.txt",
                file_name="doc2.txt",
                file_size=2048,
                mime_type="text/plain",
                status=DocumentStatus.PENDING,
                correlation_id="correlation_002",
                context="Test context 2",
                output_text="Test output 2",
            ),
        ]

        # Mock progress callback
        progress_callback = Mock()

        with patch("main.logging.getLogger"):
            with patch("main.ThreadPoolExecutor") as mock_executor_class:
                mock_executor = Mock()
                mock_executor.__enter__ = Mock(return_value=mock_executor)
                mock_executor.__exit__ = Mock()
                mock_executor.submit.return_value = Mock()
                mock_executor_class.return_value = mock_executor

                with patch("main.as_completed") as mock_as_completed:
                    # Mock successful processing
                    mock_future1 = Mock()
                    mock_future1.result.return_value = documents[0]
                    mock_future2 = Mock()
                    mock_future2.result.return_value = documents[1]
                    mock_as_completed.return_value = [mock_future1, mock_future2]

                    with patch.object(
                        ParallelDocumentProcessingService, "_process_single_document"
                    ) as mock_process:
                        mock_process.side_effect = documents

                        service = ParallelDocumentProcessingService(
                            test_batch_configuration, mock_llm_provider
                        )
                        results = service.process_documents_parallel(
                            documents, progress_callback
                        )

                        assert len(results) == 2
                        assert results[0].document_id == "doc_001"
                        assert results[1].document_id == "doc_002"

                        # Verify ThreadPoolExecutor was used
                        mock_executor_class.assert_called_once_with(
                            max_workers=test_batch_configuration.max_concurrent_workers
                        )
                        mock_executor.submit.assert_called()

                        # Verify progress callback was called
                        progress_callback.assert_called()

    def test_process_documents_parallel_exception(
        self, test_batch_configuration, mock_openai_client
    ):
        """Test parallel document processing with exception."""
        mock_llm_provider = Mock()

        documents = [
            BatchDocument(
                document_id="doc_001",
                batch_id="batch_001",
                file_path="/test/path/doc1.txt",
                file_name="doc1.txt",
                file_size=1024,
                mime_type="text/plain",
                status=DocumentStatus.PENDING,
                correlation_id="correlation_001",
                context="Test context",
                output_text="Test output",
            )
        ]

        with patch("main.logging.getLogger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            with patch("main.ThreadPoolExecutor") as mock_executor_class:
                mock_executor = Mock()
                mock_executor.__enter__ = Mock(return_value=mock_executor)
                mock_executor.__exit__ = Mock()
                mock_executor.submit.side_effect = Exception("ThreadPool error")
                mock_executor_class.return_value = mock_executor

                service = ParallelDocumentProcessingService(
                    test_batch_configuration, mock_llm_provider
                )

                with pytest.raises(
                    Exception,
                    match="Parallel document processing failed: ThreadPool error",
                ):
                    service.process_documents_parallel(documents)

                mock_logger.error.assert_called()

    def test_process_single_document_success(
        self, test_batch_configuration, mock_openai_client
    ):
        """Test successful single document processing."""
        mock_llm_provider = Mock()

        document = BatchDocument(
            document_id="doc_001",
            batch_id="batch_001",
            file_path="/test/path/doc1.txt",
            file_name="doc1.txt",
            file_size=1024,
            mime_type="text/plain",
            status=DocumentStatus.PENDING,
            correlation_id="correlation_001",
            context="Test context",
            output_text="Test output",
        )

        # Mock evaluation result
        mock_evaluation_result = DisclosureEvaluationResult(
            input_text="Test input",
            context="Test context",
            output_text="Test output",
            criterion_evaluations=[
                CriterionEvaluation(
                    criterion_id="article_5_1",
                    criterion_name="個人情報保護",
                    article="第5条第1号",
                    steps=[
                        EvaluationStep(
                            step="ステップ1", result="YES", reasoning="理由1"
                        )
                    ],
                    score=3,
                    score_reasoning="スコア理由",
                )
            ],
            evaluation_timestamp="2025-01-01T12:00:00",
        )

        with patch("main.logging.getLogger"):
            with patch("builtins.open", mock_open(read_data="Test document content")):
                with patch("main.DisclosureEvaluator") as mock_evaluator_class:
                    mock_evaluator = Mock()
                    mock_evaluator.evaluate_disclosure.return_value = (
                        mock_evaluation_result
                    )
                    mock_evaluator_class.return_value = mock_evaluator

                    service = ParallelDocumentProcessingService(
                        test_batch_configuration, mock_llm_provider
                    )
                    result = service._process_single_document(document)

                    assert result.document_id == "doc_001"
                    assert result.status == DocumentStatus.COMPLETED
                    assert result.evaluation_result == mock_evaluation_result
                    assert result.success is True

                    # Verify file was read
                    mock_evaluator.evaluate_disclosure.assert_called_once()

    def test_process_single_document_file_read_error(
        self, test_batch_configuration, mock_openai_client
    ):
        """Test single document processing with file read error."""
        mock_llm_provider = Mock()

        document = BatchDocument(
            document_id="doc_001",
            batch_id="batch_001",
            file_path="/test/path/doc1.txt",
            file_name="doc1.txt",
            file_size=1024,
            mime_type="text/plain",
            status=DocumentStatus.PENDING,
            correlation_id="correlation_001",
            context="Test context",
            output_text="Test output",
        )

        with patch("main.logging.getLogger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            with patch("builtins.open", side_effect=IOError("File not found")):
                service = ParallelDocumentProcessingService(
                    test_batch_configuration, mock_llm_provider
                )
                result = service._process_single_document(document)

                assert result.document_id == "doc_001"
                assert result.status == DocumentStatus.FAILED
                assert result.evaluation_result is None
                assert result.success is False
                assert result.error_info is not None
                assert result.error_info.error_type == "FILE_READ_ERROR"
                assert "File not found" in result.error_info.error_message

                mock_logger.error.assert_called()

    def test_process_single_document_evaluation_error(
        self, test_batch_configuration, mock_openai_client
    ):
        """Test single document processing with evaluation error."""
        mock_llm_provider = Mock()

        document = BatchDocument(
            document_id="doc_001",
            batch_id="batch_001",
            file_path="/test/path/doc1.txt",
            file_name="doc1.txt",
            file_size=1024,
            mime_type="text/plain",
            status=DocumentStatus.PENDING,
            correlation_id="correlation_001",
            context="Test context",
            output_text="Test output",
        )

        with patch("main.logging.getLogger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            with patch("builtins.open", mock_open(read_data="Test document content")):
                with patch("main.DisclosureEvaluator") as mock_evaluator_class:
                    mock_evaluator = Mock()
                    mock_evaluator.evaluate_disclosure.side_effect = Exception(
                        "Evaluation failed"
                    )
                    mock_evaluator_class.return_value = mock_evaluator

                    service = ParallelDocumentProcessingService(
                        test_batch_configuration, mock_llm_provider
                    )
                    result = service._process_single_document(document)

                    assert result.document_id == "doc_001"
                    assert result.status == DocumentStatus.FAILED
                    assert result.evaluation_result is None
                    assert result.success is False
                    assert result.error_info is not None
                    assert result.error_info.error_type == "EVALUATION_ERROR"
                    assert "Evaluation failed" in result.error_info.error_message

                    mock_logger.error.assert_called()

    def test_process_single_document_retry_logic(
        self, test_batch_configuration, mock_openai_client
    ):
        """Test single document processing with retry logic."""
        mock_llm_provider = Mock()

        document = BatchDocument(
            document_id="doc_001",
            batch_id="batch_001",
            file_path="/test/path/doc1.txt",
            file_name="doc1.txt",
            file_size=1024,
            mime_type="text/plain",
            status=DocumentStatus.PENDING,
            correlation_id="correlation_001",
            context="Test context",
            output_text="Test output",
            retry_count=1,
        )

        with patch("main.logging.getLogger"):
            with patch("builtins.open", mock_open(read_data="Test document content")):
                with patch("main.DisclosureEvaluator") as mock_evaluator_class:
                    mock_evaluator = Mock()
                    mock_evaluator.evaluate_disclosure.side_effect = Exception(
                        "Temporary error"
                    )
                    mock_evaluator_class.return_value = mock_evaluator

                    service = ParallelDocumentProcessingService(
                        test_batch_configuration, mock_llm_provider
                    )
                    result = service._process_single_document(document)

                    assert result.document_id == "doc_001"
                    assert result.status == DocumentStatus.FAILED
                    assert result.retry_count == 2  # Should be incremented

    def test_process_single_document_max_retries_exceeded(
        self, test_batch_configuration, mock_openai_client
    ):
        """Test single document processing when max retries are exceeded."""
        mock_llm_provider = Mock()

        document = BatchDocument(
            document_id="doc_001",
            batch_id="batch_001",
            file_path="/test/path/doc1.txt",
            file_name="doc1.txt",
            file_size=1024,
            mime_type="text/plain",
            status=DocumentStatus.PENDING,
            correlation_id="correlation_001",
            context="Test context",
            output_text="Test output",
            retry_count=3,  # Already at max retries
        )

        with patch("main.logging.getLogger"):
            with patch("builtins.open", mock_open(read_data="Test document content")):
                with patch("main.DisclosureEvaluator") as mock_evaluator_class:
                    mock_evaluator = Mock()
                    mock_evaluator.evaluate_disclosure.side_effect = Exception(
                        "Persistent error"
                    )
                    mock_evaluator_class.return_value = mock_evaluator

                    service = ParallelDocumentProcessingService(
                        test_batch_configuration, mock_llm_provider
                    )
                    result = service._process_single_document(document)

                    assert result.document_id == "doc_001"
                    assert result.status == DocumentStatus.FAILED
                    assert result.retry_count == 3  # Should not be incremented
                    assert result.error_info.error_type == "MAX_RETRIES_EXCEEDED"

    def test_process_single_document_timeout(
        self, test_batch_configuration, mock_openai_client
    ):
        """Test single document processing with timeout."""
        mock_llm_provider = Mock()

        document = BatchDocument(
            document_id="doc_001",
            batch_id="batch_001",
            file_path="/test/path/doc1.txt",
            file_name="doc1.txt",
            file_size=1024,
            mime_type="text/plain",
            status=DocumentStatus.PENDING,
            correlation_id="correlation_001",
            context="Test context",
            output_text="Test output",
        )

        with patch("main.logging.getLogger"):
            with patch("builtins.open", mock_open(read_data="Test document content")):
                with patch("main.DisclosureEvaluator") as mock_evaluator_class:
                    mock_evaluator = Mock()
                    mock_evaluator.evaluate_disclosure.side_effect = TimeoutError(
                        "Evaluation timeout"
                    )
                    mock_evaluator_class.return_value = mock_evaluator

                    service = ParallelDocumentProcessingService(
                        test_batch_configuration, mock_llm_provider
                    )
                    result = service._process_single_document(document)

                    assert result.document_id == "doc_001"
                    assert result.status == DocumentStatus.FAILED
                    assert result.error_info.error_type == "TIMEOUT_ERROR"
                    assert "Evaluation timeout" in result.error_info.error_message

    def test_process_single_document_large_file(
        self, test_batch_configuration, mock_openai_client
    ):
        """Test single document processing with large file."""
        mock_llm_provider = Mock()

        document = BatchDocument(
            document_id="doc_001",
            batch_id="batch_001",
            file_path="/test/path/large_doc.txt",
            file_name="large_doc.txt",
            file_size=100 * 1024 * 1024,  # 100MB
            mime_type="text/plain",
            status=DocumentStatus.PENDING,
            correlation_id="correlation_001",
            context="Test context",
            output_text="Test output",
        )

        # Mock large file content
        large_content = "x" * (100 * 1024 * 1024)  # 100MB of content

        with patch("main.logging.getLogger"):
            with patch("builtins.open", mock_open(read_data=large_content)):
                with patch("main.DisclosureEvaluator") as mock_evaluator_class:
                    mock_evaluator = Mock()
                    mock_evaluation_result = DisclosureEvaluationResult(
                        input_text=large_content[:1000],  # Truncated for evaluation
                        context="Test context",
                        output_text="Test output",
                        criterion_evaluations=[],
                        evaluation_timestamp="2025-01-01T12:00:00",
                    )
                    mock_evaluator.evaluate_disclosure.return_value = (
                        mock_evaluation_result
                    )
                    mock_evaluator_class.return_value = mock_evaluator

                    service = ParallelDocumentProcessingService(
                        test_batch_configuration, mock_llm_provider
                    )
                    result = service._process_single_document(document)

                    assert result.document_id == "doc_001"
                    assert result.status == DocumentStatus.COMPLETED
                    assert result.success is True
                    assert result.evaluation_result is not None

    def test_process_single_document_memory_error(
        self, test_batch_configuration, mock_openai_client
    ):
        """Test single document processing with memory error."""
        mock_llm_provider = Mock()

        document = BatchDocument(
            document_id="doc_001",
            batch_id="batch_001",
            file_path="/test/path/doc1.txt",
            file_name="doc1.txt",
            file_size=1024,
            mime_type="text/plain",
            status=DocumentStatus.PENDING,
            correlation_id="correlation_001",
            context="Test context",
            output_text="Test output",
        )

        with patch("main.logging.getLogger"):
            with patch("builtins.open", side_effect=MemoryError("Out of memory")):
                service = ParallelDocumentProcessingService(
                    test_batch_configuration, mock_llm_provider
                )
                result = service._process_single_document(document)

                assert result.document_id == "doc_001"
                assert result.status == DocumentStatus.FAILED
                assert result.error_info.error_type == "MEMORY_ERROR"
                assert "Out of memory" in result.error_info.error_message

    def test_process_single_document_permission_error(
        self, test_batch_configuration, mock_openai_client
    ):
        """Test single document processing with permission error."""
        mock_llm_provider = Mock()

        document = BatchDocument(
            document_id="doc_001",
            batch_id="batch_001",
            file_path="/test/path/doc1.txt",
            file_name="doc1.txt",
            file_size=1024,
            mime_type="text/plain",
            status=DocumentStatus.PENDING,
            correlation_id="correlation_001",
            context="Test context",
            output_text="Test output",
        )

        with patch("main.logging.getLogger"):
            with patch(
                "builtins.open", side_effect=PermissionError("Permission denied")
            ):
                service = ParallelDocumentProcessingService(
                    test_batch_configuration, mock_llm_provider
                )
                result = service._process_single_document(document)

                assert result.document_id == "doc_001"
                assert result.status == DocumentStatus.FAILED
                assert result.error_info.error_type == "PERMISSION_ERROR"
                assert "Permission denied" in result.error_info.error_message

    def test_process_single_document_unicode_content(
        self, test_batch_configuration, mock_openai_client
    ):
        """Test single document processing with Unicode content."""
        mock_llm_provider = Mock()

        document = BatchDocument(
            document_id="doc_001",
            batch_id="batch_001",
            file_path="/test/path/unicode_doc.txt",
            file_name="unicode_doc.txt",
            file_size=1024,
            mime_type="text/plain",
            status=DocumentStatus.PENDING,
            correlation_id="correlation_001",
            context="テストコンテキスト",
            output_text="テスト出力",
        )

        unicode_content = (
            "これは日本語のテスト文書です。個人情報: 田中太郎、住所: 東京都渋谷区"
        )

        with patch("main.logging.getLogger"):
            with patch("builtins.open", mock_open(read_data=unicode_content)):
                with patch("main.DisclosureEvaluator") as mock_evaluator_class:
                    mock_evaluator = Mock()
                    mock_evaluation_result = DisclosureEvaluationResult(
                        input_text=unicode_content,
                        context="テストコンテキスト",
                        output_text="テスト出力",
                        criterion_evaluations=[],
                        evaluation_timestamp="2025-01-01T12:00:00",
                    )
                    mock_evaluator.evaluate_disclosure.return_value = (
                        mock_evaluation_result
                    )
                    mock_evaluator_class.return_value = mock_evaluator

                    service = ParallelDocumentProcessingService(
                        test_batch_configuration, mock_llm_provider
                    )
                    result = service._process_single_document(document)

                    assert result.document_id == "doc_001"
                    assert result.status == DocumentStatus.COMPLETED
                    assert result.success is True
                    assert result.evaluation_result.input_text == unicode_content
                    assert result.evaluation_result.context == "テストコンテキスト"
                    assert result.evaluation_result.output_text == "テスト出力"

    def test_process_documents_parallel_empty_list(
        self, test_batch_configuration, mock_openai_client
    ):
        """Test parallel document processing with empty document list."""
        mock_llm_provider = Mock()

        with patch("main.logging.getLogger"):
            with patch("main.ThreadPoolExecutor") as mock_executor_class:
                mock_executor = Mock()
                mock_executor.__enter__ = Mock(return_value=mock_executor)
                mock_executor.__exit__ = Mock()
                mock_executor_class.return_value = mock_executor

                service = ParallelDocumentProcessingService(
                    test_batch_configuration, mock_llm_provider
                )
                results = service.process_documents_parallel([])

                assert len(results) == 0
                mock_executor.submit.assert_not_called()

    def test_process_documents_parallel_progress_callback(
        self, test_batch_configuration, mock_openai_client
    ):
        """Test parallel document processing with progress callback."""
        mock_llm_provider = Mock()

        documents = [
            BatchDocument(
                document_id="doc_001",
                batch_id="batch_001",
                file_path="/test/path/doc1.txt",
                file_name="doc1.txt",
                file_size=1024,
                mime_type="text/plain",
                status=DocumentStatus.PENDING,
                correlation_id="correlation_001",
                context="Test context",
                output_text="Test output",
            )
        ]

        progress_callback = Mock()

        with patch("main.logging.getLogger"):
            with patch("main.ThreadPoolExecutor") as mock_executor_class:
                mock_executor = Mock()
                mock_executor.__enter__ = Mock(return_value=mock_executor)
                mock_executor.__exit__ = Mock()
                mock_executor.submit.return_value = Mock()
                mock_executor_class.return_value = mock_executor

                with patch("main.as_completed") as mock_as_completed:
                    mock_future = Mock()
                    mock_future.result.return_value = documents[0]
                    mock_as_completed.return_value = [mock_future]

                    with patch.object(
                        ParallelDocumentProcessingService, "_process_single_document"
                    ) as mock_process:
                        mock_process.return_value = documents[0]

                        service = ParallelDocumentProcessingService(
                            test_batch_configuration, mock_llm_provider
                        )
                        results = service.process_documents_parallel(
                            documents, progress_callback
                        )

                        # Verify progress callback was called
                        progress_callback.assert_called()
                        assert len(results) == 1
