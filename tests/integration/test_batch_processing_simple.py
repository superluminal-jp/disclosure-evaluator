"""
Simplified integration tests for batch processing workflow.
Tests the core batch processing functionality with simplified mocking.
"""

import json
import pytest
from unittest.mock import Mock, patch, mock_open
from datetime import datetime, timedelta
from pathlib import Path

from main import (
    BatchEvaluator,
    BatchConfiguration,
    DocumentInput,
    BatchDocument,
    BatchEvaluation,
    BatchStatus,
    DocumentStatus,
    ProcessingPhase,
    BatchProgress,
    BatchResult,
)


class TestBatchProcessingSimple:
    """Simplified batch processing tests focusing on core functionality."""

    def test_batch_creation_success(self, test_batch_configuration):
        """Test successful batch creation."""
        documents = [
            DocumentInput(file_path="/test/doc1.txt", file_name="doc1.txt"),
            DocumentInput(file_path="/test/doc2.txt", file_name="doc2.txt"),
        ]

        with patch("main.logging.getLogger"):
            with patch("main.BatchStatePersistenceService") as mock_state_class:
                mock_state_service = Mock()
                # 修正: active_dirのモックを適切に設定
                from pathlib import Path

                mock_active_dir = Mock(spec=Path)
                mock_active_dir.__truediv__ = Mock(
                    return_value=Path("/tmp/test_batch_documents.json")
                )
                mock_state_service.active_dir = mock_active_dir
                mock_state_class.return_value = mock_state_service

                with patch("main.create_llm_provider"):
                    with patch("main.ParallelDocumentProcessingService"):
                        with patch("main.DocumentDiscoveryService"):  # 追加
                            with patch("builtins.open", mock_open()) as mock_file:
                                with patch("pathlib.Path.exists", return_value=True):
                                    evaluator = BatchEvaluator(test_batch_configuration)

                                    # Test batch creation
                                    batch_id = evaluator.create_batch(documents)
                                    assert batch_id is not None
                                    assert batch_id.startswith("batch_")

                                    # Verify state service was called
                                    mock_state_service.save_batch_state.assert_called()

    def test_batch_creation_from_folder_success(
        self, test_batch_configuration, sample_documents_dir
    ):
        """Test successful batch creation from folder."""
        with patch("main.logging.getLogger"):
            with patch("main.DocumentDiscoveryService") as mock_discovery_class:
                mock_discovery_service = Mock()
                mock_discovery_service.discover_documents_from_folder.return_value = [
                    DocumentInput(file_path="/test/doc1.txt", file_name="doc1.txt"),
                    DocumentInput(file_path="/test/doc2.txt", file_name="doc2.txt"),
                ]
                mock_discovery_class.return_value = mock_discovery_service

                with patch("main.BatchStatePersistenceService") as mock_state_class:
                    mock_state_service = Mock()
                    # 修正: active_dirのモックを適切に設定
                    from pathlib import Path

                    mock_active_dir = Mock(spec=Path)
                    mock_active_dir.__truediv__ = Mock(
                        return_value=Path("/tmp/test_batch_documents.json")
                    )
                    mock_state_service.active_dir = mock_active_dir
                    mock_state_class.return_value = mock_state_service

                    with patch("main.create_llm_provider"):
                        with patch("main.ParallelDocumentProcessingService"):
                            with patch("builtins.open", mock_open()) as mock_file:
                                with patch("pathlib.Path.exists", return_value=True):
                                    evaluator = BatchEvaluator(test_batch_configuration)

                                    # Test batch creation from folder
                                    batch_id = evaluator.create_batch_from_folder(
                                        sample_documents_dir
                                    )
                                    assert batch_id is not None
                                    assert batch_id.startswith("batch_")

                                    # Verify discovery service was called
                                    mock_discovery_service.discover_documents_from_folder.assert_called_once()

    def test_batch_status_retrieval(self, test_batch_configuration):
        """Test batch status retrieval."""
        with patch("main.logging.getLogger"):
            with patch("main.BatchStatePersistenceService") as mock_state_class:
                mock_state_service = Mock()
                # 修正: active_dirのモックを適切に設定
                from pathlib import Path

                mock_active_dir = Mock(spec=Path)
                mock_active_dir.__truediv__ = Mock(
                    return_value=Path("/tmp/test_batch_documents.json")
                )
                mock_state_service.active_dir = mock_active_dir
                mock_batch_state = BatchEvaluation(
                    batch_id="batch_001",
                    created_at=datetime.now(),
                    status=BatchStatus.PROCESSING,
                    total_documents=2,
                    processed_documents=1,
                    successful_documents=1,
                    failed_documents=0,
                    correlation_id="correlation_001",
                    configuration=test_batch_configuration,
                )
                mock_state_service.load_batch_state.return_value = mock_batch_state
                mock_state_class.return_value = mock_state_service

                with patch("main.create_llm_provider"):
                    with patch("main.ParallelDocumentProcessingService"):
                        with patch("main.DocumentDiscoveryService"):
                            evaluator = BatchEvaluator(test_batch_configuration)

                            # Test batch retrieval
                            batch = evaluator.get_batch("batch_001")
                            assert batch is not None
                            assert batch.batch_id == "batch_001"
                            assert batch.status == BatchStatus.PROCESSING

    def test_batch_progress_calculation(self, test_batch_configuration):
        """Test batch progress calculation."""
        with patch("main.logging.getLogger"):
            with patch("main.BatchStatePersistenceService") as mock_state_class:
                mock_state_service = Mock()
                # 修正: active_dirのモックを適切に設定
                from pathlib import Path

                mock_active_dir = Mock(spec=Path)
                mock_active_dir.__truediv__ = Mock(
                    return_value=Path("/tmp/test_batch_documents.json")
                )
                mock_state_service.active_dir = mock_active_dir
                mock_batch_state = BatchEvaluation(
                    batch_id="batch_001",
                    created_at=datetime.now(),
                    status=BatchStatus.PROCESSING,
                    total_documents=4,
                    processed_documents=2,
                    successful_documents=1,
                    failed_documents=1,
                    correlation_id="correlation_001",
                    configuration=test_batch_configuration,
                )
                mock_state_service.load_batch_state.return_value = mock_batch_state
                mock_state_class.return_value = mock_state_service

                with patch("main.create_llm_provider"):
                    with patch("main.ParallelDocumentProcessingService"):
                        with patch("main.DocumentDiscoveryService"):
                            evaluator = BatchEvaluator(test_batch_configuration)

                            # Test progress calculation
                            progress = evaluator.get_batch_progress("batch_001")
                            assert progress is not None
                            assert progress.batch_id == "batch_001"
                            assert progress.total_documents == 4
                            assert progress.processed_documents == 2
                            assert progress.successful_documents == 1
                            assert progress.failed_documents == 1

    def test_batch_results_generation(self, test_batch_configuration):
        """Test batch results generation."""
        with patch("main.logging.getLogger"):
            with patch("main.BatchStatePersistenceService") as mock_state_class:
                mock_state_service = Mock()
                # 修正: active_dirのモックを適切に設定
                from pathlib import Path

                mock_active_dir = Mock(spec=Path)
                mock_active_dir.__truediv__ = Mock(
                    return_value=Path("/tmp/test_batch_documents.json")
                )
                mock_state_service.active_dir = mock_active_dir
                mock_batch_state = BatchEvaluation(
                    batch_id="batch_001",
                    created_at=datetime.now(),
                    status=BatchStatus.COMPLETED,
                    total_documents=2,
                    processed_documents=2,
                    successful_documents=2,
                    failed_documents=0,
                    correlation_id="correlation_001",
                    configuration=test_batch_configuration,
                )
                mock_state_service.load_batch_state.return_value = mock_batch_state
                mock_state_class.return_value = mock_state_service

                with patch("main.create_llm_provider"):
                    with patch("main.ParallelDocumentProcessingService"):
                        with patch("main.DocumentDiscoveryService"):
                            evaluator = BatchEvaluator(test_batch_configuration)

                            # Test results generation
                            results = evaluator.get_batch_results("batch_001")
                            assert results is not None
                            assert isinstance(results, BatchResult)
                            assert results.batch_id == "batch_001"
                            assert results.total_documents == 2
                            assert results.successful_evaluations == 2
                            assert results.failed_evaluations == 0
                            assert results.success_rate == 1.0

    def test_batch_start_with_pending_status(self, test_batch_configuration):
        """Test batch start with pending status."""
        with patch("main.logging.getLogger"):
            with patch("main.BatchStatePersistenceService") as mock_state_class:
                mock_state_service = Mock()
                # 修正: active_dirのモックを適切に設定
                from pathlib import Path

                mock_active_dir = Mock(spec=Path)
                mock_active_dir.__truediv__ = Mock(
                    return_value=Path("/tmp/test_batch_documents.json")
                )
                mock_state_service.active_dir = mock_active_dir
                mock_batch_state = BatchEvaluation(
                    batch_id="batch_001",
                    created_at=datetime.now(),
                    status=BatchStatus.PENDING,
                    total_documents=2,
                    processed_documents=0,
                    successful_documents=0,
                    failed_documents=0,
                    correlation_id="correlation_001",
                    configuration=test_batch_configuration,
                )
                mock_state_service.load_batch_state.return_value = mock_batch_state
                mock_state_class.return_value = mock_state_service

                with patch("main.create_llm_provider"):
                    with patch("main.ParallelDocumentProcessingService"):
                        with patch("main.DocumentDiscoveryService"):
                            evaluator = BatchEvaluator(test_batch_configuration)

                            # Mock the _process_batch_documents method
                            with patch.object(
                                evaluator, "_process_batch_documents"
                            ) as mock_process:
                                mock_process.return_value = None

                                # Test batch start
                                result = evaluator.start_batch("batch_001")
                                assert result is True
                                mock_process.assert_called_once_with("batch_001")

    def test_batch_start_with_non_pending_status(self, test_batch_configuration):
        """Test batch start with non-pending status raises error."""
        with patch("main.logging.getLogger"):
            with patch("main.BatchStatePersistenceService") as mock_state_class:
                mock_state_service = Mock()
                # 修正: active_dirのモックを適切に設定
                from pathlib import Path

                mock_active_dir = Mock(spec=Path)
                mock_active_dir.__truediv__ = Mock(
                    return_value=Path("/tmp/test_batch_documents.json")
                )
                mock_state_service.active_dir = mock_active_dir
                mock_batch_state = BatchEvaluation(
                    batch_id="batch_001",
                    created_at=datetime.now(),
                    status=BatchStatus.PROCESSING,
                    total_documents=2,
                    processed_documents=0,
                    successful_documents=0,
                    failed_documents=0,
                    correlation_id="correlation_001",
                    configuration=test_batch_configuration,
                )
                mock_state_service.load_batch_state.return_value = mock_batch_state
                mock_state_class.return_value = mock_state_service

                with patch("main.create_llm_provider"):
                    with patch("main.ParallelDocumentProcessingService"):
                        with patch("main.DocumentDiscoveryService"):
                            evaluator = BatchEvaluator(test_batch_configuration)

                            # Test batch start with non-pending status
                            with pytest.raises(
                                ValueError, match="is not in pending status"
                            ):
                                evaluator.start_batch("batch_001")

    def test_batch_resume_with_failed_status(self, test_batch_configuration):
        """Test batch resume with failed status."""
        with patch("main.logging.getLogger"):
            with patch("main.BatchStatePersistenceService") as mock_state_class:
                mock_state_service = Mock()
                # 修正: active_dirのモックを適切に設定
                from pathlib import Path

                mock_active_dir = Mock(spec=Path)
                mock_active_dir.__truediv__ = Mock(
                    return_value=Path("/tmp/test_batch_documents.json")
                )
                mock_state_service.active_dir = mock_active_dir
                mock_batch_state = BatchEvaluation(
                    batch_id="batch_001",
                    created_at=datetime.now(),
                    status=BatchStatus.FAILED,
                    total_documents=2,
                    processed_documents=1,
                    successful_documents=0,
                    failed_documents=1,
                    correlation_id="correlation_001",
                    configuration=test_batch_configuration,
                )
                mock_state_service.load_batch_state.return_value = mock_batch_state
                mock_state_class.return_value = mock_state_service

                with patch("main.create_llm_provider"):
                    with patch("main.ParallelDocumentProcessingService"):
                        with patch("main.DocumentDiscoveryService"):
                            evaluator = BatchEvaluator(test_batch_configuration)

                            # Test batch resume (just updates status, doesn't process)
                            result = evaluator.resume_batch("batch_001")
                            assert result is True

                            # Verify batch status was updated
                            mock_state_service.save_batch_state.assert_called()

    def test_batch_resume_with_non_failed_status(self, test_batch_configuration):
        """Test batch resume with non-failed status raises error."""
        with patch("main.logging.getLogger"):
            with patch("main.BatchStatePersistenceService") as mock_state_class:
                mock_state_service = Mock()
                # 修正: active_dirのモックを適切に設定
                from pathlib import Path

                mock_active_dir = Mock(spec=Path)
                mock_active_dir.__truediv__ = Mock(
                    return_value=Path("/tmp/test_batch_documents.json")
                )
                mock_state_service.active_dir = mock_active_dir
                mock_batch_state = BatchEvaluation(
                    batch_id="batch_001",
                    created_at=datetime.now(),
                    status=BatchStatus.COMPLETED,
                    total_documents=2,
                    processed_documents=2,
                    successful_documents=2,
                    failed_documents=0,
                    correlation_id="correlation_001",
                    configuration=test_batch_configuration,
                )
                mock_state_service.load_batch_state.return_value = mock_batch_state
                mock_state_class.return_value = mock_state_service

                with patch("main.create_llm_provider"):
                    with patch("main.ParallelDocumentProcessingService"):
                        with patch("main.DocumentDiscoveryService"):
                            evaluator = BatchEvaluator(test_batch_configuration)

                            # Test batch resume with non-failed status
                            with pytest.raises(ValueError, match="cannot be resumed"):
                                evaluator.resume_batch("batch_001")

    def test_document_retry(self, test_batch_configuration):
        """Test document retry functionality."""
        with patch("main.logging.getLogger"):
            with patch("main.BatchStatePersistenceService") as mock_state_class:
                mock_state_service = Mock()
                # 修正: active_dirのモックを適切に設定
                from pathlib import Path

                mock_active_dir = Mock(spec=Path)
                mock_active_dir.__truediv__ = Mock(
                    return_value=Path("/tmp/test_batch_documents.json")
                )
                mock_state_service.active_dir = mock_active_dir
                mock_batch_state = BatchEvaluation(
                    batch_id="batch_001",
                    created_at=datetime.now(),
                    status=BatchStatus.PROCESSING,
                    total_documents=2,
                    processed_documents=1,
                    successful_documents=1,
                    failed_documents=0,
                    correlation_id="correlation_001",
                    configuration=test_batch_configuration,
                )
                mock_state_service.load_batch_state.return_value = mock_batch_state
                mock_state_class.return_value = mock_state_service

                with patch("main.create_llm_provider"):
                    with patch("main.ParallelDocumentProcessingService"):
                        with patch("main.DocumentDiscoveryService"):
                            evaluator = BatchEvaluator(test_batch_configuration)

                            # Test document retry (just returns True, doesn't actually process)
                            result = evaluator.retry_document("batch_001", "doc_001")
                            assert result is True

    def test_batch_not_found(self, test_batch_configuration):
        """Test handling of non-existent batch."""
        with patch("main.logging.getLogger"):
            with patch("main.BatchStatePersistenceService") as mock_state_class:
                mock_state_service = Mock()
                # 修正: active_dirのモックを適切に設定
                from pathlib import Path

                mock_active_dir = Mock(spec=Path)
                mock_active_dir.__truediv__ = Mock(
                    return_value=Path("/tmp/test_batch_documents.json")
                )
                mock_state_service.active_dir = mock_active_dir
                mock_state_service.load_batch_state.return_value = None
                mock_state_class.return_value = mock_state_service

                with patch("main.create_llm_provider"):
                    with patch("main.ParallelDocumentProcessingService"):
                        with patch("main.DocumentDiscoveryService"):
                            evaluator = BatchEvaluator(test_batch_configuration)

                            # Test non-existent batch
                            batch = evaluator.get_batch("non_existent_batch")
                            assert batch is None

                            progress = evaluator.get_batch_progress(
                                "non_existent_batch"
                            )
                            assert progress is None

                            results = evaluator.get_batch_results("non_existent_batch")
                            assert results is None
