"""
Unit tests for BatchEvaluator orchestration class.
Tests lines 1990-2485: BatchEvaluator orchestration class
"""

import json
import pytest
from unittest.mock import Mock, patch, MagicMock
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
    DocumentResult,
    DisclosureEvaluationResult,
)


class TestBatchEvaluator:
    """Test BatchEvaluator class."""

    def test_batch_evaluator_init(self, test_batch_configuration, mock_openai_client):
        """Test BatchEvaluator initialization."""
        with patch("main.logging.getLogger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            with patch("main.DocumentDiscoveryService") as mock_discovery_class:
                with patch("main.BatchStatePersistenceService") as mock_state_class:
                    with patch("main.create_llm_provider") as mock_create_provider:
                        with patch(
                            "main.ParallelDocumentProcessingService"
                        ) as mock_processing_class:
                            mock_create_provider.return_value = Mock()

                            evaluator = BatchEvaluator(test_batch_configuration)

                            assert evaluator.config == test_batch_configuration
                            assert evaluator.logger == mock_logger
                            assert evaluator.discovery_service is not None
                            assert evaluator.state_service is not None
                            assert evaluator.llm_provider is not None
                            assert evaluator.processing_service is not None

                            mock_discovery_class.assert_called_once_with(
                                test_batch_configuration
                            )
                            mock_state_class.assert_called_once_with(
                                test_batch_configuration
                            )
                            mock_create_provider.assert_called_once()
                            mock_processing_class.assert_called_once()

    def test_batch_evaluator_init_default_config(self, mock_openai_client):
        """Test BatchEvaluator initialization with default configuration."""
        with patch("main.logging.getLogger"):
            with patch("main.DocumentDiscoveryService"):
                with patch("main.BatchStatePersistenceService"):
                    with patch("main.create_llm_provider"):
                        with patch("main.ParallelDocumentProcessingService"):
                            evaluator = BatchEvaluator()  # No config provided

                            assert isinstance(evaluator.config, BatchConfiguration)
                            assert (
                                evaluator.config.max_concurrent_workers == 5
                            )  # Default value

    def test_create_batch_success(
        self, test_batch_configuration, sample_document_input
    ):
        """Test successful batch creation."""
        documents = [sample_document_input]

        with patch("main.logging.getLogger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            with patch("main.DocumentDiscoveryService"):
                with patch("main.BatchStatePersistenceService") as mock_state_class:
                    mock_state_service = Mock()
                    mock_state_class.return_value = mock_state_service

                    with patch("main.create_llm_provider"):
                        with patch("main.ParallelDocumentProcessingService"):
                            with patch.object(
                                BatchEvaluator, "_store_batch_documents"
                            ) as mock_store:
                                evaluator = BatchEvaluator(test_batch_configuration)
                                batch_id = evaluator.create_batch(documents)

                                assert batch_id.startswith("batch_")
                                assert len(batch_id) > 6  # Should have timestamp

                                # Verify batch state was saved
                                mock_state_service.save_batch_state.assert_called_once()

                                # Verify documents were stored
                                mock_store.assert_called_once_with(batch_id, documents)

                                mock_logger.info.assert_called()

    def test_create_batch_exception(
        self, test_batch_configuration, sample_document_input
    ):
        """Test batch creation with exception."""
        documents = [sample_document_input]

        with patch("main.logging.getLogger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            with patch("main.DocumentDiscoveryService"):
                with patch("main.BatchStatePersistenceService") as mock_state_class:
                    mock_state_service = Mock()
                    mock_state_service.save_batch_state.side_effect = Exception(
                        "Save failed"
                    )
                    mock_state_class.return_value = mock_state_service

                    with patch("main.create_llm_provider"):
                        with patch("main.ParallelDocumentProcessingService"):
                            evaluator = BatchEvaluator(test_batch_configuration)

                            with pytest.raises(Exception, match="Save failed"):
                                evaluator.create_batch(documents)

                            mock_logger.error.assert_called()

    def test_store_batch_documents_success(
        self, test_batch_configuration, sample_document_input
    ):
        """Test successful batch document storage."""
        documents = [sample_document_input]
        batch_id = "test_batch_001"

        with patch("main.logging.getLogger"):
            with patch("main.DocumentDiscoveryService"):
                with patch("main.BatchStatePersistenceService") as mock_state_class:
                    mock_state_service = Mock()
                    mock_state_service.active_dir = Path("/test/active")
                    mock_state_class.return_value = mock_state_service

                    with patch("main.create_llm_provider"):
                        with patch("main.ParallelDocumentProcessingService"):
                            with patch("builtins.open", mock_open()) as mock_file:
                                evaluator = BatchEvaluator(test_batch_configuration)
                                evaluator._store_batch_documents(batch_id, documents)

                                # Verify file was opened for writing
                                mock_file.assert_called_once()
                                call_args = mock_file.call_args
                                assert "w" in call_args[1]["mode"]
                                assert "utf-8" in call_args[1]["encoding"]

                                # Verify JSON was written
                                written_data = mock_file().write.call_args[0][0]
                                json_data = json.loads(written_data)
                                assert len(json_data) == 1
                                assert (
                                    json_data[0]["file_path"]
                                    == sample_document_input.file_path
                                )
                                assert (
                                    json_data[0]["file_name"]
                                    == sample_document_input.file_name
                                )
                                assert (
                                    json_data[0]["context"]
                                    == sample_document_input.context
                                )
                                assert (
                                    json_data[0]["output_text"]
                                    == sample_document_input.output_text
                                )

    def test_store_batch_documents_exception(
        self, test_batch_configuration, sample_document_input
    ):
        """Test batch document storage with exception."""
        documents = [sample_document_input]
        batch_id = "test_batch_001"

        with patch("main.logging.getLogger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            with patch("main.DocumentDiscoveryService"):
                with patch("main.BatchStatePersistenceService") as mock_state_class:
                    mock_state_service = Mock()
                    mock_state_service.active_dir = Path("/test/active")
                    mock_state_class.return_value = mock_state_service

                    with patch("main.create_llm_provider"):
                        with patch("main.ParallelDocumentProcessingService"):
                            with patch(
                                "builtins.open",
                                side_effect=IOError("Permission denied"),
                            ):
                                evaluator = BatchEvaluator(test_batch_configuration)

                                with pytest.raises(IOError, match="Permission denied"):
                                    evaluator._store_batch_documents(
                                        batch_id, documents
                                    )

                                mock_logger.error.assert_called()

    def test_convert_document_inputs_to_batch_documents_success(
        self, test_batch_configuration, sample_document_input
    ):
        """Test successful conversion of DocumentInput to BatchDocument."""
        documents = [sample_document_input]
        batch_id = "test_batch_001"

        with patch("main.logging.getLogger"):
            with patch("main.DocumentDiscoveryService"):
                with patch("main.BatchStatePersistenceService"):
                    with patch("main.create_llm_provider"):
                        with patch("main.ParallelDocumentProcessingService"):
                            with patch("pathlib.Path.exists", return_value=True):
                                with patch("pathlib.Path.stat") as mock_stat:
                                    mock_stat.return_value.st_size = 1024

                                    with patch(
                                        "main.mimetypes.guess_type",
                                        return_value=("text/plain", None),
                                    ):
                                        evaluator = BatchEvaluator(
                                            test_batch_configuration
                                        )
                                        batch_documents = evaluator._convert_document_inputs_to_batch_documents(
                                            documents, batch_id
                                        )

                                        assert len(batch_documents) == 1
                                        assert isinstance(
                                            batch_documents[0], BatchDocument
                                        )
                                        assert (
                                            batch_documents[0].document_id
                                            == "doc_test_batch_001_0000"
                                        )
                                        assert batch_documents[0].batch_id == batch_id
                                        assert (
                                            batch_documents[0].file_path
                                            == sample_document_input.file_path
                                        )
                                        assert (
                                            batch_documents[0].file_name
                                            == sample_document_input.file_name
                                        )
                                        assert batch_documents[0].file_size == 1024
                                        assert (
                                            batch_documents[0].mime_type == "text/plain"
                                        )
                                        assert (
                                            batch_documents[0].status
                                            == DocumentStatus.PENDING
                                        )
                                        assert (
                                            batch_documents[0].context
                                            == sample_document_input.context
                                        )
                                        assert (
                                            batch_documents[0].output_text
                                            == sample_document_input.output_text
                                        )

    def test_convert_document_inputs_to_batch_documents_file_not_found(
        self, test_batch_configuration, sample_document_input
    ):
        """Test conversion when file doesn't exist."""
        documents = [sample_document_input]
        batch_id = "test_batch_001"

        with patch("main.logging.getLogger"):
            with patch("main.DocumentDiscoveryService"):
                with patch("main.BatchStatePersistenceService"):
                    with patch("main.create_llm_provider"):
                        with patch("main.ParallelDocumentProcessingService"):
                            with patch("pathlib.Path.exists", return_value=False):
                                with patch(
                                    "main.mimetypes.guess_type",
                                    return_value=("text/plain", None),
                                ):
                                    evaluator = BatchEvaluator(test_batch_configuration)
                                    batch_documents = evaluator._convert_document_inputs_to_batch_documents(
                                        documents, batch_id
                                    )

                                    assert len(batch_documents) == 1
                                    assert (
                                        batch_documents[0].file_size == 0
                                    )  # Should be 0 for non-existent file

    def test_convert_document_inputs_to_batch_documents_exception(
        self, test_batch_configuration, sample_document_input
    ):
        """Test conversion with exception."""
        documents = [sample_document_input]
        batch_id = "test_batch_001"

        with patch("main.logging.getLogger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            with patch("main.DocumentDiscoveryService"):
                with patch("main.BatchStatePersistenceService"):
                    with patch("main.create_llm_provider"):
                        with patch("main.ParallelDocumentProcessingService"):
                            with patch(
                                "pathlib.Path.exists",
                                side_effect=OSError("Permission denied"),
                            ):
                                evaluator = BatchEvaluator(test_batch_configuration)

                                with pytest.raises(OSError, match="Permission denied"):
                                    evaluator._convert_document_inputs_to_batch_documents(
                                        documents, batch_id
                                    )

                                mock_logger.error.assert_called()

    def test_create_batch_from_folder_success(
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

                with patch("main.BatchStatePersistenceService"):
                    with patch("main.create_llm_provider"):
                        with patch("main.ParallelDocumentProcessingService"):
                            with patch.object(
                                BatchEvaluator,
                                "create_batch",
                                return_value="test_batch_001",
                            ):
                                evaluator = BatchEvaluator(test_batch_configuration)
                                batch_id = evaluator.create_batch_from_folder(
                                    sample_documents_dir
                                )

                                assert batch_id == "test_batch_001"
                                mock_discovery_service.discover_documents_from_folder.assert_called_once()

    def test_create_batch_from_folder_empty_folder(
        self, test_batch_configuration, sample_documents_dir
    ):
        """Test batch creation from empty folder."""
        with patch("main.logging.getLogger"):
            with patch("main.DocumentDiscoveryService") as mock_discovery_class:
                mock_discovery_service = Mock()
                mock_discovery_service.discover_documents_from_folder.return_value = []
                mock_discovery_class.return_value = mock_discovery_service

                with patch("main.BatchStatePersistenceService"):
                    with patch("main.create_llm_provider"):
                        with patch("main.ParallelDocumentProcessingService"):
                            evaluator = BatchEvaluator(test_batch_configuration)

                            with pytest.raises(
                                ValueError,
                                match=f"No documents found in {sample_documents_dir}",
                            ):
                                evaluator.create_batch_from_folder(sample_documents_dir)

    def test_create_batch_from_folder_exception(
        self, test_batch_configuration, sample_documents_dir
    ):
        """Test batch creation from folder with exception."""
        with patch("main.logging.getLogger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            with patch("main.DocumentDiscoveryService") as mock_discovery_class:
                mock_discovery_service = Mock()
                mock_discovery_service.discover_documents_from_folder.side_effect = (
                    Exception("Discovery failed")
                )
                mock_discovery_class.return_value = mock_discovery_service

                with patch("main.BatchStatePersistenceService"):
                    with patch("main.create_llm_provider"):
                        with patch("main.ParallelDocumentProcessingService"):
                            evaluator = BatchEvaluator(test_batch_configuration)

                            with pytest.raises(Exception, match="Discovery failed"):
                                evaluator.create_batch_from_folder(sample_documents_dir)

                            mock_logger.error.assert_called()

    def test_get_batch_success(self, test_batch_configuration, sample_batch_evaluation):
        """Test successful batch retrieval."""
        with patch("main.logging.getLogger"):
            with patch("main.DocumentDiscoveryService"):
                with patch("main.BatchStatePersistenceService") as mock_state_class:
                    mock_state_service = Mock()
                    mock_state_service.load_batch_state.return_value = (
                        sample_batch_evaluation
                    )
                    mock_state_class.return_value = mock_state_service

                    with patch("main.create_llm_provider"):
                        with patch("main.ParallelDocumentProcessingService"):
                            evaluator = BatchEvaluator(test_batch_configuration)
                            result = evaluator.get_batch("test_batch_001")

                            assert result == sample_batch_evaluation
                            mock_state_service.load_batch_state.assert_called_once_with(
                                "test_batch_001"
                            )

    def test_get_batch_not_found(self, test_batch_configuration):
        """Test batch retrieval when batch not found."""
        with patch("main.logging.getLogger"):
            with patch("main.DocumentDiscoveryService"):
                with patch("main.BatchStatePersistenceService") as mock_state_class:
                    mock_state_service = Mock()
                    mock_state_service.load_batch_state.return_value = None
                    mock_state_class.return_value = mock_state_service

                    with patch("main.create_llm_provider"):
                        with patch("main.ParallelDocumentProcessingService"):
                            evaluator = BatchEvaluator(test_batch_configuration)
                            result = evaluator.get_batch("nonexistent_batch")

                            assert result is None

    def test_get_batch_progress_success(
        self, test_batch_configuration, sample_batch_evaluation
    ):
        """Test successful batch progress retrieval."""
        with patch("main.logging.getLogger"):
            with patch("main.DocumentDiscoveryService"):
                with patch("main.BatchStatePersistenceService") as mock_state_class:
                    mock_state_service = Mock()
                    mock_state_service.load_batch_state.return_value = (
                        sample_batch_evaluation
                    )
                    mock_state_class.return_value = mock_state_service

                    with patch("main.create_llm_provider"):
                        with patch("main.ParallelDocumentProcessingService"):
                            with patch("main.datetime") as mock_datetime:
                                mock_datetime.now.return_value = datetime(
                                    2025, 1, 1, 12, 0, 0
                                )

                                evaluator = BatchEvaluator(test_batch_configuration)
                                result = evaluator.get_batch_progress("test_batch_001")

                                assert isinstance(result, BatchProgress)
                                assert result.batch_id == "test_batch_001"
                                assert (
                                    result.total_documents
                                    == sample_batch_evaluation.total_documents
                                )
                                assert (
                                    result.processed_documents
                                    == sample_batch_evaluation.processed_documents
                                )
                                assert (
                                    result.successful_documents
                                    == sample_batch_evaluation.successful_documents
                                )
                                assert (
                                    result.failed_documents
                                    == sample_batch_evaluation.failed_documents
                                )

    def test_get_batch_progress_not_found(self, test_batch_configuration):
        """Test batch progress retrieval when batch not found."""
        with patch("main.logging.getLogger"):
            with patch("main.DocumentDiscoveryService"):
                with patch("main.BatchStatePersistenceService") as mock_state_class:
                    mock_state_service = Mock()
                    mock_state_service.load_batch_state.return_value = None
                    mock_state_class.return_value = mock_state_service

                    with patch("main.create_llm_provider"):
                        with patch("main.ParallelDocumentProcessingService"):
                            evaluator = BatchEvaluator(test_batch_configuration)
                            result = evaluator.get_batch_progress("nonexistent_batch")

                            assert result is None

    def test_get_batch_results_success(
        self, test_batch_configuration, sample_batch_evaluation
    ):
        """Test successful batch results retrieval."""
        with patch("main.logging.getLogger"):
            with patch("main.DocumentDiscoveryService"):
                with patch("main.BatchStatePersistenceService") as mock_state_class:
                    mock_state_service = Mock()
                    mock_state_service.load_batch_state.return_value = (
                        sample_batch_evaluation
                    )
                    mock_state_class.return_value = mock_state_service

                    with patch("main.create_llm_provider"):
                        with patch("main.ParallelDocumentProcessingService"):
                            with patch("main.datetime") as mock_datetime:
                                mock_datetime.now.return_value = datetime(
                                    2025, 1, 1, 12, 0, 0
                                )

                                evaluator = BatchEvaluator(test_batch_configuration)
                                result = evaluator.get_batch_results("test_batch_001")

                                assert isinstance(result, BatchResult)
                                assert result.batch_id == "test_batch_001"
                                assert (
                                    result.total_documents
                                    == sample_batch_evaluation.total_documents
                                )
                                assert (
                                    result.successful_evaluations
                                    == sample_batch_evaluation.successful_documents
                                )
                                assert (
                                    result.failed_evaluations
                                    == sample_batch_evaluation.failed_documents
                                )

    def test_get_batch_results_not_found(self, test_batch_configuration):
        """Test batch results retrieval when batch not found."""
        with patch("main.logging.getLogger"):
            with patch("main.DocumentDiscoveryService"):
                with patch("main.BatchStatePersistenceService") as mock_state_class:
                    mock_state_service = Mock()
                    mock_state_service.load_batch_state.return_value = None
                    mock_state_class.return_value = mock_state_service

                    with patch("main.create_llm_provider"):
                        with patch("main.ParallelDocumentProcessingService"):
                            evaluator = BatchEvaluator(test_batch_configuration)
                            result = evaluator.get_batch_results("nonexistent_batch")

                            assert result is None

    def test_get_batch_results_incomplete_batch(
        self, test_batch_configuration, sample_batch_evaluation
    ):
        """Test batch results retrieval for incomplete batch."""
        # Set batch status to PROCESSING (not completed)
        sample_batch_evaluation.status = BatchStatus.PROCESSING

        with patch("main.logging.getLogger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            with patch("main.DocumentDiscoveryService"):
                with patch("main.BatchStatePersistenceService") as mock_state_class:
                    mock_state_service = Mock()
                    mock_state_service.load_batch_state.return_value = (
                        sample_batch_evaluation
                    )
                    mock_state_class.return_value = mock_state_service

                    with patch("main.create_llm_provider"):
                        with patch("main.ParallelDocumentProcessingService"):
                            evaluator = BatchEvaluator(test_batch_configuration)
                            result = evaluator.get_batch_results("test_batch_001")

                            assert result is None
                            mock_logger.warning.assert_called()

    def test_get_batch_results_exception(
        self, test_batch_configuration, sample_batch_evaluation
    ):
        """Test batch results retrieval with exception."""
        with patch("main.logging.getLogger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            with patch("main.DocumentDiscoveryService"):
                with patch("main.BatchStatePersistenceService") as mock_state_class:
                    mock_state_service = Mock()
                    mock_state_service.load_batch_state.return_value = (
                        sample_batch_evaluation
                    )
                    mock_state_class.return_value = mock_state_service

                    with patch("main.create_llm_provider"):
                        with patch("main.ParallelDocumentProcessingService"):
                            with patch(
                                "main.datetime", side_effect=Exception("Datetime error")
                            ):
                                evaluator = BatchEvaluator(test_batch_configuration)
                                result = evaluator.get_batch_results("test_batch_001")

                                assert result is None
                                mock_logger.error.assert_called()

    def test_start_batch_success(
        self, test_batch_configuration, sample_batch_evaluation
    ):
        """Test successful batch start."""
        with patch("main.logging.getLogger"):
            with patch("main.DocumentDiscoveryService"):
                with patch("main.BatchStatePersistenceService") as mock_state_class:
                    mock_state_service = Mock()
                    mock_state_service.load_batch_state.return_value = (
                        sample_batch_evaluation
                    )
                    mock_state_class.return_value = mock_state_service

                    with patch("main.create_llm_provider"):
                        with patch(
                            "main.ParallelDocumentProcessingService"
                        ) as mock_processing_class:
                            mock_processing_service = Mock()
                            mock_processing_class.return_value = mock_processing_service

                            with patch(
                                "builtins.open", mock_open(read_data=json.dumps([]))
                            ):
                                with patch("pathlib.Path.exists", return_value=True):
                                    with patch.object(
                                        BatchEvaluator,
                                        "_convert_document_inputs_to_batch_documents",
                                    ) as mock_convert:
                                        mock_convert.return_value = []

                                        evaluator = BatchEvaluator(
                                            test_batch_configuration
                                        )
                                        evaluator.start_batch("test_batch_001")

                                        # Verify batch status was updated
                                        mock_state_service.save_batch_state.assert_called()
                                        mock_processing_service.process_documents_parallel.assert_called_once()

    def test_start_batch_not_found(self, test_batch_configuration):
        """Test batch start when batch not found."""
        with patch("main.logging.getLogger"):
            with patch("main.DocumentDiscoveryService"):
                with patch("main.BatchStatePersistenceService") as mock_state_class:
                    mock_state_service = Mock()
                    mock_state_service.load_batch_state.return_value = None
                    mock_state_class.return_value = mock_state_service

                    with patch("main.create_llm_provider"):
                        with patch("main.ParallelDocumentProcessingService"):
                            evaluator = BatchEvaluator(test_batch_configuration)

                            with pytest.raises(
                                ValueError, match="Batch test_batch_001 not found"
                            ):
                                evaluator.start_batch("test_batch_001")

    def test_start_batch_exception(
        self, test_batch_configuration, sample_batch_evaluation
    ):
        """Test batch start with exception."""
        with patch("main.logging.getLogger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            with patch("main.DocumentDiscoveryService"):
                with patch("main.BatchStatePersistenceService") as mock_state_class:
                    mock_state_service = Mock()
                    mock_state_service.load_batch_state.return_value = (
                        sample_batch_evaluation
                    )
                    mock_state_class.return_value = mock_state_service

                    with patch("main.create_llm_provider"):
                        with patch("main.ParallelDocumentProcessingService"):
                            with patch(
                                "builtins.open", side_effect=IOError("File not found")
                            ):
                                evaluator = BatchEvaluator(test_batch_configuration)

                                with pytest.raises(IOError, match="File not found"):
                                    evaluator.start_batch("test_batch_001")

                                mock_logger.error.assert_called()

    def test_resume_batch_success(
        self, test_batch_configuration, sample_batch_evaluation
    ):
        """Test successful batch resumption."""
        with patch("main.logging.getLogger"):
            with patch("main.DocumentDiscoveryService"):
                with patch("main.BatchStatePersistenceService") as mock_state_class:
                    mock_state_service = Mock()
                    mock_state_service.load_batch_state.return_value = (
                        sample_batch_evaluation
                    )
                    mock_state_class.return_value = mock_state_service

                    with patch("main.create_llm_provider"):
                        with patch(
                            "main.ParallelDocumentProcessingService"
                        ) as mock_processing_class:
                            mock_processing_service = Mock()
                            mock_processing_class.return_value = mock_processing_service

                            with patch(
                                "builtins.open", mock_open(read_data=json.dumps([]))
                            ):
                                with patch("pathlib.Path.exists", return_value=True):
                                    with patch.object(
                                        BatchEvaluator,
                                        "_convert_document_inputs_to_batch_documents",
                                    ) as mock_convert:
                                        mock_convert.return_value = []

                                        evaluator = BatchEvaluator(
                                            test_batch_configuration
                                        )
                                        evaluator.resume_batch("test_batch_001")

                                        # Verify batch was resumed
                                        mock_state_service.save_batch_state.assert_called()
                                        mock_processing_service.process_documents_parallel.assert_called_once()

    def test_resume_batch_not_found(self, test_batch_configuration):
        """Test batch resumption when batch not found."""
        with patch("main.logging.getLogger"):
            with patch("main.DocumentDiscoveryService"):
                with patch("main.BatchStatePersistenceService") as mock_state_class:
                    mock_state_service = Mock()
                    mock_state_service.load_batch_state.return_value = None
                    mock_state_class.return_value = mock_state_service

                    with patch("main.create_llm_provider"):
                        with patch("main.ParallelDocumentProcessingService"):
                            evaluator = BatchEvaluator(test_batch_configuration)

                            with pytest.raises(
                                ValueError, match="Batch test_batch_001 not found"
                            ):
                                evaluator.resume_batch("test_batch_001")

    def test_retry_failed_documents_success(
        self, test_batch_configuration, sample_batch_evaluation
    ):
        """Test successful retry of failed documents."""
        with patch("main.logging.getLogger"):
            with patch("main.DocumentDiscoveryService"):
                with patch("main.BatchStatePersistenceService") as mock_state_class:
                    mock_state_service = Mock()
                    mock_state_service.load_batch_state.return_value = (
                        sample_batch_evaluation
                    )
                    mock_state_class.return_value = mock_state_service

                    with patch("main.create_llm_provider"):
                        with patch(
                            "main.ParallelDocumentProcessingService"
                        ) as mock_processing_class:
                            mock_processing_service = Mock()
                            mock_processing_class.return_value = mock_processing_service

                            with patch(
                                "builtins.open", mock_open(read_data=json.dumps([]))
                            ):
                                with patch("pathlib.Path.exists", return_value=True):
                                    with patch.object(
                                        BatchEvaluator,
                                        "_convert_document_inputs_to_batch_documents",
                                    ) as mock_convert:
                                        mock_convert.return_value = []

                                        evaluator = BatchEvaluator(
                                            test_batch_configuration
                                        )
                                        evaluator.retry_failed_documents(
                                            "test_batch_001", ["doc_001", "doc_002"]
                                        )

                                        # Verify batch state was updated
                                        mock_state_service.save_batch_state.assert_called()
                                        mock_processing_service.process_documents_parallel.assert_called_once()

    def test_retry_failed_documents_not_found(self, test_batch_configuration):
        """Test retry failed documents when batch not found."""
        with patch("main.logging.getLogger"):
            with patch("main.DocumentDiscoveryService"):
                with patch("main.BatchStatePersistenceService") as mock_state_class:
                    mock_state_service = Mock()
                    mock_state_service.load_batch_state.return_value = None
                    mock_state_class.return_value = mock_state_service

                    with patch("main.create_llm_provider"):
                        with patch("main.ParallelDocumentProcessingService"):
                            evaluator = BatchEvaluator(test_batch_configuration)

                            with pytest.raises(
                                ValueError, match="Batch test_batch_001 not found"
                            ):
                                evaluator.retry_failed_documents(
                                    "test_batch_001", ["doc_001"]
                                )

    def test_retry_failed_documents_exception(
        self, test_batch_configuration, sample_batch_evaluation
    ):
        """Test retry failed documents with exception."""
        with patch("main.logging.getLogger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            with patch("main.DocumentDiscoveryService"):
                with patch("main.BatchStatePersistenceService") as mock_state_class:
                    mock_state_service = Mock()
                    mock_state_service.load_batch_state.return_value = (
                        sample_batch_evaluation
                    )
                    mock_state_class.return_value = mock_state_service

                    with patch("main.create_llm_provider"):
                        with patch("main.ParallelDocumentProcessingService"):
                            with patch(
                                "builtins.open", side_effect=IOError("File not found")
                            ):
                                evaluator = BatchEvaluator(test_batch_configuration)

                                with pytest.raises(IOError, match="File not found"):
                                    evaluator.retry_failed_documents(
                                        "test_batch_001", ["doc_001"]
                                    )

                                mock_logger.error.assert_called()

    def test_batch_evaluator_with_custom_config(self, mock_openai_client):
        """Test BatchEvaluator with custom configuration."""
        custom_config = BatchConfiguration(
            max_concurrent_workers=10,
            max_retry_attempts=5,
            timeout_seconds=600,
            file_size_limit=100 * 1024 * 1024,
            output_formats=["json", "csv"],
        )

        with patch("main.logging.getLogger"):
            with patch("main.DocumentDiscoveryService") as mock_discovery_class:
                with patch("main.BatchStatePersistenceService") as mock_state_class:
                    with patch("main.create_llm_provider"):
                        with patch(
                            "main.ParallelDocumentProcessingService"
                        ) as mock_processing_class:
                            evaluator = BatchEvaluator(custom_config)

                            # Verify services were initialized with custom config
                            mock_discovery_class.assert_called_once_with(custom_config)
                            mock_state_class.assert_called_once_with(custom_config)
                            mock_processing_class.assert_called_once_with(
                                custom_config, evaluator.llm_provider
                            )

    def test_batch_evaluator_unicode_handling(
        self, test_batch_configuration, mock_openai_client
    ):
        """Test BatchEvaluator with Unicode content."""
        unicode_document = DocumentInput(
            file_path="/test/unicode_doc.txt",
            file_name="unicode_doc.txt",
            context="テストコンテキスト",
            output_text="テスト出力",
        )

        with patch("main.logging.getLogger"):
            with patch("main.DocumentDiscoveryService"):
                with patch("main.BatchStatePersistenceService") as mock_state_class:
                    mock_state_service = Mock()
                    mock_state_class.return_value = mock_state_service

                    with patch("main.create_llm_provider"):
                        with patch("main.ParallelDocumentProcessingService"):
                            with patch("builtins.open", mock_open()) as mock_file:
                                evaluator = BatchEvaluator(test_batch_configuration)
                                evaluator._store_batch_documents(
                                    "test_batch_001", [unicode_document]
                                )

                                # Verify Unicode content was stored correctly
                                written_data = mock_file().write.call_args[0][0]
                                json_data = json.loads(written_data)
                                assert json_data[0]["context"] == "テストコンテキスト"
                                assert json_data[0]["output_text"] == "テスト出力"
