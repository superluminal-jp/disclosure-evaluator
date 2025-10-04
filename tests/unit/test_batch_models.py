"""
Unit tests for batch processing models.

These tests validate the data models and their validation rules.
"""

import pytest
from datetime import datetime, timedelta
from pathlib import Path
from evaluator import (
    BatchEvaluation,
    BatchDocument,
    BatchResult,
    BatchProgress,
    BatchConfiguration,
    BatchSummaryStatistics,
    DocumentResult,
    DocumentError,
    DocumentInput,
    BatchStatus,
    DocumentStatus,
    ProcessingPhase,
)


class TestBatchStatus:
    """Test BatchStatus enum"""

    def test_batch_status_values(self):
        """Test batch status enum values"""
        assert BatchStatus.PENDING.value == "pending"
        assert BatchStatus.PROCESSING.value == "processing"
        assert BatchStatus.COMPLETED.value == "completed"
        assert BatchStatus.FAILED.value == "failed"
        assert BatchStatus.PARTIALLY_FAILED.value == "partially_failed"


class TestDocumentStatus:
    """Test DocumentStatus enum"""

    def test_document_status_values(self):
        """Test document status enum values"""
        assert DocumentStatus.PENDING.value == "pending"
        assert DocumentStatus.PROCESSING.value == "processing"
        assert DocumentStatus.COMPLETED.value == "completed"
        assert DocumentStatus.FAILED.value == "failed"


class TestProcessingPhase:
    """Test ProcessingPhase enum"""

    def test_processing_phase_values(self):
        """Test processing phase enum values"""
        assert ProcessingPhase.INITIALIZING.value == "initializing"
        assert ProcessingPhase.DISCOVERING.value == "discovering"
        assert ProcessingPhase.PROCESSING.value == "processing"
        assert ProcessingPhase.AGGREGATING.value == "aggregating"
        assert ProcessingPhase.COMPLETED.value == "completed"


class TestBatchConfiguration:
    """Test BatchConfiguration model"""

    def test_default_configuration(self):
        """Test default configuration values"""
        config = BatchConfiguration()

        assert config.max_concurrent_workers == 5
        assert config.max_retry_attempts == 3
        assert config.timeout_seconds == 300
        assert config.progress_update_interval == 10
        assert config.enable_resumption is True
        assert config.output_formats == ["json", "summary"]
        assert config.file_size_limit == 50 * 1024 * 1024
        assert config.memory_limit_mb == 2048
        assert config.api_rate_limit_delay == 0.1
        assert config.retry_delay_seconds == 30
        assert config.exponential_backoff is True

    def test_custom_configuration(self):
        """Test custom configuration values"""
        config = BatchConfiguration(
            max_concurrent_workers=10,
            max_retry_attempts=5,
            timeout_seconds=600,
            progress_update_interval=5,
            enable_resumption=False,
            output_formats=["json", "csv"],
            file_size_limit=100 * 1024 * 1024,
            memory_limit_mb=4096,
            api_rate_limit_delay=0.5,
            retry_delay_seconds=60,
            exponential_backoff=False,
        )

        assert config.max_concurrent_workers == 10
        assert config.max_retry_attempts == 5
        assert config.timeout_seconds == 600
        assert config.progress_update_interval == 5
        assert config.enable_resumption is False
        assert config.output_formats == ["json", "csv"]
        assert config.file_size_limit == 100 * 1024 * 1024
        assert config.memory_limit_mb == 4096
        assert config.api_rate_limit_delay == 0.5
        assert config.retry_delay_seconds == 60
        assert config.exponential_backoff is False

    def test_configuration_validation(self):
        """Test configuration validation rules"""
        # Test valid values
        config = BatchConfiguration(
            max_concurrent_workers=1,
            max_retry_attempts=0,
            timeout_seconds=30,
            progress_update_interval=1,
            file_size_limit=1024,
            memory_limit_mb=256,
            api_rate_limit_delay=0.0,
            retry_delay_seconds=1,
        )
        assert config.max_concurrent_workers == 1
        assert config.max_retry_attempts == 0
        assert config.timeout_seconds == 30
        assert config.progress_update_interval == 1
        assert config.file_size_limit == 1024
        assert config.memory_limit_mb == 256
        assert config.api_rate_limit_delay == 0.0
        assert config.retry_delay_seconds == 1

        # Test invalid values
        with pytest.raises(ValueError):
            BatchConfiguration(max_concurrent_workers=0)

        with pytest.raises(ValueError):
            BatchConfiguration(max_concurrent_workers=21)

        with pytest.raises(ValueError):
            BatchConfiguration(max_retry_attempts=-1)

        with pytest.raises(ValueError):
            BatchConfiguration(max_retry_attempts=11)

        with pytest.raises(ValueError):
            BatchConfiguration(timeout_seconds=29)

        with pytest.raises(ValueError):
            BatchConfiguration(timeout_seconds=3601)

        with pytest.raises(ValueError):
            BatchConfiguration(progress_update_interval=0)

        with pytest.raises(ValueError):
            BatchConfiguration(progress_update_interval=101)

        with pytest.raises(ValueError):
            BatchConfiguration(file_size_limit=1023)

        with pytest.raises(ValueError):
            BatchConfiguration(memory_limit_mb=255)

        with pytest.raises(ValueError):
            BatchConfiguration(api_rate_limit_delay=-0.1)

        with pytest.raises(ValueError):
            BatchConfiguration(api_rate_limit_delay=10.1)

        with pytest.raises(ValueError):
            BatchConfiguration(retry_delay_seconds=0)

        with pytest.raises(ValueError):
            BatchConfiguration(retry_delay_seconds=301)


class TestBatchEvaluation:
    """Test BatchEvaluation model"""

    def test_batch_evaluation_creation(self):
        """Test batch evaluation creation"""
        now = datetime.now()
        config = BatchConfiguration()

        batch = BatchEvaluation(
            batch_id="test_batch_001",
            created_at=now,
            status=BatchStatus.PENDING,
            total_documents=10,
            correlation_id="batch_test_batch_001",
            configuration=config,
        )

        assert batch.batch_id == "test_batch_001"
        assert batch.created_at == now
        assert batch.status == BatchStatus.PENDING
        assert batch.total_documents == 10
        assert batch.processed_documents == 0
        assert batch.successful_documents == 0
        assert batch.failed_documents == 0
        assert batch.processing_started_at is None
        assert batch.processing_completed_at is None
        assert batch.error_summary is None
        assert batch.correlation_id == "batch_test_batch_001"
        assert batch.configuration == config

    def test_batch_evaluation_validation(self):
        """Test batch evaluation validation"""
        config = BatchConfiguration()

        # Test valid batch
        batch = BatchEvaluation(
            batch_id="test_batch_001",
            created_at=datetime.now(),
            status=BatchStatus.PENDING,
            total_documents=10,
            correlation_id="batch_test_batch_001",
            configuration=config,
        )
        assert batch.total_documents == 10

        # Test invalid total_documents
        with pytest.raises(ValueError):
            BatchEvaluation(
                batch_id="test_batch_001",
                created_at=datetime.now(),
                status=BatchStatus.PENDING,
                total_documents=-1,
                correlation_id="batch_test_batch_001",
                configuration=config,
            )

        # Test invalid processed_documents
        with pytest.raises(ValueError):
            BatchEvaluation(
                batch_id="test_batch_001",
                created_at=datetime.now(),
                status=BatchStatus.PENDING,
                total_documents=10,
                processed_documents=-1,
                correlation_id="batch_test_batch_001",
                configuration=config,
            )


class TestBatchDocument:
    """Test BatchDocument model"""

    def test_batch_document_creation(self):
        """Test batch document creation"""
        doc = BatchDocument(
            document_id="doc_001",
            batch_id="batch_001",
            file_path="/path/to/document.txt",
            file_name="document.txt",
            file_size=1024,
            mime_type="text/plain",
            status=DocumentStatus.PENDING,
            correlation_id="doc_doc_001",
        )

        assert doc.document_id == "doc_001"
        assert doc.batch_id == "batch_001"
        assert doc.file_path == "/path/to/document.txt"
        assert doc.file_name == "document.txt"
        assert doc.file_size == 1024
        assert doc.mime_type == "text/plain"
        assert doc.status == DocumentStatus.PENDING
        assert doc.processing_started_at is None
        assert doc.processing_completed_at is None
        assert doc.evaluation_result is None
        assert doc.error_message is None
        assert doc.retry_count == 0
        assert doc.correlation_id == "doc_doc_001"

    def test_batch_document_validation(self):
        """Test batch document validation"""
        # Test valid document
        doc = BatchDocument(
            document_id="doc_001",
            batch_id="batch_001",
            file_path="/path/to/document.txt",
            file_name="document.txt",
            file_size=1024,
            mime_type="text/plain",
            status=DocumentStatus.PENDING,
            correlation_id="doc_doc_001",
        )
        assert doc.file_size == 1024

        # Test invalid file_size
        with pytest.raises(ValueError):
            BatchDocument(
                document_id="doc_001",
                batch_id="batch_001",
                file_path="/path/to/document.txt",
                file_name="document.txt",
                file_size=-1,
                mime_type="text/plain",
                status=DocumentStatus.PENDING,
                correlation_id="doc_doc_001",
            )

        # Test invalid retry_count
        with pytest.raises(ValueError):
            BatchDocument(
                document_id="doc_001",
                batch_id="batch_001",
                file_path="/path/to/document.txt",
                file_name="document.txt",
                file_size=1024,
                mime_type="text/plain",
                status=DocumentStatus.PENDING,
                retry_count=-1,
                correlation_id="doc_doc_001",
            )


class TestBatchProgress:
    """Test BatchProgress model"""

    def test_batch_progress_creation(self):
        """Test batch progress creation"""
        now = datetime.now()

        progress = BatchProgress(
            batch_id="batch_001",
            current_phase=ProcessingPhase.PROCESSING,
            total_documents=10,
            processed_documents=5,
            successful_documents=4,
            failed_documents=1,
            progress_percentage=50.0,
            active_workers=2,
            error_count=1,
            last_updated=now,
        )

        assert progress.batch_id == "batch_001"
        assert progress.current_phase == ProcessingPhase.PROCESSING
        assert progress.total_documents == 10
        assert progress.processed_documents == 5
        assert progress.successful_documents == 4
        assert progress.failed_documents == 1
        assert progress.progress_percentage == 50.0
        assert progress.active_workers == 2
        assert progress.error_count == 1
        assert progress.last_updated == now

    def test_batch_progress_validation(self):
        """Test batch progress validation"""
        now = datetime.now()

        # Test valid progress
        progress = BatchProgress(
            batch_id="batch_001",
            current_phase=ProcessingPhase.PROCESSING,
            total_documents=10,
            progress_percentage=50.0,
            last_updated=now,
        )
        assert progress.progress_percentage == 50.0

        # Test invalid progress_percentage
        with pytest.raises(ValueError):
            BatchProgress(
                batch_id="batch_001",
                current_phase=ProcessingPhase.PROCESSING,
                total_documents=10,
                progress_percentage=-1.0,
                last_updated=now,
            )

        with pytest.raises(ValueError):
            BatchProgress(
                batch_id="batch_001",
                current_phase=ProcessingPhase.PROCESSING,
                total_documents=10,
                progress_percentage=101.0,
                last_updated=now,
            )

        # Test invalid active_workers
        with pytest.raises(ValueError):
            BatchProgress(
                batch_id="batch_001",
                current_phase=ProcessingPhase.PROCESSING,
                total_documents=10,
                progress_percentage=50.0,
                active_workers=-1,
                last_updated=now,
            )


class TestBatchResult:
    """Test BatchResult model"""

    def test_batch_result_creation(self):
        """Test batch result creation"""
        now = datetime.now()
        duration = timedelta(minutes=5)

        result = BatchResult(
            batch_id="batch_001",
            total_documents=10,
            successful_evaluations=8,
            failed_evaluations=2,
            success_rate=0.8,
            processing_duration=duration,
            generated_at=now,
        )

        assert result.batch_id == "batch_001"
        assert result.total_documents == 10
        assert result.successful_evaluations == 8
        assert result.failed_evaluations == 2
        assert result.success_rate == 0.8
        assert result.processing_duration == duration
        assert result.generated_at == now

    def test_batch_result_validation(self):
        """Test batch result validation"""
        now = datetime.now()
        duration = timedelta(minutes=5)

        # Test valid result
        result = BatchResult(
            batch_id="batch_001",
            total_documents=10,
            successful_evaluations=8,
            failed_evaluations=2,
            success_rate=0.8,
            processing_duration=duration,
            generated_at=now,
        )
        assert result.success_rate == 0.8

        # Test invalid success_rate
        with pytest.raises(ValueError):
            BatchResult(
                batch_id="batch_001",
                total_documents=10,
                successful_evaluations=8,
                failed_evaluations=2,
                success_rate=-0.1,
                processing_duration=duration,
                generated_at=now,
            )

        with pytest.raises(ValueError):
            BatchResult(
                batch_id="batch_001",
                total_documents=10,
                successful_evaluations=8,
                failed_evaluations=2,
                success_rate=1.1,
                processing_duration=duration,
                generated_at=now,
            )


class TestDocumentInput:
    """Test DocumentInput model"""

    def test_document_input_creation(self):
        """Test document input creation"""
        doc_input = DocumentInput(
            file_path="/path/to/document.txt",
            file_name="document.txt",
            context="Test context",
            output_text="Test output",
        )

        assert doc_input.file_path == "/path/to/document.txt"
        assert doc_input.file_name == "document.txt"
        assert doc_input.context == "Test context"
        assert doc_input.output_text == "Test output"

    def test_document_input_optional_fields(self):
        """Test document input with optional fields"""
        doc_input = DocumentInput(file_path="/path/to/document.txt")

        assert doc_input.file_path == "/path/to/document.txt"
        assert doc_input.file_name is None
        assert doc_input.context is None
        assert doc_input.output_text is None


class TestBatchSummaryStatistics:
    """Test BatchSummaryStatistics model"""

    def test_summary_statistics_creation(self):
        """Test summary statistics creation"""
        stats = BatchSummaryStatistics(
            average_score=3.5,
            score_distribution={1: 2, 2: 3, 3: 5, 4: 4, 5: 1},
            most_common_criteria=["article_5_1", "article_5_2"],
            processing_efficiency=12.5,
            error_rate=0.1,
        )

        assert stats.average_score == 3.5
        assert stats.score_distribution == {1: 2, 2: 3, 3: 5, 4: 4, 5: 1}
        assert stats.most_common_criteria == ["article_5_1", "article_5_2"]
        assert stats.processing_efficiency == 12.5
        assert stats.error_rate == 0.1

    def test_summary_statistics_validation(self):
        """Test summary statistics validation"""
        # Test valid statistics
        stats = BatchSummaryStatistics(
            average_score=3.5,
            score_distribution={1: 2, 2: 3, 3: 5, 4: 4, 5: 1},
            most_common_criteria=["article_5_1", "article_5_2"],
            processing_efficiency=12.5,
            error_rate=0.1,
        )
        assert stats.average_score == 3.5

        # Test invalid average_score
        with pytest.raises(ValueError):
            BatchSummaryStatistics(
                average_score=0.5,
                score_distribution={1: 2, 2: 3, 3: 5, 4: 4, 5: 1},
                most_common_criteria=["article_5_1", "article_5_2"],
                processing_efficiency=12.5,
                error_rate=0.1,
            )

        with pytest.raises(ValueError):
            BatchSummaryStatistics(
                average_score=5.5,
                score_distribution={1: 2, 2: 3, 3: 5, 4: 4, 5: 1},
                most_common_criteria=["article_5_1", "article_5_2"],
                processing_efficiency=12.5,
                error_rate=0.1,
            )

        # Test invalid error_rate
        with pytest.raises(ValueError):
            BatchSummaryStatistics(
                average_score=3.5,
                score_distribution={1: 2, 2: 3, 3: 5, 4: 4, 5: 1},
                most_common_criteria=["article_5_1", "article_5_2"],
                processing_efficiency=12.5,
                error_rate=-0.1,
            )

        with pytest.raises(ValueError):
            BatchSummaryStatistics(
                average_score=3.5,
                score_distribution={1: 2, 2: 3, 3: 5, 4: 4, 5: 1},
                most_common_criteria=["article_5_1", "article_5_2"],
                processing_efficiency=12.5,
                error_rate=1.1,
            )


class TestDocumentResult:
    """Test DocumentResult model"""

    def test_document_result_creation(self):
        """Test document result creation"""
        duration = timedelta(seconds=30)

        result = DocumentResult(
            document_id="doc_001", processing_time=duration, success=True
        )

        assert result.document_id == "doc_001"
        assert result.evaluation_result is None
        assert result.processing_time == duration
        assert result.success is True


class TestDocumentError:
    """Test DocumentError model"""

    def test_document_error_creation(self):
        """Test document error creation"""
        now = datetime.now()

        error = DocumentError(
            document_id="doc_001",
            error_type="PROCESSING_ERROR",
            error_message="Failed to process document",
            retry_count=2,
            occurred_at=now,
        )

        assert error.document_id == "doc_001"
        assert error.error_type == "PROCESSING_ERROR"
        assert error.error_message == "Failed to process document"
        assert error.retry_count == 2
        assert error.occurred_at == now

    def test_document_error_defaults(self):
        """Test document error with default values"""
        now = datetime.now()

        error = DocumentError(
            document_id="doc_001",
            error_type="PROCESSING_ERROR",
            error_message="Failed to process document",
            occurred_at=now,
        )

        assert error.retry_count == 0
