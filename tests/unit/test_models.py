"""
Unit tests for Pydantic models.
Tests lines 394-698: All Pydantic models validation
"""

import pytest
from datetime import datetime, timedelta
from pydantic import ValidationError

from main import (
    EvaluationStep,
    CriterionEvaluation,
    OverallEvaluation,
    DisclosureEvaluationResult,
    BatchStatus,
    DocumentStatus,
    ProcessingPhase,
    BatchConfiguration,
    BatchSummaryStatistics,
    DocumentResult,
    DocumentError,
    BatchEvaluation,
    BatchDocument,
    BatchProgress,
    BatchResult,
    DocumentInput,
)


class TestEvaluationStep:
    """Test EvaluationStep model."""

    def test_evaluation_step_valid(self):
        """Test valid EvaluationStep creation."""
        step = EvaluationStep(
            step="Test step", result="YES", reasoning="Test reasoning"
        )

        assert step.step == "Test step"
        assert step.result == "YES"
        assert step.reasoning == "Test reasoning"

    def test_evaluation_step_result_no(self):
        """Test EvaluationStep with NO result."""
        step = EvaluationStep(step="Test step", result="NO", reasoning="Test reasoning")

        assert step.result == "NO"

    def test_evaluation_step_extra_fields_forbidden(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            EvaluationStep(
                step="Test step",
                result="YES",
                reasoning="Test reasoning",
                extra_field="not allowed",
            )


class TestCriterionEvaluation:
    """Test CriterionEvaluation model."""

    def test_criterion_evaluation_valid(self, sample_evaluation_step):
        """Test valid CriterionEvaluation creation."""
        criterion = CriterionEvaluation(
            criterion_id="article_5_1",
            criterion_name="個人情報保護",
            article="第5条第1号",
            steps=[sample_evaluation_step],
            score=3,
            score_reasoning="Test reasoning",
        )

        assert criterion.criterion_id == "article_5_1"
        assert criterion.criterion_name == "個人情報保護"
        assert criterion.article == "第5条第1号"
        assert len(criterion.steps) == 1
        assert criterion.score == 3
        assert criterion.score_reasoning == "Test reasoning"

    def test_criterion_evaluation_score_constraints(self, sample_evaluation_step):
        """Test CriterionEvaluation score constraints."""
        # Valid scores (1-5)
        for score in [1, 2, 3, 4, 5]:
            criterion = CriterionEvaluation(
                criterion_id="test",
                criterion_name="Test",
                article="Test",
                steps=[sample_evaluation_step],
                score=score,
                score_reasoning="Test",
            )
            assert criterion.score == score

        # Invalid scores
        for score in [0, 6, -1, 10]:
            with pytest.raises(
                ValidationError, match="Input should be greater than or equal to 1"
            ):
                CriterionEvaluation(
                    criterion_id="test",
                    criterion_name="Test",
                    article="Test",
                    steps=[sample_evaluation_step],
                    score=score,
                    score_reasoning="Test",
                )

    def test_criterion_evaluation_multiple_steps(self):
        """Test CriterionEvaluation with multiple steps."""
        steps = [
            EvaluationStep(step="Step 1", result="YES", reasoning="Reason 1"),
            EvaluationStep(step="Step 2", result="NO", reasoning="Reason 2"),
            EvaluationStep(step="Step 3", result="YES", reasoning="Reason 3"),
        ]

        criterion = CriterionEvaluation(
            criterion_id="test",
            criterion_name="Test",
            article="Test",
            steps=steps,
            score=3,
            score_reasoning="Test reasoning",
        )

        assert len(criterion.steps) == 3
        assert criterion.steps[0].result == "YES"
        assert criterion.steps[1].result == "NO"
        assert criterion.steps[2].result == "YES"


class TestOverallEvaluation:
    """Test OverallEvaluation model."""

    def test_overall_evaluation_creation(self):
        """Test OverallEvaluation model creation."""
        # OverallEvaluation is an empty model, just test it can be instantiated
        overall = OverallEvaluation()
        assert isinstance(overall, OverallEvaluation)


class TestDisclosureEvaluationResult:
    """Test DisclosureEvaluationResult model."""

    def test_disclosure_evaluation_result_valid(self, sample_criterion):
        """Test valid DisclosureEvaluationResult creation."""
        criterion_eval = CriterionEvaluation(
            criterion_id=sample_criterion["id"],
            criterion_name=sample_criterion["name"],
            article=sample_criterion["article"],
            steps=[EvaluationStep(step="Test", result="YES", reasoning="Test")],
            score=3,
            score_reasoning="Test",
        )

        result = DisclosureEvaluationResult(
            input_text="Test input",
            context="Test context",
            output_text="Test output",
            criterion_evaluations=[criterion_eval],
            evaluation_timestamp="2025-01-01T12:00:00",
        )

        assert result.input_text == "Test input"
        assert result.context == "Test context"
        assert result.output_text == "Test output"
        assert len(result.criterion_evaluations) == 1
        assert result.evaluation_timestamp == "2025-01-01T12:00:00"

    def test_disclosure_evaluation_result_empty_criteria(self):
        """Test DisclosureEvaluationResult with empty criteria list."""
        result = DisclosureEvaluationResult(
            input_text="Test input",
            context="Test context",
            output_text="Test output",
            criterion_evaluations=[],
            evaluation_timestamp="2025-01-01T12:00:00",
        )

        assert len(result.criterion_evaluations) == 0


class TestEnums:
    """Test enum models."""

    def test_batch_status_enum(self):
        """Test BatchStatus enum values."""
        assert BatchStatus.PENDING == "pending"
        assert BatchStatus.PROCESSING == "processing"
        assert BatchStatus.COMPLETED == "completed"
        assert BatchStatus.FAILED == "failed"
        assert BatchStatus.PARTIALLY_FAILED == "partially_failed"

    def test_document_status_enum(self):
        """Test DocumentStatus enum values."""
        assert DocumentStatus.PENDING == "pending"
        assert DocumentStatus.PROCESSING == "processing"
        assert DocumentStatus.COMPLETED == "completed"
        assert DocumentStatus.FAILED == "failed"

    def test_processing_phase_enum(self):
        """Test ProcessingPhase enum values."""
        assert ProcessingPhase.INITIALIZING == "initializing"
        assert ProcessingPhase.DISCOVERING == "discovering"
        assert ProcessingPhase.PROCESSING == "processing"
        assert ProcessingPhase.AGGREGATING == "aggregating"
        assert ProcessingPhase.COMPLETED == "completed"


class TestBatchConfiguration:
    """Test BatchConfiguration model."""

    def test_batch_configuration_defaults(self):
        """Test BatchConfiguration with default values."""
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

    def test_batch_configuration_custom_values(self):
        """Test BatchConfiguration with custom values."""
        config = BatchConfiguration(
            max_concurrent_workers=10,
            max_retry_attempts=5,
            timeout_seconds=600,
            progress_update_interval=20,
            enable_resumption=False,
            output_formats=["json"],
            file_size_limit=100 * 1024 * 1024,
            memory_limit_mb=4096,
            api_rate_limit_delay=0.5,
            retry_delay_seconds=60,
            exponential_backoff=False,
        )

        assert config.max_concurrent_workers == 10
        assert config.max_retry_attempts == 5
        assert config.timeout_seconds == 600
        assert config.progress_update_interval == 20
        assert config.enable_resumption is False
        assert config.output_formats == ["json"]
        assert config.file_size_limit == 100 * 1024 * 1024
        assert config.memory_limit_mb == 4096
        assert config.api_rate_limit_delay == 0.5
        assert config.retry_delay_seconds == 60
        assert config.exponential_backoff is False

    def test_batch_configuration_constraints(self):
        """Test BatchConfiguration field constraints."""
        # Test max_concurrent_workers constraints
        with pytest.raises(
            ValidationError, match="Input should be greater than or equal to 1"
        ):
            BatchConfiguration(max_concurrent_workers=0)

        with pytest.raises(
            ValidationError, match="Input should be less than or equal to 20"
        ):
            BatchConfiguration(max_concurrent_workers=21)

        # Test max_retry_attempts constraints
        with pytest.raises(
            ValidationError, match="Input should be greater than or equal to 0"
        ):
            BatchConfiguration(max_retry_attempts=-1)

        with pytest.raises(
            ValidationError, match="Input should be less than or equal to 10"
        ):
            BatchConfiguration(max_retry_attempts=11)

        # Test timeout_seconds constraints
        with pytest.raises(
            ValidationError, match="Input should be greater than or equal to 30"
        ):
            BatchConfiguration(timeout_seconds=29)

        with pytest.raises(
            ValidationError, match="Input should be less than or equal to 3600"
        ):
            BatchConfiguration(timeout_seconds=3601)

        # Test file_size_limit constraints
        with pytest.raises(
            ValidationError, match="Input should be greater than or equal to 1024"
        ):
            BatchConfiguration(file_size_limit=1023)

        # Test memory_limit_mb constraints
        with pytest.raises(
            ValidationError, match="Input should be greater than or equal to 256"
        ):
            BatchConfiguration(memory_limit_mb=255)

        # Test api_rate_limit_delay constraints
        with pytest.raises(
            ValidationError, match="Input should be greater than or equal to 0.0"
        ):
            BatchConfiguration(api_rate_limit_delay=-0.1)

        with pytest.raises(
            ValidationError, match="Input should be less than or equal to 10.0"
        ):
            BatchConfiguration(api_rate_limit_delay=10.1)

        # Test retry_delay_seconds constraints
        with pytest.raises(
            ValidationError, match="Input should be greater than or equal to 1"
        ):
            BatchConfiguration(retry_delay_seconds=0)

        with pytest.raises(
            ValidationError, match="Input should be less than or equal to 300"
        ):
            BatchConfiguration(retry_delay_seconds=301)


class TestBatchSummaryStatistics:
    """Test BatchSummaryStatistics model."""

    def test_batch_summary_statistics_valid(self):
        """Test valid BatchSummaryStatistics creation."""
        stats = BatchSummaryStatistics(
            average_score=3.5,
            score_distribution={1: 2, 2: 3, 3: 5, 4: 3, 5: 2},
            most_common_criteria=["article_5_1", "article_5_2"],
            processing_efficiency=10.5,
            error_rate=0.1,
        )

        assert stats.average_score == 3.5
        assert stats.score_distribution[1] == 2
        assert len(stats.most_common_criteria) == 2
        assert stats.processing_efficiency == 10.5
        assert stats.error_rate == 0.1

    def test_batch_summary_statistics_constraints(self):
        """Test BatchSummaryStatistics field constraints."""
        # Test average_score constraints
        with pytest.raises(
            ValidationError, match="Input should be greater than or equal to 1.0"
        ):
            BatchSummaryStatistics(
                average_score=0.5,
                score_distribution={},
                most_common_criteria=[],
                processing_efficiency=0.0,
                error_rate=0.0,
            )

        with pytest.raises(
            ValidationError, match="Input should be less than or equal to 5.0"
        ):
            BatchSummaryStatistics(
                average_score=5.5,
                score_distribution={},
                most_common_criteria=[],
                processing_efficiency=0.0,
                error_rate=0.0,
            )

        # Test processing_efficiency constraints
        with pytest.raises(
            ValidationError, match="Input should be greater than or equal to 0.0"
        ):
            BatchSummaryStatistics(
                average_score=3.0,
                score_distribution={},
                most_common_criteria=[],
                processing_efficiency=-1.0,
                error_rate=0.0,
            )

        # Test error_rate constraints
        with pytest.raises(
            ValidationError, match="Input should be greater than or equal to 0.0"
        ):
            BatchSummaryStatistics(
                average_score=3.0,
                score_distribution={},
                most_common_criteria=[],
                processing_efficiency=0.0,
                error_rate=-0.1,
            )

        with pytest.raises(
            ValidationError, match="Input should be less than or equal to 1.0"
        ):
            BatchSummaryStatistics(
                average_score=3.0,
                score_distribution={},
                most_common_criteria=[],
                processing_efficiency=0.0,
                error_rate=1.1,
            )


class TestDocumentResult:
    """Test DocumentResult model."""

    def test_document_result_valid(self, sample_document_result):
        """Test valid DocumentResult creation."""
        result = DocumentResult(
            document_id="doc_001",
            evaluation_result=DisclosureEvaluationResult(
                input_text="Test input",
                context="Test context",
                output_text="Test output",
                criterion_evaluations=[],
                evaluation_timestamp="2025-01-01T12:00:00",
            ),
            processing_time=timedelta(seconds=30),
            success=True,
        )

        assert result.document_id == "doc_001"
        assert result.evaluation_result is not None
        assert result.processing_time == timedelta(seconds=30)
        assert result.success is True

    def test_document_result_with_none_evaluation(self):
        """Test DocumentResult with None evaluation_result."""
        result = DocumentResult(
            document_id="doc_001",
            evaluation_result=None,
            processing_time=timedelta(seconds=30),
            success=False,
        )

        assert result.document_id == "doc_001"
        assert result.evaluation_result is None
        assert result.success is False


class TestDocumentError:
    """Test DocumentError model."""

    def test_document_error_valid(self):
        """Test valid DocumentError creation."""
        error = DocumentError(
            document_id="doc_001",
            error_type="API_ERROR",
            error_message="API request failed",
            retry_count=2,
            occurred_at=datetime.now(),
        )

        assert error.document_id == "doc_001"
        assert error.error_type == "API_ERROR"
        assert error.error_message == "API request failed"
        assert error.retry_count == 2
        assert isinstance(error.occurred_at, datetime)


class TestBatchEvaluation:
    """Test BatchEvaluation model."""

    def test_batch_evaluation_valid(self, test_batch_configuration):
        """Test valid BatchEvaluation creation."""
        batch = BatchEvaluation(
            batch_id="batch_001",
            created_at=datetime.now(),
            status=BatchStatus.PENDING,
            total_documents=10,
            correlation_id="correlation_001",
            configuration=test_batch_configuration,
        )

        assert batch.batch_id == "batch_001"
        assert batch.status == BatchStatus.PENDING
        assert batch.total_documents == 10
        assert batch.correlation_id == "correlation_001"
        assert batch.configuration == test_batch_configuration


class TestBatchDocument:
    """Test BatchDocument model."""

    def test_batch_document_valid(self):
        """Test valid BatchDocument creation."""
        document = BatchDocument(
            document_id="doc_001",
            batch_id="batch_001",
            file_path="/path/to/file.txt",
            file_name="file.txt",
            file_size=1024,
            mime_type="text/plain",
            status=DocumentStatus.PENDING,
            correlation_id="correlation_001",
            context="Test context",
            output_text="Test output",
        )

        assert document.document_id == "doc_001"
        assert document.batch_id == "batch_001"
        assert document.file_path == "/path/to/file.txt"
        assert document.file_name == "file.txt"
        assert document.file_size == 1024
        assert document.mime_type == "text/plain"
        assert document.status == DocumentStatus.PENDING
        assert document.correlation_id == "correlation_001"
        assert document.context == "Test context"
        assert document.output_text == "Test output"


class TestBatchProgress:
    """Test BatchProgress model."""

    def test_batch_progress_valid(self):
        """Test valid BatchProgress creation."""
        progress = BatchProgress(
            batch_id="batch_001",
            current_phase=ProcessingPhase.PROCESSING,
            total_documents=10,
            processed_documents=5,
            successful_documents=4,
            failed_documents=1,
            progress_percentage=50.0,
            last_updated=datetime.now(),
        )

        assert progress.batch_id == "batch_001"
        assert progress.current_phase == ProcessingPhase.PROCESSING
        assert progress.total_documents == 10
        assert progress.processed_documents == 5
        assert progress.successful_documents == 4
        assert progress.failed_documents == 1
        assert progress.progress_percentage == 50.0
        assert isinstance(progress.last_updated, datetime)

    def test_batch_progress_percentage_calculation(self):
        """Test BatchProgress percentage calculation."""
        # Test 100% progress
        progress = BatchProgress(
            batch_id="batch_001",
            current_phase=ProcessingPhase.PROCESSING,
            total_documents=10,
            processed_documents=10,
            successful_documents=8,
            failed_documents=2,
            progress_percentage=100.0,
            last_updated=datetime.now(),
        )

        assert progress.progress_percentage == 100.0

        # Test 0% progress
        progress = BatchProgress(
            batch_id="batch_001",
            current_phase=ProcessingPhase.PROCESSING,
            total_documents=10,
            processed_documents=0,
            successful_documents=0,
            failed_documents=0,
            progress_percentage=0.0,
            last_updated=datetime.now(),
        )

        assert progress.progress_percentage == 0.0


class TestBatchResult:
    """Test BatchResult model."""

    def test_batch_result_valid(self, sample_document_result):
        """Test valid BatchResult creation."""
        result = BatchResult(
            batch_id="batch_001",
            total_documents=10,
            successful_evaluations=8,
            failed_evaluations=2,
            success_rate=0.8,
            processing_duration=timedelta(minutes=5),
            average_evaluation_time=timedelta(seconds=30),
            individual_results=[sample_document_result],
            generated_at=datetime.now(),
        )

        assert result.batch_id == "batch_001"
        assert result.total_documents == 10
        assert result.successful_evaluations == 8
        assert result.failed_evaluations == 2
        assert result.success_rate == 0.8
        assert result.processing_duration == timedelta(minutes=5)
        assert result.average_evaluation_time == timedelta(seconds=30)
        assert len(result.individual_results) == 1
        assert isinstance(result.generated_at, datetime)


class TestDocumentInput:
    """Test DocumentInput model."""

    def test_document_input_valid(self):
        """Test valid DocumentInput creation."""
        doc_input = DocumentInput(
            file_path="/path/to/file.txt",
            file_name="file.txt",
            context="Test context",
            output_text="Test output",
        )

        assert doc_input.file_path == "/path/to/file.txt"
        assert doc_input.file_name == "file.txt"
        assert doc_input.context == "Test context"
        assert doc_input.output_text == "Test output"

    def test_document_input_optional_fields(self):
        """Test DocumentInput with optional fields."""
        doc_input = DocumentInput(file_path="/path/to/file.txt")

        assert doc_input.file_path == "/path/to/file.txt"
        assert doc_input.file_name is None
        assert doc_input.context is None
        assert doc_input.output_text is None
