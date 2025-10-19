"""
Unit tests for BatchStatePersistenceService class.
Tests lines 1812-1872: BatchStatePersistenceService including save/load operations
"""

import json
import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from datetime import datetime

from main import (
    BatchStatePersistenceService,
    BatchConfiguration,
    BatchEvaluation,
    BatchStatus,
)


class TestBatchStatePersistenceService:
    """Test BatchStatePersistenceService class."""

    def test_batch_state_persistence_service_init(
        self, test_batch_configuration, temp_batch_state_dir
    ):
        """Test BatchStatePersistenceService initialization."""
        with patch("main.logging.getLogger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            with patch("pathlib.Path.mkdir") as mock_mkdir:
                service = BatchStatePersistenceService(test_batch_configuration)

                assert service.config == test_batch_configuration
                assert service.logger == mock_logger
                assert service.state_dir == Path("batch_state")
                assert service.active_dir == Path("batch_state/active_batches")
                assert service.completed_dir == Path("batch_state/completed_batches")

                # Verify directories are created
                assert mock_mkdir.call_count == 2
                mock_mkdir.assert_any_call(parents=True, exist_ok=True)
                mock_get_logger.assert_called_once_with("BatchStatePersistenceService")

    def test_save_batch_state_success(
        self, test_batch_configuration, sample_batch_evaluation
    ):
        """Test successful batch state saving."""
        with patch("main.logging.getLogger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            with patch("pathlib.Path.mkdir"):
                with patch("builtins.open", mock_open()) as mock_file:
                    service = BatchStatePersistenceService(test_batch_configuration)
                    service.save_batch_state(sample_batch_evaluation)

                    # Verify file was opened for writing
                    mock_file.assert_called_once()
                    call_args = mock_file.call_args
                    assert "w" in call_args[1]["mode"]
                    assert "utf-8" in call_args[1]["encoding"]

                    # Verify JSON was written
                    written_data = mock_file().write.call_args[0][0]
                    assert "batch_id" in written_data
                    assert "test_batch_001" in written_data
                    assert "status" in written_data
                    assert "pending" in written_data

                    mock_logger.info.assert_called()

    def test_save_batch_state_exception(
        self, test_batch_configuration, sample_batch_evaluation
    ):
        """Test batch state saving with exception."""
        with patch("main.logging.getLogger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            with patch("pathlib.Path.mkdir"):
                with patch("builtins.open", side_effect=IOError("Permission denied")):
                    service = BatchStatePersistenceService(test_batch_configuration)

                    with pytest.raises(IOError, match="Permission denied"):
                        service.save_batch_state(sample_batch_evaluation)

                    mock_logger.error.assert_called_with(
                        "Failed to save batch state: Permission denied"
                    )

    def test_save_batch_state_enum_serialization(
        self, test_batch_configuration, sample_batch_evaluation
    ):
        """Test that enums are properly serialized to strings."""
        with patch("main.logging.getLogger"):
            with patch("pathlib.Path.mkdir"):
                with patch("builtins.open", mock_open()) as mock_file:
                    service = BatchStatePersistenceService(test_batch_configuration)
                    service.save_batch_state(sample_batch_evaluation)

                    # Verify the written JSON contains string enum values
                    written_data = mock_file().write.call_args[0][0]
                    json_data = json.loads(written_data)
                    assert json_data["status"] == "pending"  # String, not enum
                    assert isinstance(json_data["status"], str)

    def test_save_batch_state_datetime_serialization(
        self, test_batch_configuration, sample_batch_evaluation
    ):
        """Test that datetime objects are properly serialized."""
        with patch("main.logging.getLogger"):
            with patch("pathlib.Path.mkdir"):
                with patch("builtins.open", mock_open()) as mock_file:
                    service = BatchStatePersistenceService(test_batch_configuration)
                    service.save_batch_state(sample_batch_evaluation)

                    # Verify the written JSON contains serialized datetime
                    written_data = mock_file().write.call_args[0][0]
                    json_data = json.loads(written_data)
                    assert "created_at" in json_data
                    assert isinstance(json_data["created_at"], str)

    def test_load_batch_state_success(
        self, test_batch_configuration, sample_batch_evaluation
    ):
        """Test successful batch state loading."""
        with patch("main.logging.getLogger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            with patch("pathlib.Path.mkdir"):
                # Mock file content
                batch_data = {
                    "batch_id": "test_batch_001",
                    "created_at": "2025-01-01T12:00:00",
                    "status": "pending",
                    "total_documents": 5,
                    "correlation_id": "test_correlation",
                    "configuration": {
                        "max_concurrent_workers": 5,
                        "max_retry_attempts": 3,
                        "timeout_seconds": 300,
                        "progress_update_interval": 10,
                        "enable_resumption": True,
                        "output_formats": ["json", "summary"],
                        "file_size_limit": 52428800,
                        "memory_limit_mb": 2048,
                        "api_rate_limit_delay": 0.1,
                        "retry_delay_seconds": 30,
                        "exponential_backoff": True,
                    },
                }

                with patch(
                    "builtins.open", mock_open(read_data=json.dumps(batch_data))
                ):
                    with patch("pathlib.Path.exists", return_value=True):
                        service = BatchStatePersistenceService(test_batch_configuration)
                        result = service.load_batch_state("test_batch_001")

                        assert result is not None
                        assert isinstance(result, BatchEvaluation)
                        assert result.batch_id == "test_batch_001"
                        assert result.status == BatchStatus.PENDING
                        assert result.total_documents == 5

    def test_load_batch_state_file_not_found(self, test_batch_configuration):
        """Test batch state loading when file doesn't exist."""
        with patch("main.logging.getLogger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            with patch("pathlib.Path.mkdir"):
                with patch("pathlib.Path.exists", return_value=False):
                    service = BatchStatePersistenceService(test_batch_configuration)
                    result = service.load_batch_state("nonexistent_batch")

                    assert result is None

    def test_load_batch_state_exception(self, test_batch_configuration):
        """Test batch state loading with exception."""
        with patch("main.logging.getLogger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            with patch("pathlib.Path.mkdir"):
                with patch("pathlib.Path.exists", return_value=True):
                    with patch(
                        "builtins.open", side_effect=IOError("Permission denied")
                    ):
                        service = BatchStatePersistenceService(test_batch_configuration)
                        result = service.load_batch_state("test_batch_001")

                        assert result is None
                        mock_logger.error.assert_called_with(
                            "Failed to load batch state: Permission denied"
                        )

    def test_load_batch_state_invalid_json(self, test_batch_configuration):
        """Test batch state loading with invalid JSON."""
        with patch("main.logging.getLogger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            with patch("pathlib.Path.mkdir"):
                with patch("pathlib.Path.exists", return_value=True):
                    with patch("builtins.open", mock_open(read_data="invalid json")):
                        service = BatchStatePersistenceService(test_batch_configuration)
                        result = service.load_batch_state("test_batch_001")

                        assert result is None
                        mock_logger.error.assert_called()

    def test_load_batch_state_missing_required_fields(self, test_batch_configuration):
        """Test batch state loading with missing required fields."""
        with patch("main.logging.getLogger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            with patch("pathlib.Path.mkdir"):
                with patch("pathlib.Path.exists", return_value=True):
                    # Incomplete batch data
                    incomplete_data = {
                        "batch_id": "test_batch_001"
                        # Missing other required fields
                    }

                    with patch(
                        "builtins.open",
                        mock_open(read_data=json.dumps(incomplete_data)),
                    ):
                        service = BatchStatePersistenceService(test_batch_configuration)
                        result = service.load_batch_state("test_batch_001")

                        # Should return None due to validation error
                        assert result is None
                        mock_logger.error.assert_called()

    def test_move_to_completed_success(self, test_batch_configuration):
        """Test successful batch state move to completed."""
        with patch("main.logging.getLogger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            with patch("pathlib.Path.mkdir"):
                with patch("pathlib.Path.exists", return_value=True):
                    with patch("pathlib.Path.rename") as mock_rename:
                        service = BatchStatePersistenceService(test_batch_configuration)
                        service.move_to_completed("test_batch_001")

                        mock_rename.assert_called_once()
                        mock_logger.info.assert_called_with(
                            "Batch moved to completed: test_batch_001"
                        )

    def test_move_to_completed_file_not_found(self, test_batch_configuration):
        """Test batch state move to completed when file doesn't exist."""
        with patch("main.logging.getLogger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            with patch("pathlib.Path.mkdir"):
                with patch("pathlib.Path.exists", return_value=False):
                    with patch("pathlib.Path.rename") as mock_rename:
                        service = BatchStatePersistenceService(test_batch_configuration)
                        service.move_to_completed("nonexistent_batch")

                        # Should not call rename if file doesn't exist
                        mock_rename.assert_not_called()
                        mock_logger.info.assert_not_called()

    def test_move_to_completed_exception(self, test_batch_configuration):
        """Test batch state move to completed with exception."""
        with patch("main.logging.getLogger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            with patch("pathlib.Path.mkdir"):
                with patch("pathlib.Path.exists", return_value=True):
                    with patch(
                        "pathlib.Path.rename", side_effect=OSError("Permission denied")
                    ):
                        service = BatchStatePersistenceService(test_batch_configuration)

                        with pytest.raises(OSError, match="Permission denied"):
                            service.move_to_completed("test_batch_001")

                        mock_logger.error.assert_called_with(
                            "Failed to move batch to completed: Permission denied"
                        )

    def test_save_batch_state_with_complex_configuration(
        self, test_batch_configuration
    ):
        """Test saving batch state with complex configuration."""
        complex_config = BatchConfiguration(
            max_concurrent_workers=10,
            max_retry_attempts=5,
            timeout_seconds=600,
            progress_update_interval=20,
            enable_resumption=False,
            output_formats=["json", "csv"],
            file_size_limit=100 * 1024 * 1024,
            memory_limit_mb=4096,
            api_rate_limit_delay=0.5,
            retry_delay_seconds=60,
            exponential_backoff=False,
        )

        batch_evaluation = BatchEvaluation(
            batch_id="complex_batch_001",
            created_at=datetime.now(),
            status=BatchStatus.PROCESSING,
            total_documents=100,
            correlation_id="complex_correlation",
            configuration=complex_config,
        )

        with patch("main.logging.getLogger"):
            with patch("pathlib.Path.mkdir"):
                with patch("builtins.open", mock_open()) as mock_file:
                    service = BatchStatePersistenceService(test_batch_configuration)
                    service.save_batch_state(batch_evaluation)

                    # Verify complex configuration was saved
                    written_data = mock_file().write.call_args[0][0]
                    json_data = json.loads(written_data)
                    assert json_data["configuration"]["max_concurrent_workers"] == 10
                    assert json_data["configuration"]["max_retry_attempts"] == 5
                    assert json_data["configuration"]["timeout_seconds"] == 600
                    assert json_data["configuration"]["enable_resumption"] is False
                    assert json_data["configuration"]["output_formats"] == [
                        "json",
                        "csv",
                    ]
                    assert json_data["configuration"]["file_size_limit"] == 104857600
                    assert json_data["configuration"]["memory_limit_mb"] == 4096
                    assert json_data["configuration"]["api_rate_limit_delay"] == 0.5
                    assert json_data["configuration"]["retry_delay_seconds"] == 60
                    assert json_data["configuration"]["exponential_backoff"] is False

    def test_load_batch_state_with_different_statuses(self, test_batch_configuration):
        """Test loading batch state with different status values."""
        statuses = ["pending", "processing", "completed", "failed", "partially_failed"]

        for status in statuses:
            batch_data = {
                "batch_id": f"test_batch_{status}",
                "created_at": "2025-01-01T12:00:00",
                "status": status,
                "total_documents": 5,
                "correlation_id": "test_correlation",
                "configuration": {
                    "max_concurrent_workers": 5,
                    "max_retry_attempts": 3,
                    "timeout_seconds": 300,
                    "progress_update_interval": 10,
                    "enable_resumption": True,
                    "output_formats": ["json", "summary"],
                    "file_size_limit": 52428800,
                    "memory_limit_mb": 2048,
                    "api_rate_limit_delay": 0.1,
                    "retry_delay_seconds": 30,
                    "exponential_backoff": True,
                },
            }

            with patch("main.logging.getLogger"):
                with patch("pathlib.Path.mkdir"):
                    with patch("pathlib.Path.exists", return_value=True):
                        with patch(
                            "builtins.open", mock_open(read_data=json.dumps(batch_data))
                        ):
                            service = BatchStatePersistenceService(
                                test_batch_configuration
                            )
                            result = service.load_batch_state(f"test_batch_{status}")

                            assert result is not None
                            assert result.batch_id == f"test_batch_{status}"
                            assert result.status.value == status

    def test_save_batch_state_unicode_content(self, test_batch_configuration):
        """Test saving batch state with Unicode content."""
        batch_evaluation = BatchEvaluation(
            batch_id="unicode_batch_001",
            created_at=datetime.now(),
            status=BatchStatus.PENDING,
            total_documents=5,
            correlation_id="unicode_correlation_テスト",
            configuration=test_batch_configuration,
        )

        with patch("main.logging.getLogger"):
            with patch("pathlib.Path.mkdir"):
                with patch("builtins.open", mock_open()) as mock_file:
                    service = BatchStatePersistenceService(test_batch_configuration)
                    service.save_batch_state(batch_evaluation)

                    # Verify Unicode content was saved correctly
                    written_data = mock_file().write.call_args[0][0]
                    json_data = json.loads(written_data)
                    assert json_data["batch_id"] == "unicode_batch_001"
                    assert json_data["correlation_id"] == "unicode_correlation_テスト"

    def test_load_batch_state_unicode_content(self, test_batch_configuration):
        """Test loading batch state with Unicode content."""
        batch_data = {
            "batch_id": "unicode_batch_001",
            "created_at": "2025-01-01T12:00:00",
            "status": "pending",
            "total_documents": 5,
            "correlation_id": "unicode_correlation_テスト",
            "configuration": {
                "max_concurrent_workers": 5,
                "max_retry_attempts": 3,
                "timeout_seconds": 300,
                "progress_update_interval": 10,
                "enable_resumption": True,
                "output_formats": ["json", "summary"],
                "file_size_limit": 52428800,
                "memory_limit_mb": 2048,
                "api_rate_limit_delay": 0.1,
                "retry_delay_seconds": 30,
                "exponential_backoff": True,
            },
        }

        with patch("main.logging.getLogger"):
            with patch("pathlib.Path.mkdir"):
                with patch("pathlib.Path.exists", return_value=True):
                    with patch(
                        "builtins.open",
                        mock_open(read_data=json.dumps(batch_data, ensure_ascii=False)),
                    ):
                        service = BatchStatePersistenceService(test_batch_configuration)
                        result = service.load_batch_state("unicode_batch_001")

                        assert result is not None
                        assert result.batch_id == "unicode_batch_001"
                        assert result.correlation_id == "unicode_correlation_テスト"

    def test_save_batch_state_large_batch(self, test_batch_configuration):
        """Test saving batch state with large batch data."""
        large_batch = BatchEvaluation(
            batch_id="large_batch_001",
            created_at=datetime.now(),
            status=BatchStatus.PROCESSING,
            total_documents=10000,
            correlation_id="large_correlation",
            configuration=test_batch_configuration,
        )

        with patch("main.logging.getLogger"):
            with patch("pathlib.Path.mkdir"):
                with patch("builtins.open", mock_open()) as mock_file:
                    service = BatchStatePersistenceService(test_batch_configuration)
                    service.save_batch_state(large_batch)

                    # Verify large batch was saved
                    written_data = mock_file().write.call_args[0][0]
                    json_data = json.loads(written_data)
                    assert json_data["total_documents"] == 10000
                    assert json_data["batch_id"] == "large_batch_001"
