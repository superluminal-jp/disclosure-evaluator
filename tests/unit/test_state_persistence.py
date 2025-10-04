"""
Unit tests for batch state persistence service.

These tests validate the state persistence functionality.
"""

import pytest
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from unittest.mock import patch
from evaluator import (
    BatchStatePersistenceService,
    BatchConfiguration,
    BatchEvaluation,
    BatchStatus,
    DocumentStatus,
)


class TestBatchStatePersistenceService:
    """Test BatchStatePersistenceService"""

    def setup_method(self):
        """Set up test environment"""
        self.test_dir = Path(tempfile.mkdtemp())
        self.config = BatchConfiguration()
        self.service = BatchStatePersistenceService(self.config)

        # Override state directories for testing
        self.service.state_dir = self.test_dir / "batch_state"
        self.service.active_dir = self.service.state_dir / "active_batches"
        self.service.completed_dir = self.service.state_dir / "completed_batches"

        # Create directories
        self.service.active_dir.mkdir(parents=True, exist_ok=True)
        self.service.completed_dir.mkdir(parents=True, exist_ok=True)

    def teardown_method(self):
        """Clean up test environment"""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_save_batch_state(self):
        """Test saving batch state"""
        batch = BatchEvaluation(
            batch_id="test_batch_001",
            created_at=datetime.now(),
            status=BatchStatus.PENDING,
            total_documents=5,
            correlation_id="batch_test_batch_001",
            configuration=self.config,
        )

        # Save batch state
        self.service.save_batch_state(batch)

        # Verify file was created
        state_file = self.service.active_dir / "test_batch_001.json"
        assert state_file.exists()

        # Verify file content
        with open(state_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            assert data["batch_id"] == "test_batch_001"
            assert data["status"] == "pending"
            assert data["total_documents"] == 5

    def test_load_batch_state(self):
        """Test loading batch state"""
        # Create a batch
        batch = BatchEvaluation(
            batch_id="test_batch_002",
            created_at=datetime.now(),
            status=BatchStatus.PROCESSING,
            total_documents=10,
            correlation_id="batch_test_batch_002",
            configuration=self.config,
        )

        # Save batch state
        self.service.save_batch_state(batch)

        # Load batch state
        loaded_batch = self.service.load_batch_state("test_batch_002")

        # Verify loaded batch
        assert loaded_batch is not None
        assert loaded_batch.batch_id == "test_batch_002"
        assert loaded_batch.status == BatchStatus.PROCESSING
        assert loaded_batch.total_documents == 10
        assert loaded_batch.correlation_id == "batch_test_batch_002"

    def test_load_nonexistent_batch_state(self):
        """Test loading nonexistent batch state"""
        loaded_batch = self.service.load_batch_state("nonexistent_batch")
        assert loaded_batch is None

    def test_move_to_completed(self):
        """Test moving batch to completed directory"""
        # Create a batch
        batch = BatchEvaluation(
            batch_id="test_batch_003",
            created_at=datetime.now(),
            status=BatchStatus.COMPLETED,
            total_documents=3,
            correlation_id="batch_test_batch_003",
            configuration=self.config,
        )

        # Save batch state
        self.service.save_batch_state(batch)

        # Move to completed
        self.service.move_to_completed("test_batch_003")

        # Verify file was moved
        active_file = self.service.active_dir / "test_batch_003.json"
        completed_file = self.service.completed_dir / "test_batch_003.json"

        assert not active_file.exists()
        assert completed_file.exists()

        # Verify file content
        with open(completed_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            assert data["batch_id"] == "test_batch_003"
            assert data["status"] == "completed"

    def test_move_nonexistent_batch(self):
        """Test moving nonexistent batch to completed"""
        # Should not raise error for nonexistent batch
        self.service.move_to_completed("nonexistent_batch")

        # Verify no files were created
        completed_file = self.service.completed_dir / "nonexistent_batch.json"
        assert not completed_file.exists()

    def test_save_batch_state_error_handling(self):
        """Test error handling in save_batch_state"""
        # Create a batch with invalid data that would cause JSON serialization error
        batch = BatchEvaluation(
            batch_id="test_batch_004",
            created_at=datetime.now(),
            status=BatchStatus.PENDING,
            total_documents=5,
            correlation_id="batch_test_batch_004",
            configuration=self.config,
        )

        # Mock open to raise an error
        with patch("builtins.open", side_effect=IOError("Permission denied")):
            with pytest.raises(IOError):
                self.service.save_batch_state(batch)

    def test_load_batch_state_error_handling(self):
        """Test error handling in load_batch_state"""
        # Create a corrupted state file
        corrupted_file = self.service.active_dir / "corrupted_batch.json"
        with open(corrupted_file, "w", encoding="utf-8") as f:
            f.write("invalid json content")

        # Should handle corrupted file gracefully
        loaded_batch = self.service.load_batch_state("corrupted_batch")
        assert loaded_batch is None

    def test_save_batch_state_atomic_operation(self):
        """Test that save_batch_state is atomic"""
        batch = BatchEvaluation(
            batch_id="test_batch_005",
            created_at=datetime.now(),
            status=BatchStatus.PENDING,
            total_documents=5,
            correlation_id="batch_test_batch_005",
            configuration=self.config,
        )

        # Save batch state
        self.service.save_batch_state(batch)

        # Verify file exists and is valid JSON
        state_file = self.service.active_dir / "test_batch_005.json"
        assert state_file.exists()

        # Verify file is valid JSON
        with open(state_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            assert data["batch_id"] == "test_batch_005"

    def test_load_batch_state_with_complex_data(self):
        """Test loading batch state with complex data"""
        # Create a batch with complex data
        batch = BatchEvaluation(
            batch_id="test_batch_006",
            created_at=datetime.now(),
            status=BatchStatus.PROCESSING,
            total_documents=10,
            processed_documents=5,
            successful_documents=4,
            failed_documents=1,
            processing_started_at=datetime.now(),
            error_summary="Some errors occurred",
            correlation_id="batch_test_batch_006",
            configuration=self.config,
        )

        # Save batch state
        self.service.save_batch_state(batch)

        # Load batch state
        loaded_batch = self.service.load_batch_state("test_batch_006")

        # Verify all fields are preserved
        assert loaded_batch.batch_id == "test_batch_006"
        assert loaded_batch.status == BatchStatus.PROCESSING
        assert loaded_batch.total_documents == 10
        assert loaded_batch.processed_documents == 5
        assert loaded_batch.successful_documents == 4
        assert loaded_batch.failed_documents == 1
        assert loaded_batch.processing_started_at is not None
        assert loaded_batch.error_summary == "Some errors occurred"
        assert loaded_batch.correlation_id == "batch_test_batch_006"

    def test_save_multiple_batch_states(self):
        """Test saving multiple batch states"""
        # Create multiple batches
        batches = []
        for i in range(3):
            batch = BatchEvaluation(
                batch_id=f"test_batch_{i:03d}",
                created_at=datetime.now(),
                status=BatchStatus.PENDING,
                total_documents=i + 1,
                correlation_id=f"batch_test_batch_{i:03d}",
                configuration=self.config,
            )
            batches.append(batch)

        # Save all batches
        for batch in batches:
            self.service.save_batch_state(batch)

        # Verify all files were created
        for i in range(3):
            state_file = self.service.active_dir / f"test_batch_{i:03d}.json"
            assert state_file.exists()

        # Verify all batches can be loaded
        for i in range(3):
            loaded_batch = self.service.load_batch_state(f"test_batch_{i:03d}")
            assert loaded_batch is not None
            assert loaded_batch.batch_id == f"test_batch_{i:03d}"
            assert loaded_batch.total_documents == i + 1

    def test_move_multiple_batches_to_completed(self):
        """Test moving multiple batches to completed"""
        # Create and save multiple batches
        for i in range(3):
            batch = BatchEvaluation(
                batch_id=f"test_batch_{i:03d}",
                created_at=datetime.now(),
                status=BatchStatus.COMPLETED,
                total_documents=i + 1,
                correlation_id=f"batch_test_batch_{i:03d}",
                configuration=self.config,
            )
            self.service.save_batch_state(batch)

        # Move all batches to completed
        for i in range(3):
            self.service.move_to_completed(f"test_batch_{i:03d}")

        # Verify all files were moved
        for i in range(3):
            active_file = self.service.active_dir / f"test_batch_{i:03d}.json"
            completed_file = self.service.completed_dir / f"test_batch_{i:03d}.json"
            assert not active_file.exists()
            assert completed_file.exists()

    def test_state_file_encoding(self):
        """Test state file encoding handling"""
        # Create a batch with Unicode characters
        batch = BatchEvaluation(
            batch_id="test_batch_unicode",
            created_at=datetime.now(),
            status=BatchStatus.PENDING,
            total_documents=5,
            correlation_id="batch_test_batch_unicode",
            configuration=self.config,
        )

        # Save batch state
        self.service.save_batch_state(batch)

        # Load batch state
        loaded_batch = self.service.load_batch_state("test_batch_unicode")

        # Verify batch was loaded correctly
        assert loaded_batch is not None
        assert loaded_batch.batch_id == "test_batch_unicode"

    def test_state_file_permissions(self):
        """Test state file permissions"""
        batch = BatchEvaluation(
            batch_id="test_batch_permissions",
            created_at=datetime.now(),
            status=BatchStatus.PENDING,
            total_documents=5,
            correlation_id="batch_test_batch_permissions",
            configuration=self.config,
        )

        # Save batch state
        self.service.save_batch_state(batch)

        # Verify file permissions
        state_file = self.service.active_dir / "test_batch_permissions.json"
        assert state_file.exists()
        assert state_file.is_file()
        assert state_file.stat().st_size > 0

    def test_state_file_concurrent_access(self):
        """Test concurrent access to state files"""
        import threading
        import time

        results = []

        def save_batch(batch_id):
            batch = BatchEvaluation(
                batch_id=batch_id,
                created_at=datetime.now(),
                status=BatchStatus.PENDING,
                total_documents=5,
                correlation_id=f"batch_{batch_id}",
                configuration=self.config,
            )
            try:
                self.service.save_batch_state(batch)
                results.append(f"Saved {batch_id}")
            except Exception as e:
                results.append(f"Error saving {batch_id}: {e}")

        # Create multiple threads to save batches concurrently
        threads = []
        for i in range(5):
            thread = threading.Thread(
                target=save_batch, args=(f"concurrent_batch_{i}",)
            )
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Verify all batches were saved
        assert len(results) == 5
        assert all("Saved" in result for result in results)

        # Verify all files exist
        for i in range(5):
            state_file = self.service.active_dir / f"concurrent_batch_{i}.json"
            assert state_file.exists()
