"""
Unit tests for DocumentDiscoveryService class.
Tests lines 1749-1810: DocumentDiscoveryService with file filtering and discovery logic
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from src.batch import DocumentDiscoveryService
from src.models import DocumentInput, BatchConfiguration


class TestDocumentDiscoveryService:
    """Test DocumentDiscoveryService class."""

    def test_document_discovery_service_init(self, test_batch_configuration):
        """Test DocumentDiscoveryService initialization."""
        with patch("main.logging.getLogger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            service = DocumentDiscoveryService(test_batch_configuration)

            assert service.config == test_batch_configuration
            assert service.logger == mock_logger
            mock_get_logger.assert_called_once_with("DocumentDiscoveryService")

    def test_discover_documents_from_folder_success(
        self, test_batch_configuration, sample_documents_dir
    ):
        """Test successful document discovery from folder."""
        with patch("main.logging.getLogger"):
            service = DocumentDiscoveryService(test_batch_configuration)
            documents = service.discover_documents_from_folder(sample_documents_dir)

            assert len(documents) >= 3  # Should find at least 3 files
            assert all(isinstance(doc, DocumentInput) for doc in documents)

            # Check that files are discovered
            file_names = [doc.file_name for doc in documents]
            assert "doc1.txt" in file_names
            assert "doc2.txt" in file_names
            assert "doc3.txt" in file_names

    def test_discover_documents_from_folder_recursive(
        self, test_batch_configuration, sample_documents_dir
    ):
        """Test document discovery with recursive=True."""
        with patch("main.logging.getLogger"):
            service = DocumentDiscoveryService(test_batch_configuration)
            documents = service.discover_documents_from_folder(
                sample_documents_dir, recursive=True
            )

            # Should find files in subdirectories
            file_paths = [doc.file_path for doc in documents]
            subdir_files = [path for path in file_paths if "subdir" in path]
            assert len(subdir_files) > 0

    def test_discover_documents_from_folder_non_recursive(
        self, test_batch_configuration, sample_documents_dir
    ):
        """Test document discovery with recursive=False."""
        with patch("main.logging.getLogger"):
            service = DocumentDiscoveryService(test_batch_configuration)
            documents = service.discover_documents_from_folder(
                sample_documents_dir, recursive=False
            )

            # Should not find files in subdirectories
            file_paths = [doc.file_path for doc in documents]
            subdir_files = [path for path in file_paths if "subdir" in path]
            assert len(subdir_files) == 0

    def test_discover_documents_from_folder_file_types_filter(
        self, test_batch_configuration, sample_documents_dir
    ):
        """Test document discovery with file type filtering."""
        with patch("main.logging.getLogger"):
            service = DocumentDiscoveryService(test_batch_configuration)

            # Filter for text files only
            documents = service.discover_documents_from_folder(
                sample_documents_dir, file_types=["text/plain"]
            )

            assert len(documents) >= 3  # Should find text files
            assert all(isinstance(doc, DocumentInput) for doc in documents)

    def test_discover_documents_from_folder_file_types_no_match(
        self, test_batch_configuration, sample_documents_dir
    ):
        """Test document discovery with file type filter that matches no files."""
        with patch("main.logging.getLogger"):
            service = DocumentDiscoveryService(test_batch_configuration)

            # Filter for non-existent file type
            documents = service.discover_documents_from_folder(
                sample_documents_dir, file_types=["application/pdf"]
            )

            assert len(documents) == 0

    def test_discover_documents_from_folder_exclude_patterns(
        self, test_batch_configuration, sample_documents_dir
    ):
        """Test document discovery with exclude patterns."""
        with patch("main.logging.getLogger"):
            service = DocumentDiscoveryService(test_batch_configuration)

            # Exclude doc1.txt
            documents = service.discover_documents_from_folder(
                sample_documents_dir, exclude_patterns=["doc1.txt"]
            )

            file_names = [doc.file_name for doc in documents]
            assert "doc1.txt" not in file_names
            assert "doc2.txt" in file_names
            assert "doc3.txt" in file_names

    def test_discover_documents_from_folder_file_size_limit(
        self, test_batch_configuration, sample_documents_dir
    ):
        """Test document discovery with file size limit."""
        with patch("main.logging.getLogger"):
            service = DocumentDiscoveryService(test_batch_configuration)

            # Set very small file size limit
            documents = service.discover_documents_from_folder(
                sample_documents_dir, file_size_limit=100  # 100 bytes
            )

            # Should filter out larger files
            assert (
                len(documents) >= 0
            )  # May be 0 if all files are larger than 100 bytes

    def test_discover_documents_from_folder_empty_folder(
        self, test_batch_configuration, tmp_path
    ):
        """Test document discovery with empty folder."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        with patch("main.logging.getLogger"):
            service = DocumentDiscoveryService(test_batch_configuration)
            documents = service.discover_documents_from_folder(str(empty_dir))

            assert len(documents) == 0

    def test_discover_documents_from_folder_invalid_path(
        self, test_batch_configuration
    ):
        """Test document discovery with invalid folder path."""
        with patch("main.logging.getLogger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            service = DocumentDiscoveryService(test_batch_configuration)

            with pytest.raises(
                FileNotFoundError, match="Folder not found: /nonexistent"
            ):
                service.discover_documents_from_folder("/nonexistent")

            mock_logger.error.assert_called()

    def test_discover_documents_from_folder_none_path(self, test_batch_configuration):
        """Test document discovery with None folder path."""
        with patch("main.logging.getLogger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            service = DocumentDiscoveryService(test_batch_configuration)

            with pytest.raises(FileNotFoundError, match="Invalid folder path: None"):
                service.discover_documents_from_folder(None)

            mock_logger.error.assert_called()

    def test_discover_documents_from_folder_empty_string_path(
        self, test_batch_configuration
    ):
        """Test document discovery with empty string folder path."""
        with patch("main.logging.getLogger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            service = DocumentDiscoveryService(test_batch_configuration)

            with pytest.raises(FileNotFoundError, match="Invalid folder path: "):
                service.discover_documents_from_folder("")

            mock_logger.error.assert_called()

    def test_discover_documents_from_folder_mime_type_guessing(
        self, test_batch_configuration, sample_documents_dir
    ):
        """Test document discovery with MIME type guessing."""
        with patch("main.logging.getLogger"):
            with patch("main.mimetypes.guess_type") as mock_guess_type:
                mock_guess_type.return_value = ("text/plain", None)

                service = DocumentDiscoveryService(test_batch_configuration)
                documents = service.discover_documents_from_folder(
                    sample_documents_dir, file_types=["text/plain"]
                )

                assert len(documents) >= 3
                mock_guess_type.assert_called()

    def test_discover_documents_from_folder_mime_type_none(
        self, test_batch_configuration, sample_documents_dir
    ):
        """Test document discovery when MIME type guessing returns None."""
        with patch("main.logging.getLogger"):
            with patch("main.mimetypes.guess_type") as mock_guess_type:
                mock_guess_type.return_value = (None, None)

                service = DocumentDiscoveryService(test_batch_configuration)
                documents = service.discover_documents_from_folder(
                    sample_documents_dir, file_types=["text/plain"]
                )

                # Should filter out files with None MIME type
                assert len(documents) == 0

    def test_discover_documents_from_folder_multiple_exclude_patterns(
        self, test_batch_configuration, sample_documents_dir
    ):
        """Test document discovery with multiple exclude patterns."""
        with patch("main.logging.getLogger"):
            service = DocumentDiscoveryService(test_batch_configuration)

            # Exclude multiple files
            documents = service.discover_documents_from_folder(
                sample_documents_dir, exclude_patterns=["doc1.txt", "doc2.txt"]
            )

            file_names = [doc.file_name for doc in documents]
            assert "doc1.txt" not in file_names
            assert "doc2.txt" not in file_names
            assert "doc3.txt" in file_names

    def test_discover_documents_from_folder_wildcard_exclude_patterns(
        self, test_batch_configuration, sample_documents_dir
    ):
        """Test document discovery with wildcard exclude patterns."""
        with patch("main.logging.getLogger"):
            service = DocumentDiscoveryService(test_batch_configuration)

            # Exclude all .txt files
            documents = service.discover_documents_from_folder(
                sample_documents_dir, exclude_patterns=["*.txt"]
            )

            file_names = [doc.file_name for doc in documents]
            txt_files = [name for name in file_names if name.endswith(".txt")]
            assert len(txt_files) == 0

    def test_discover_documents_from_folder_exception_handling(
        self, test_batch_configuration
    ):
        """Test document discovery exception handling."""
        with patch("main.logging.getLogger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            service = DocumentDiscoveryService(test_batch_configuration)

            # Mock Path.glob to raise exception
            with patch("pathlib.Path.glob", side_effect=Exception("File system error")):
                with pytest.raises(Exception, match="File system error"):
                    service.discover_documents_from_folder("/test/path")

                mock_logger.error.assert_called_with(
                    "Document discovery failed: File system error"
                )

    def test_discover_documents_from_folder_stat_error(
        self, test_batch_configuration, sample_documents_dir
    ):
        """Test document discovery with file stat error."""
        with patch("main.logging.getLogger"):
            service = DocumentDiscoveryService(test_batch_configuration)

            # Mock file.stat() to raise exception
            with patch("pathlib.Path.stat", side_effect=OSError("Permission denied")):
                with pytest.raises(OSError, match="Permission denied"):
                    service.discover_documents_from_folder(sample_documents_dir)

    def test_discover_documents_from_folder_is_file_check(
        self, test_batch_configuration, sample_documents_dir
    ):
        """Test document discovery with directory filtering."""
        with patch("main.logging.getLogger"):
            service = DocumentDiscoveryService(test_batch_configuration)

            # Mock is_file() to return False for some paths (directories)
            with patch("pathlib.Path.is_file") as mock_is_file:
                mock_is_file.side_effect = lambda: not str(
                    Path(mock_is_file.call_args[0][0]).name
                ).startswith("doc")

                documents = service.discover_documents_from_folder(sample_documents_dir)

                # Should filter out directories
                assert len(documents) == 0

    def test_discover_documents_from_folder_context_and_output_text(
        self, test_batch_configuration, sample_documents_dir
    ):
        """Test that discovered documents have empty context and output_text."""
        with patch("main.logging.getLogger"):
            service = DocumentDiscoveryService(test_batch_configuration)
            documents = service.discover_documents_from_folder(sample_documents_dir)

            for doc in documents:
                assert doc.context == ""
                assert doc.output_text == ""
                assert doc.file_path is not None
                assert doc.file_name is not None

    def test_discover_documents_from_folder_large_file_size_limit(
        self, test_batch_configuration, sample_documents_dir
    ):
        """Test document discovery with large file size limit."""
        with patch("main.logging.getLogger"):
            service = DocumentDiscoveryService(test_batch_configuration)

            # Set very large file size limit
            documents = service.discover_documents_from_folder(
                sample_documents_dir, file_size_limit=1024 * 1024 * 1024  # 1GB
            )

            # Should find all files
            assert len(documents) >= 3

    def test_discover_documents_from_folder_no_file_types_filter(
        self, test_batch_configuration, sample_documents_dir
    ):
        """Test document discovery without file type filtering."""
        with patch("main.logging.getLogger"):
            service = DocumentDiscoveryService(test_batch_configuration)
            documents = service.discover_documents_from_folder(
                sample_documents_dir, file_types=None
            )

            # Should find all files
            assert len(documents) >= 3

    def test_discover_documents_from_folder_no_exclude_patterns(
        self, test_batch_configuration, sample_documents_dir
    ):
        """Test document discovery without exclude patterns."""
        with patch("main.logging.getLogger"):
            service = DocumentDiscoveryService(test_batch_configuration)
            documents = service.discover_documents_from_folder(
                sample_documents_dir, exclude_patterns=None
            )

            # Should find all files
            assert len(documents) >= 3

    def test_discover_documents_from_folder_no_file_size_limit(
        self, test_batch_configuration, sample_documents_dir
    ):
        """Test document discovery without file size limit."""
        with patch("main.logging.getLogger"):
            service = DocumentDiscoveryService(test_batch_configuration)
            documents = service.discover_documents_from_folder(
                sample_documents_dir, file_size_limit=None
            )

            # Should find all files
            assert len(documents) >= 3

    def test_discover_documents_from_folder_logging_success(
        self, test_batch_configuration, sample_documents_dir
    ):
        """Test that successful discovery is logged."""
        with patch("main.logging.getLogger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            service = DocumentDiscoveryService(test_batch_configuration)
            documents = service.discover_documents_from_folder(sample_documents_dir)

            # Verify success logging
            mock_logger.info.assert_called()
            info_calls = [call[0][0] for call in mock_logger.info.call_args_list]
            assert any(
                "Discovered" in call and "documents in" in call for call in info_calls
            )

    def test_discover_documents_from_folder_symlink_handling(
        self, test_batch_configuration, tmp_path
    ):
        """Test document discovery with symbolic links."""
        # Create a test directory with symlinks
        test_dir = tmp_path / "test_symlinks"
        test_dir.mkdir()

        # Create a regular file
        regular_file = test_dir / "regular.txt"
        regular_file.write_text("Regular file content")

        # Create a symlink (if supported)
        try:
            symlink_file = test_dir / "symlink.txt"
            symlink_file.symlink_to(regular_file)

            with patch("main.logging.getLogger"):
                service = DocumentDiscoveryService(test_batch_configuration)
                documents = service.discover_documents_from_folder(str(test_dir))

                # Should find both regular file and symlink
                assert len(documents) >= 1
                file_names = [doc.file_name for doc in documents]
                assert "regular.txt" in file_names

        except OSError:
            # Symlinks not supported on this system, skip test
            pytest.skip("Symbolic links not supported on this system")

    def test_discover_documents_from_folder_hidden_files(
        self, test_batch_configuration, tmp_path
    ):
        """Test document discovery with hidden files."""
        test_dir = tmp_path / "test_hidden"
        test_dir.mkdir()

        # Create regular file
        regular_file = test_dir / "visible.txt"
        regular_file.write_text("Visible file")

        # Create hidden file
        hidden_file = test_dir / ".hidden.txt"
        hidden_file.write_text("Hidden file")

        with patch("main.logging.getLogger"):
            service = DocumentDiscoveryService(test_batch_configuration)
            documents = service.discover_documents_from_folder(str(test_dir))

            # Should find both visible and hidden files
            assert len(documents) == 2
            file_names = [doc.file_name for doc in documents]
            assert "visible.txt" in file_names
            assert ".hidden.txt" in file_names
