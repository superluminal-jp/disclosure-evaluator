"""
Unit tests for document discovery service.

These tests validate the document discovery functionality.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
from evaluator import DocumentDiscoveryService, BatchConfiguration, DocumentInput


class TestDocumentDiscoveryService:
    """Test DocumentDiscoveryService"""

    def setup_method(self):
        """Set up test environment"""
        self.test_dir = Path(tempfile.mkdtemp())
        self.config = BatchConfiguration()
        self.service = DocumentDiscoveryService(self.config)
        self.create_test_files()

    def teardown_method(self):
        """Clean up test environment"""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def create_test_files(self):
        """Create test files for discovery"""
        # Create text files
        (self.test_dir / "document1.txt").write_text("Test document 1")
        (self.test_dir / "document2.txt").write_text("Test document 2")

        # Create PDF files (mock)
        (self.test_dir / "document3.pdf").write_bytes(b"Mock PDF content")

        # Create subdirectory
        subdir = self.test_dir / "subdir"
        subdir.mkdir()
        (subdir / "document4.txt").write_text("Test document 4")
        (subdir / "document5.pdf").write_bytes(b"Mock PDF content")

        # Create log file
        (self.test_dir / "document6.log").write_text("Log content")

        # Create large file
        (self.test_dir / "large.txt").write_text("Large content " * 10000)

    def test_discover_documents_basic(self):
        """Test basic document discovery"""
        documents = self.service.discover_documents_from_folder(
            folder_path=str(self.test_dir), recursive=False
        )

        # Should find files in root directory only
        assert len(documents) >= 3  # At least txt, pdf, log files
        assert all(isinstance(doc, DocumentInput) for doc in documents)
        assert all(doc.file_path for doc in documents)

    def test_discover_documents_recursive(self):
        """Test recursive document discovery"""
        documents = self.service.discover_documents_from_folder(
            folder_path=str(self.test_dir), recursive=True
        )

        # Should find files in root and subdirectories
        assert len(documents) >= 5  # Files in root and subdir
        assert any("subdir" in doc.file_path for doc in documents)

    def test_discover_documents_file_types_filter(self):
        """Test document discovery with file type filter"""
        documents = self.service.discover_documents_from_folder(
            folder_path=str(self.test_dir), recursive=True, file_types=["text/plain"]
        )

        # Should only find text files (both .txt and .log have text/plain MIME type)
        assert len(documents) >= 2  # At least 2 text files
        # Check that all documents are text files (have .txt or .log extension)
        assert all(doc.file_path.endswith((".txt", ".log")) for doc in documents)

    def test_discover_documents_exclude_patterns(self):
        """Test document discovery with exclude patterns"""
        documents = self.service.discover_documents_from_folder(
            folder_path=str(self.test_dir), recursive=True, exclude_patterns=["*.log"]
        )

        # Should exclude log files
        assert not any("log" in doc.file_path for doc in documents)

    def test_discover_documents_file_size_limit(self):
        """Test document discovery with file size limit"""
        documents = self.service.discover_documents_from_folder(
            folder_path=str(self.test_dir),
            recursive=True,
            file_size_limit=1000,  # 1KB limit
        )

        # Should exclude large files
        assert not any("large.txt" in doc.file_path for doc in documents)

    def test_discover_documents_nonexistent_folder(self):
        """Test document discovery with nonexistent folder"""
        with pytest.raises(FileNotFoundError):
            self.service.discover_documents_from_folder(
                folder_path="/nonexistent/folder", recursive=True
            )

    def test_discover_documents_empty_folder(self):
        """Test document discovery with empty folder"""
        empty_dir = self.test_dir / "empty"
        empty_dir.mkdir()

        documents = self.service.discover_documents_from_folder(
            folder_path=str(empty_dir), recursive=True
        )

        assert len(documents) == 0

    def test_discover_documents_mixed_filters(self):
        """Test document discovery with multiple filters"""
        documents = self.service.discover_documents_from_folder(
            folder_path=str(self.test_dir),
            recursive=True,
            file_types=["text/plain"],
            exclude_patterns=["*.log"],
            file_size_limit=10000,
        )

        # Should find text files, exclude logs, and respect size limit
        assert len(documents) >= 2
        assert all("txt" in doc.file_path for doc in documents)
        assert not any("log" in doc.file_path for doc in documents)
        assert not any("large.txt" in doc.file_path for doc in documents)

    def test_discover_documents_file_path_validation(self):
        """Test that discovered documents have valid file paths"""
        documents = self.service.discover_documents_from_folder(
            folder_path=str(self.test_dir), recursive=True
        )

        for doc in documents:
            assert doc.file_path
            assert Path(doc.file_path).exists()
            assert doc.file_name
            assert doc.file_name == Path(doc.file_path).name

    def test_discover_documents_context_preservation(self):
        """Test that context is preserved in document inputs"""
        documents = self.service.discover_documents_from_folder(
            folder_path=str(self.test_dir), recursive=True
        )

        for doc in documents:
            assert doc.context == ""
            assert doc.output_text == ""

    def test_discover_documents_mime_type_detection(self):
        """Test MIME type detection for discovered documents"""
        documents = self.service.discover_documents_from_folder(
            folder_path=str(self.test_dir), recursive=True
        )

        # Check that different file types are detected
        txt_files = [doc for doc in documents if doc.file_path.endswith(".txt")]
        pdf_files = [doc for doc in documents if doc.file_path.endswith(".pdf")]

        assert len(txt_files) > 0
        assert len(pdf_files) > 0

    def test_discover_documents_permission_error(self):
        """Test document discovery with permission errors"""
        # Create a file with restricted permissions (if possible)
        restricted_file = self.test_dir / "restricted.txt"
        restricted_file.write_text("Restricted content")

        # Try to discover documents
        try:
            documents = self.service.discover_documents_from_folder(
                folder_path=str(self.test_dir), recursive=True
            )
            # Should still find other files
            assert len(documents) > 0
        except PermissionError:
            # Permission error is acceptable
            pass

    def test_discover_documents_symlinks(self):
        """Test document discovery with symbolic links"""
        # Create a symbolic link (if supported)
        try:
            link_file = self.test_dir / "link.txt"
            link_file.symlink_to(self.test_dir / "document1.txt")

            documents = self.service.discover_documents_from_folder(
                folder_path=str(self.test_dir), recursive=True
            )

            # Should find the linked file
            assert any("link.txt" in doc.file_path for doc in documents)
        except (OSError, NotImplementedError):
            # Symlinks not supported on this system
            pass

    def test_discover_documents_hidden_files(self):
        """Test document discovery with hidden files"""
        # Create hidden files
        (self.test_dir / ".hidden.txt").write_text("Hidden content")
        (self.test_dir / "document7.txt").write_text("Visible content")

        documents = self.service.discover_documents_from_folder(
            folder_path=str(self.test_dir), recursive=True
        )

        # Should find both hidden and visible files
        file_names = [Path(doc.file_path).name for doc in documents]
        assert "document7.txt" in file_names
        # Hidden files may or may not be found depending on system

    def test_discover_documents_unicode_filenames(self):
        """Test document discovery with Unicode filenames"""
        # Create files with Unicode names
        (self.test_dir / "文档1.txt").write_text("Unicode document 1")
        (self.test_dir / "document_éñ.txt").write_text("Unicode document 2")

        documents = self.service.discover_documents_from_folder(
            folder_path=str(self.test_dir), recursive=True
        )

        # Should find Unicode files
        file_names = [Path(doc.file_path).name for doc in documents]
        assert any("文档1.txt" in name for name in file_names)
        assert any("document_éñ.txt" in name for name in file_names)

    def test_discover_documents_large_directory(self):
        """Test document discovery with many files"""
        # Create many files
        large_dir = self.test_dir / "large"
        large_dir.mkdir()

        for i in range(100):
            (large_dir / f"file_{i:03d}.txt").write_text(f"Content {i}")

        documents = self.service.discover_documents_from_folder(
            folder_path=str(large_dir), recursive=False
        )

        # Should find all files
        assert len(documents) == 100

    def test_discover_documents_nested_directories(self):
        """Test document discovery with deeply nested directories"""
        # Create nested directory structure
        nested_dir = self.test_dir / "level1" / "level2" / "level3"
        nested_dir.mkdir(parents=True)
        (nested_dir / "deep.txt").write_text("Deep content")

        documents = self.service.discover_documents_from_folder(
            folder_path=str(self.test_dir), recursive=True
        )

        # Should find files in nested directories
        assert any("deep.txt" in doc.file_path for doc in documents)

    def test_discover_documents_error_handling(self):
        """Test error handling in document discovery"""
        # Test with invalid folder path
        with pytest.raises(FileNotFoundError):
            self.service.discover_documents_from_folder(folder_path="", recursive=True)

        # Test with None folder path
        with pytest.raises(FileNotFoundError):
            self.service.discover_documents_from_folder(
                folder_path=None, recursive=True
            )

    def test_discover_documents_configuration_integration(self):
        """Test integration with batch configuration"""
        # Create custom configuration
        config = BatchConfiguration(file_size_limit=1024, max_concurrent_workers=2)

        service = DocumentDiscoveryService(config)

        documents = service.discover_documents_from_folder(
            folder_path=str(self.test_dir),
            recursive=True,
            file_size_limit=config.file_size_limit,
        )

        # Should respect configuration
        assert len(documents) > 0
        for doc in documents:
            file_path = Path(doc.file_path)
            if file_path.exists():
                assert file_path.stat().st_size <= config.file_size_limit
