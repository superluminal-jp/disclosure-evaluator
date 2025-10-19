"""
Service for discovering documents in folders and file systems.
"""

import mimetypes
import logging
from pathlib import Path
from typing import List, Optional
from ...models.batch import DocumentInput


class DocumentDiscoveryService:
    """Service for discovering documents in folders and file systems"""

    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("DocumentDiscoveryService")

    def discover_documents_from_folder(
        self,
        folder_path: str,
        recursive: bool = True,
        file_types: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
        file_size_limit: Optional[int] = None,
    ) -> List[DocumentInput]:
        """Discover documents in a folder with filtering options"""
        try:
            if not folder_path or folder_path is None:
                raise FileNotFoundError(f"Invalid folder path: {folder_path}")

            folder = Path(folder_path)
            if not folder.exists():
                raise FileNotFoundError(f"Folder not found: {folder_path}")

            documents = []
            pattern = "**/*" if recursive else "*"

            for file_path in folder.glob(pattern):
                if file_path.is_file():
                    # Check file size limit
                    if file_size_limit and file_path.stat().st_size > file_size_limit:
                        continue

                    # Check file type filter
                    if file_types:
                        mime_type, _ = mimetypes.guess_type(str(file_path))
                        if not mime_type or mime_type not in file_types:
                            continue

                    # Check exclude patterns
                    if exclude_patterns:
                        if any(
                            file_path.match(pattern) for pattern in exclude_patterns
                        ):
                            continue

                    # Create document input
                    doc_input = DocumentInput(
                        file_path=str(file_path),
                        file_name=file_path.name,
                        context="",
                        output_text="",
                    )
                    documents.append(doc_input)

            self.logger.info(f"Discovered {len(documents)} documents in {folder_path}")
            return documents

        except Exception as e:
            self.logger.error(f"Document discovery failed: {str(e)}")
            raise
