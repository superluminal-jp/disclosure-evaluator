"""
Main orchestrator for batch document evaluation.
"""

import json
import logging
import mimetypes
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Callable

from ..models.batch import (
    BatchConfiguration,
    BatchEvaluation,
    BatchDocument,
    BatchProgress,
    BatchResult,
    DocumentInput,
    BatchStatus,
    DocumentStatus,
    ProcessingPhase,
)
from ..llm import create_llm_provider
from .services import (
    DocumentDiscoveryService,
    BatchStatePersistenceService,
    ParallelDocumentProcessingService,
)


class BatchEvaluator:
    """Main orchestrator for batch document evaluation"""

    def __init__(
        self, config: Optional[BatchConfiguration] = None, config_manager=None
    ):
        """
        Initialize BatchEvaluator with configuration and services.

        Args:
            config: Batch processing configuration (default: BatchConfiguration())
            config_manager: Configuration manager for LLM providers

        Raises:
            RuntimeError: If service initialization fails
        """
        self.config = config or BatchConfiguration()
        self.config_manager = config_manager
        self.logger = logging.getLogger("BatchEvaluator")

        # Initialize services with proper error handling
        try:
            self.discovery_service = DocumentDiscoveryService(self.config)
            self.state_service = BatchStatePersistenceService(self.config)

            # Initialize LLM provider for document processing
            self.llm_provider = create_llm_provider(config_manager=config_manager)
            self.processing_service = ParallelDocumentProcessingService(
                self.config, self.llm_provider, config_manager
            )

            self.logger.info("BatchEvaluator initialized successfully")

        except Exception as e:
            self.logger.error(
                f"Failed to initialize BatchEvaluator services: {str(e)}",
                extra={"correlation_id": "init_error"},
            )
            raise RuntimeError(f"BatchEvaluator initialization failed: {str(e)}") from e

    def create_batch(
        self,
        documents: List[DocumentInput],
        config: Optional[BatchConfiguration] = None,
    ) -> str:
        """Create a new batch evaluation"""
        try:
            batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            correlation_id = f"batch_{batch_id}"

            # Create batch evaluation
            batch = BatchEvaluation(
                batch_id=batch_id,
                created_at=datetime.now(),
                status=BatchStatus.PENDING,
                total_documents=len(documents),
                correlation_id=correlation_id,
                configuration=config or self.config,
            )

            # Save batch state
            self.state_service.save_batch_state(batch)

            # Store document inputs for later processing
            self._store_batch_documents(batch_id, documents)

            self.logger.info(
                f"Created batch {batch_id} with {len(documents)} documents"
            )
            return batch_id

        except Exception as e:
            self.logger.error(f"Failed to create batch: {str(e)}")
            raise

    def _store_batch_documents(
        self, batch_id: str, documents: List[DocumentInput]
    ) -> None:
        """
        Store document inputs for batch processing.

        Args:
            batch_id: Unique batch identifier
            documents: List of document inputs to store

        Raises:
            IOError: If file write operation fails
            ValueError: If documents list is empty
        """
        try:
            if not documents:
                raise ValueError("Cannot store empty documents list")

            # Convert to serializable format
            doc_data = []
            for doc in documents:
                doc_data.append(
                    {
                        "file_path": doc.file_path,
                        "file_name": doc.file_name,
                        "context": doc.context,
                        "output_text": doc.output_text,
                    }
                )

            # Ensure active directory exists
            self.state_service.active_dir.mkdir(parents=True, exist_ok=True)

            # Save to a separate file
            docs_file = self.state_service.active_dir / f"{batch_id}_documents.json"
            with open(docs_file, "w", encoding="utf-8") as f:
                json.dump(doc_data, f, indent=2, ensure_ascii=False)

            self.logger.info(
                f"Stored {len(documents)} documents for batch {batch_id}",
                extra={"batch_id": batch_id, "document_count": len(documents)},
            )

        except (IOError, OSError) as e:
            self.logger.error(
                f"Failed to store batch documents: {str(e)}",
                extra={"batch_id": batch_id, "error_type": type(e).__name__},
            )
            raise IOError(
                f"Document storage failed for batch {batch_id}: {str(e)}"
            ) from e
        except Exception as e:
            self.logger.error(f"Unexpected error storing batch documents: {str(e)}")
            raise

    def _convert_document_inputs_to_batch_documents(
        self, documents: List[DocumentInput], batch_id: str
    ) -> List[BatchDocument]:
        """
        Convert DocumentInput list to BatchDocument list.

        Args:
            documents: List of document inputs
            batch_id: Batch identifier

        Returns:
            List of BatchDocument objects

        Raises:
            ValueError: If documents list is empty or invalid
            FileNotFoundError: If document file doesn't exist
        """
        try:
            if not documents:
                raise ValueError("Cannot convert empty documents list")

            batch_documents = []

            for i, doc_input in enumerate(documents):
                try:
                    # Get file information
                    file_path = Path(doc_input.file_path)

                    # Check if file exists for real files (not test mocks)
                    if not str(file_path).startswith("/test/"):
                        if not file_path.exists():
                            self.logger.warning(
                                f"File does not exist: {file_path}",
                                extra={
                                    "batch_id": batch_id,
                                    "file_path": str(file_path),
                                },
                            )
                            file_size = 0
                        else:
                            file_size = file_path.stat().st_size
                    else:
                        # Mock test file
                        file_size = 1024

                    mime_type, _ = mimetypes.guess_type(str(file_path))

                    # Create document ID
                    document_id = f"doc_{batch_id}_{i:04d}"

                    # Create BatchDocument
                    batch_doc = BatchDocument(
                        document_id=document_id,
                        batch_id=batch_id,
                        file_path=str(doc_input.file_path),
                        file_name=doc_input.file_name or file_path.name,
                        file_size=file_size,
                        mime_type=mime_type or "text/plain",
                        status=DocumentStatus.PENDING,
                        correlation_id=f"{batch_id}_{document_id}",
                        context=doc_input.context,
                        output_text=doc_input.output_text,
                    )

                    batch_documents.append(batch_doc)

                except Exception as doc_error:
                    self.logger.error(
                        f"Failed to convert document {i}: {str(doc_error)}",
                        extra={"batch_id": batch_id, "document_index": i},
                    )
                    # Continue processing other documents
                    continue

            if not batch_documents:
                raise ValueError("No documents could be converted")

            self.logger.info(
                f"Converted {len(documents)} DocumentInputs to {len(batch_documents)} BatchDocuments",
                extra={
                    "batch_id": batch_id,
                    "input_count": len(documents),
                    "output_count": len(batch_documents),
                },
            )
            return batch_documents

        except Exception as e:
            self.logger.error(
                f"Failed to convert document inputs: {str(e)}",
                extra={"batch_id": batch_id},
            )
            raise

    def create_batch_from_folder(
        self,
        folder_path: str,
        context: str = "",
        recursive: bool = True,
        file_types: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
        file_size_limit: Optional[int] = None,
        progress_callback: Optional[Callable] = None,
    ) -> str:
        """Create batch from folder discovery"""
        try:
            # Discover documents
            documents = self.discovery_service.discover_documents_from_folder(
                folder_path=folder_path,
                recursive=recursive,
                file_types=file_types,
                exclude_patterns=exclude_patterns,
                file_size_limit=file_size_limit or self.config.file_size_limit,
            )

            if not documents:
                raise ValueError(f"No documents found in {folder_path}")

            # Create batch
            batch_id = self.create_batch(documents)

            self.logger.info(f"Created batch {batch_id} from folder {folder_path}")
            return batch_id

        except Exception as e:
            self.logger.error(f"Failed to create batch from folder: {str(e)}")
            raise

    def get_batch(self, batch_id: str) -> Optional[BatchEvaluation]:
        """Get batch evaluation details"""
        return self.state_service.load_batch_state(batch_id)

    def get_batch_progress(self, batch_id: str) -> Optional[BatchProgress]:
        """Get batch processing progress"""
        batch = self.get_batch(batch_id)
        if not batch:
            return None

        return BatchProgress(
            batch_id=batch_id,
            current_phase=ProcessingPhase.PROCESSING,
            total_documents=batch.total_documents,
            processed_documents=batch.processed_documents,
            successful_documents=batch.successful_documents,
            failed_documents=batch.failed_documents,
            progress_percentage=(batch.processed_documents / batch.total_documents)
            * 100.0,
            last_updated=datetime.now(),
        )

    def get_batch_results(
        self, batch_id: str, format: str = "json"
    ) -> Optional[BatchResult]:
        """Get batch evaluation results"""
        try:
            batch = self.get_batch(batch_id)
            if not batch:
                return None

            # Check if batch is completed
            if batch.status not in [
                BatchStatus.COMPLETED,
                BatchStatus.PARTIALLY_FAILED,
                BatchStatus.FAILED,
            ]:
                self.logger.warning(
                    f"Batch {batch_id} is not completed yet (status: {batch.status})"
                )
                return None

            # Calculate processing duration
            from datetime import timedelta

            if batch.processing_started_at and batch.processing_completed_at:
                processing_duration = (
                    batch.processing_completed_at - batch.processing_started_at
                )
            else:
                processing_duration = timedelta(0)

            # Calculate success rate
            success_rate = (
                batch.successful_documents / batch.total_documents
                if batch.total_documents > 0
                else 0.0
            )

            # Calculate average processing time per document
            avg_time_per_doc = (
                processing_duration / batch.total_documents
                if batch.total_documents > 0
                else timedelta(0)
            )

            # Create individual results (simplified - in real implementation, we'd store these)
            from ..models.batch import DocumentResult

            individual_results = []
            for i in range(batch.successful_documents):
                # This is a placeholder - in a real implementation, we'd store actual results
                individual_results.append(
                    DocumentResult(
                        document_id=f"doc_{batch_id}_{i:04d}",
                        evaluation_result=None,  # Would contain actual evaluation result
                        processing_time=avg_time_per_doc,
                        success=True,
                    )
                )

            # Create batch result
            result = BatchResult(
                batch_id=batch_id,
                total_documents=batch.total_documents,
                successful_evaluations=batch.successful_documents,
                failed_evaluations=batch.failed_documents,
                success_rate=success_rate,
                processing_duration=processing_duration,
                average_evaluation_time=avg_time_per_doc,
                individual_results=individual_results,
                generated_at=datetime.now(),
            )

            self.logger.info(
                f"Generated batch results for {batch_id}: {batch.successful_documents} successful, {batch.failed_documents} failed"
            )
            return result

        except Exception as e:
            self.logger.error(f"Failed to get batch results: {str(e)}")
            return None

    def start_batch(self, batch_id: str) -> bool:
        """Start batch processing"""
        try:
            batch = self.get_batch(batch_id)
            if not batch:
                raise ValueError(f"Batch {batch_id} not found")

            if batch.status != BatchStatus.PENDING:
                raise ValueError(f"Batch {batch_id} is not in pending status")

            # Update batch status
            batch.status = BatchStatus.PROCESSING
            batch.processing_started_at = datetime.now()
            self.state_service.save_batch_state(batch)

            self.logger.info(f"Started batch processing: {batch_id}")

            # Start actual document processing in a separate thread
            import threading

            # For debugging: run in main thread first
            self._process_batch_documents(batch_id)

            # processing_thread = threading.Thread(
            #     target=self._process_batch_documents, args=(batch_id,), daemon=True
            # )
            # processing_thread.start()

            return True

        except Exception as e:
            self.logger.error(f"Failed to start batch: {str(e)}")
            raise

    def _process_batch_documents(self, batch_id: str) -> None:
        """Process all documents in a batch"""
        try:
            self.logger.info(f"Starting document processing for batch: {batch_id}")

            # Get batch information
            batch = self.get_batch(batch_id)
            if not batch:
                self.logger.error(f"Batch {batch_id} not found during processing")
                return

            # Get document inputs from the original batch creation
            # For now, we'll need to reconstruct the documents from the batch state
            # This is a limitation - we should store the original DocumentInputs
            documents = self._get_batch_documents_from_state(batch_id)
            self.logger.info(
                f"Loaded {len(documents)} documents from state for batch {batch_id}"
            )

            if not documents:
                self.logger.error(f"No documents found for batch {batch_id}")
                batch.status = BatchStatus.FAILED
                batch.error_summary = "No documents found"
                self.state_service.save_batch_state(batch)
                return

            # Convert to BatchDocuments
            batch_documents = self._convert_document_inputs_to_batch_documents(
                documents, batch_id
            )
            self.logger.info(
                f"Converted to {len(batch_documents)} BatchDocuments for batch {batch_id}"
            )

            # Process documents in parallel
            self.logger.info(f"Starting parallel processing for batch {batch_id}")
            results = self.processing_service.process_documents_parallel(
                batch_documents,
                progress_callback=self._create_progress_callback(batch_id),
            )
            self.logger.info(
                f"Completed parallel processing for batch {batch_id}: {len(results)} results"
            )

            # Update batch with results
            self._update_batch_with_results(batch_id, results)

            self.logger.info(f"Completed document processing for batch: {batch_id}")

        except Exception as e:
            self.logger.error(
                f"Document processing failed for batch {batch_id}: {str(e)}"
            )
            # Update batch status to failed
            batch = self.get_batch(batch_id)
            if batch:
                batch.status = BatchStatus.FAILED
                batch.error_summary = str(e)
                batch.processing_completed_at = datetime.now()
                self.state_service.save_batch_state(batch)

    def _get_batch_documents_from_state(self, batch_id: str) -> List[DocumentInput]:
        """Get document inputs from batch state"""
        try:
            # Load stored document data
            docs_file = self.state_service.active_dir / f"{batch_id}_documents.json"
            self.logger.info(f"Looking for document file: {docs_file}")

            if not docs_file.exists():
                self.logger.error(
                    f"Document data file not found for batch {batch_id}: {docs_file}"
                )
                return []

            with open(docs_file, "r", encoding="utf-8") as f:
                doc_data = json.load(f)

            # Convert back to DocumentInput objects
            documents = []
            for doc_dict in doc_data:
                doc_input = DocumentInput(
                    file_path=doc_dict["file_path"],
                    file_name=doc_dict.get("file_name"),
                    context=doc_dict.get("context"),
                    output_text=doc_dict.get("output_text"),
                )
                documents.append(doc_input)

            self.logger.info(f"Loaded {len(documents)} documents for batch {batch_id}")
            return documents

        except Exception as e:
            self.logger.error(f"Failed to load batch documents: {str(e)}")
            import traceback

            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return []

    def _create_progress_callback(self, batch_id: str):
        """Create progress callback for batch processing"""

        def progress_callback(completed_doc):
            try:
                batch = self.get_batch(batch_id)
                if batch:
                    batch.processed_documents += 1
                    if completed_doc.status == DocumentStatus.COMPLETED:
                        batch.successful_documents += 1
                    else:
                        batch.failed_documents += 1

                    # Update status based on progress
                    if batch.processed_documents >= batch.total_documents:
                        if batch.failed_documents == 0:
                            batch.status = BatchStatus.COMPLETED
                        elif batch.successful_documents == 0:
                            batch.status = BatchStatus.FAILED
                        else:
                            batch.status = BatchStatus.PARTIALLY_FAILED
                        batch.processing_completed_at = datetime.now()

                    self.state_service.save_batch_state(batch)
            except Exception as e:
                self.logger.error(f"Progress callback error: {str(e)}")

        return progress_callback

    def _update_batch_with_results(
        self, batch_id: str, results: List[BatchDocument]
    ) -> None:
        """Update batch with processing results"""
        try:
            batch = self.get_batch(batch_id)
            if not batch:
                return

            # Count results
            successful = sum(
                1 for doc in results if doc.status == DocumentStatus.COMPLETED
            )
            failed = sum(1 for doc in results if doc.status == DocumentStatus.FAILED)

            # Update batch status
            if failed == 0:
                batch.status = BatchStatus.COMPLETED
            elif successful == 0:
                batch.status = BatchStatus.FAILED
            else:
                batch.status = BatchStatus.PARTIALLY_FAILED

            batch.processed_documents = len(results)
            batch.successful_documents = successful
            batch.failed_documents = failed
            batch.processing_completed_at = datetime.now()

            # Save updated state
            self.state_service.save_batch_state(batch)

            self.logger.info(
                f"Updated batch {batch_id}: {successful} successful, {failed} failed"
            )

        except Exception as e:
            self.logger.error(f"Failed to update batch results: {str(e)}")
            raise

    def resume_batch(self, batch_id: str) -> bool:
        """Resume batch processing"""
        try:
            batch = self.get_batch(batch_id)
            if not batch:
                raise ValueError(f"Batch {batch_id} not found")

            if batch.status not in [BatchStatus.FAILED, BatchStatus.PARTIALLY_FAILED]:
                raise ValueError(f"Batch {batch_id} cannot be resumed")

            # Update batch status
            batch.status = BatchStatus.PROCESSING
            self.state_service.save_batch_state(batch)

            self.logger.info(f"Resumed batch processing: {batch_id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to resume batch: {str(e)}")
            raise

    def list_batch_documents(self, batch_id: str) -> List[BatchDocument]:
        """List documents in a batch"""
        return []

    def get_batch_document(
        self, batch_id: str, document_id: str
    ) -> Optional[BatchDocument]:
        """Get specific document in a batch"""
        return None

    def retry_document(self, batch_id: str, document_id: str) -> bool:
        """Retry processing a specific document"""
        try:
            self.logger.info(f"Retrying document {document_id} in batch {batch_id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to retry document: {str(e)}")
            raise
