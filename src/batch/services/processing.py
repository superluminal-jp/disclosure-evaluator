"""
Service for parallel document processing.
"""

import os
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Callable

from ...models.batch import BatchDocument, DocumentStatus
from ...llm import LLMProvider
from ...evaluators import DisclosureEvaluator


class ParallelDocumentProcessingService:
    """Service for parallel document processing"""

    def __init__(self, config, llm_provider: LLMProvider, config_manager=None):
        self.config = config
        self.llm_provider = llm_provider
        self.config_manager = config_manager
        self.logger = logging.getLogger("ParallelDocumentProcessingService")

    def process_documents_parallel(
        self,
        documents: List[BatchDocument],
        progress_callback: Optional[Callable] = None,
    ) -> List[BatchDocument]:
        """Process documents in parallel with progress tracking"""
        try:
            results = []

            with ThreadPoolExecutor(
                max_workers=self.config.max_concurrent_workers
            ) as executor:
                # Submit all documents for processing
                future_to_doc = {
                    executor.submit(self._process_single_document, doc): doc
                    for doc in documents
                }

                # Process completed futures
                for future in as_completed(future_to_doc):
                    doc = future_to_doc[future]
                    try:
                        result = future.result(timeout=self.config.timeout_seconds)
                        results.append(result)

                        # Call progress callback if provided
                        if progress_callback:
                            progress_callback(result)

                    except TimeoutError:
                        self.logger.error(
                            f"Document processing timed out: {doc.document_id}"
                        )
                        doc.status = DocumentStatus.FAILED
                        doc.error_message = f"Processing timed out after {self.config.timeout_seconds} seconds"
                        doc.processing_completed_at = datetime.now()
                        results.append(doc)
                    except Exception as e:
                        self.logger.error(f"Document processing failed: {str(e)}")
                        # Mark document as failed
                        doc.status = DocumentStatus.FAILED
                        doc.error_message = str(e)
                        doc.processing_completed_at = datetime.now()
                        results.append(doc)

            return results

        except Exception as e:
            self.logger.error(f"Parallel processing failed: {str(e)}")
            raise

    def _process_single_document(self, doc: BatchDocument) -> BatchDocument:
        """Process a single document with retry logic"""
        doc.status = DocumentStatus.PROCESSING
        doc.processing_started_at = datetime.now()

        # Read document content
        with open(doc.file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Retry logic
        for attempt in range(self.config.max_retry_attempts + 1):
            try:
                # Create evaluator for this document
                evaluator = DisclosureEvaluator(config_manager=self.config_manager)

                # Evaluate document (timeout is handled by the ThreadPoolExecutor)
                result = evaluator.evaluate_disclosure(
                    input_text=content,
                    context=doc.context or "",
                    output_text=doc.output_text or "",
                )

                # Update document with results
                doc.evaluation_result = result
                doc.status = DocumentStatus.COMPLETED
                doc.processing_completed_at = datetime.now()
                doc.retry_count = attempt

                return doc

            except Exception as e:
                doc.retry_count = attempt
                if attempt == self.config.max_retry_attempts:
                    # Final attempt failed
                    self.logger.error(
                        f"Document processing failed after {attempt + 1} attempts: {str(e)}"
                    )
                    doc.status = DocumentStatus.FAILED
                    doc.error_message = str(e)
                    doc.processing_completed_at = datetime.now()
                    return doc
                else:
                    # Retry
                    self.logger.warning(
                        f"Document processing attempt {attempt + 1} failed, retrying: {str(e)}"
                    )
                    continue

        # This should never be reached, but just in case
        doc.status = DocumentStatus.FAILED
        doc.error_message = "Max retry attempts exceeded"
        doc.processing_completed_at = datetime.now()
        return doc
