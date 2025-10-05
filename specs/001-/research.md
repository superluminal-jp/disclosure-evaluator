# Research: 複数文書の一括評価

**Feature**: Batch Document Evaluation  
**Date**: 2025-10-04  
**Status**: Complete

## Research Questions and Findings

### 1. Python Concurrent Processing Patterns

**Research Question**: How to implement efficient parallel document processing in Python?

**Decision**: Use `concurrent.futures.ThreadPoolExecutor` with configurable worker limits

**Rationale**:

- ThreadPoolExecutor provides simple, robust parallel execution
- Configurable worker limits prevent resource exhaustion
- Compatible with existing synchronous LLM provider calls
- Built-in error handling and timeout support
- Easy to integrate with existing architecture

**Alternatives Considered**:

- `asyncio`: Overkill for I/O-bound LLM calls, adds complexity
- `multiprocessing`: Memory overhead for document processing, shared state complexity
- `queue.Queue`: Manual thread management, more error-prone

**Implementation Pattern**:

```python
with ThreadPoolExecutor(max_workers=config.batch_concurrency) as executor:
    futures = {executor.submit(process_document, doc): doc for doc in documents}
    for future in as_completed(futures):
        result = future.result()
```

### 2. File System Discovery Patterns

**Research Question**: How to discover and process multiple document types from folders?

**Decision**: Use `pathlib.Path` with recursive glob patterns and MIME type detection

**Rationale**:

- `pathlib` provides cross-platform file operations
- Recursive glob patterns handle nested folder structures
- MIME type detection ensures proper document type handling
- Extensible for new document types
- Memory efficient for large folder structures

**Alternatives Considered**:

- `os.walk()`: More verbose, less type-safe
- `glob.glob()`: Limited pattern matching, no type safety
- Manual file extension checking: Fragile, doesn't handle edge cases

**Implementation Pattern**:

```python
from pathlib import Path
import mimetypes

def discover_documents(folder_path: Path) -> List[DocumentInfo]:
    documents = []
    for file_path in folder_path.rglob("*"):
        if file_path.is_file():
            mime_type, _ = mimetypes.guess_type(str(file_path))
            if is_supported_document(mime_type):
                documents.append(DocumentInfo(file_path, mime_type))
    return documents
```

### 3. State Persistence for Resumable Operations

**Research Question**: How to persist batch processing state for resumption after failures?

**Decision**: JSON-based state files with atomic write operations

**Rationale**:

- JSON provides human-readable state inspection
- Atomic writes prevent corrupted state files
- Simple to implement and debug
- Compatible with existing file-based architecture
- Easy to extend with additional metadata

**Alternatives Considered**:

- SQLite database: Overkill for simple state tracking
- Pickle: Security concerns, version compatibility issues
- In-memory only: No resumption capability
- External state store: Adds complexity, single point of failure

**Implementation Pattern**:

```python
import json
import tempfile
import os

def save_batch_state(batch_id: str, state: BatchState) -> None:
    state_file = get_batch_state_file(batch_id)
    with tempfile.NamedTemporaryFile(mode='w', delete=False, dir=os.path.dirname(state_file)) as tmp:
        json.dump(state.dict(), tmp, indent=2)
        os.rename(tmp.name, state_file)
```

### 4. Error Handling for Parallel Processing

**Research Question**: How to handle partial failures in batch processing without losing progress?

**Decision**: Individual document error tracking with batch-level error aggregation

**Rationale**:

- Preserves successful document evaluations
- Provides detailed error reporting for failed documents
- Allows selective retry of failed documents
- Maintains batch processing integrity
- Enables user decision-making on error handling

**Alternatives Considered**:

- Fail-fast on first error: Loses all progress, poor user experience
- Silent error swallowing: Hides important failure information
- Global error handling: Too coarse-grained for document-level errors

**Implementation Pattern**:

```python
class BatchProcessingResult:
    successful_documents: List[DocumentResult]
    failed_documents: List[DocumentError]
    batch_metadata: BatchMetadata

    def has_failures(self) -> bool:
        return len(self.failed_documents) > 0

    def get_success_rate(self) -> float:
        total = len(self.successful_documents) + len(self.failed_documents)
        return len(self.successful_documents) / total if total > 0 else 0.0
```

### 5. Progress Monitoring and User Feedback

**Research Question**: How to provide real-time progress updates during batch processing?

**Decision**: Callback-based progress reporting with configurable update intervals

**Rationale**:

- Non-blocking progress updates
- Configurable update frequency prevents log spam
- Easy to integrate with existing logging system
- Supports both CLI and future web interface
- Maintains performance during processing

**Alternatives Considered**:

- Polling-based updates: Inefficient, delayed feedback
- Event-driven updates: Complex for simple progress tracking
- No progress updates: Poor user experience for long-running batches

**Implementation Pattern**:

```python
def process_batch_with_progress(documents: List[Document],
                              progress_callback: Callable[[ProgressUpdate], None]) -> BatchResult:
    total_docs = len(documents)
    completed = 0

    for document in documents:
        result = process_document(document)
        completed += 1

        if completed % progress_interval == 0:
            progress_callback(ProgressUpdate(completed, total_docs, result))
```

## Technical Decisions Summary

| Decision                  | Rationale                                | Impact                                 |
| ------------------------- | ---------------------------------------- | -------------------------------------- |
| ThreadPoolExecutor        | Simple, robust parallel processing       | Enables concurrent document evaluation |
| pathlib + MIME detection  | Cross-platform, type-safe file discovery | Supports multiple document formats     |
| JSON state persistence    | Human-readable, atomic operations        | Enables batch resumption               |
| Individual error tracking | Preserves successful results             | Maintains batch integrity              |
| Callback progress updates | Real-time user feedback                  | Improves user experience               |

## Integration Points

### Existing Architecture Compatibility

- **LLM Providers**: Reuses existing provider abstractions
- **Logging**: Extends existing structured logging with batch correlation IDs
- **Configuration**: Leverages existing config management
- **Output Formats**: Maintains existing JSON/summary output patterns

### New Components Required

- **BatchEvaluator**: Orchestrates parallel document processing
- **BatchDocument**: Wraps individual documents with metadata
- **BatchResult**: Aggregates results and error information
- **BatchProgress**: Tracks processing state and progress

### Performance Considerations

- **Memory Usage**: Process documents in batches to prevent memory exhaustion
- **API Rate Limits**: Implement backoff strategies for LLM provider limits
- **Concurrency Limits**: Configurable worker limits based on system resources
- **Timeout Handling**: Graceful timeout handling for long-running evaluations

## Risk Mitigation

### Technical Risks

- **Memory Exhaustion**: Implement document batching and memory monitoring
- **API Rate Limits**: Implement exponential backoff and request queuing
- **Partial Failures**: Comprehensive error tracking and selective retry
- **State Corruption**: Atomic state file operations with backup mechanisms

### Operational Risks

- **User Experience**: Clear progress indicators and error messages
- **Data Loss**: Robust state persistence and recovery mechanisms
- **Performance**: Configurable concurrency limits and resource monitoring
- **Compatibility**: Maintains existing single-document functionality

## Next Steps

1. **Phase 1**: Design data models and API contracts
2. **Phase 2**: Generate implementation tasks
3. **Phase 3**: Implement core batch processing functionality
4. **Phase 4**: Add progress monitoring and error handling
5. **Phase 5**: Integration testing and performance validation

---

**Research Status**: Complete  
**All NEEDS CLARIFICATION resolved**: ✅  
**Ready for Phase 1**: ✅
