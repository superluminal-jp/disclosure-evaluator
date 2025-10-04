# Quickstart: 複数文書の一括評価

**Feature**: Batch Document Evaluation  
**Date**: 2025-01-04  
**Status**: Complete

## Overview

This quickstart guide demonstrates how to use the batch document evaluation feature to process multiple documents simultaneously for disclosure compliance evaluation.

## Prerequisites

- Python 3.12+
- Disclosure Evaluator installed
- LLM API credentials configured (OpenAI, Anthropic, or AWS Bedrock)
- Documents to evaluate (text, PDF, Word, etc.)

## Basic Usage

### 1. Command Line Interface

#### Create a Batch from Individual Documents

```bash
# Create batch with specific documents
python evaluator.py --batch \
  --documents "document1.txt" "document2.pdf" "document3.docx" \
  --output-format json

# Create batch with folder discovery
python evaluator.py --batch \
  --folder "/path/to/documents" \
  --output-format json summary
```

#### Monitor Batch Progress

```bash
# Check batch status
python evaluator.py --batch-status <batch_id>

# List all batches
python evaluator.py --list-batches

# Get batch results
python evaluator.py --batch-results <batch_id> --format json
```

#### Resume Failed Batch

```bash
# Resume interrupted batch
python evaluator.py --resume-batch <batch_id>

# Retry specific failed documents
python evaluator.py --retry-documents <batch_id> --documents <document_ids>
```

### 2. Python API Usage

#### Basic Batch Processing

```python
from disclosure_evaluator import BatchEvaluator, BatchConfiguration

# Configure batch processing
config = BatchConfiguration(
    max_concurrent_workers=5,
    max_retry_attempts=3,
    timeout_seconds=300,
    enable_resumption=True
)

# Create batch evaluator
evaluator = BatchEvaluator(config=config)

# Process documents from folder
batch_id = evaluator.create_batch_from_folder(
    folder_path="/path/to/documents",
    context="Additional context for evaluation"
)

# Start processing
evaluator.start_batch(batch_id)

# Monitor progress
progress = evaluator.get_batch_progress(batch_id)
print(f"Progress: {progress.progress_percentage}%")

# Get results when complete
results = evaluator.get_batch_results(batch_id)
print(f"Success rate: {results.success_rate}")
```

#### Advanced Batch Processing

```python
from disclosure_evaluator import BatchEvaluator, DocumentInput

# Create batch with specific documents
documents = [
    DocumentInput(
        file_path="doc1.txt",
        context="Personal information document",
        output_text="Redacted version"
    ),
    DocumentInput(
        file_path="doc2.pdf",
        context="Financial report"
    )
]

# Create and start batch
batch_id = evaluator.create_batch(documents)
evaluator.start_batch(batch_id)

# Monitor with callback
def progress_callback(progress):
    print(f"Processed {progress.processed_documents}/{progress.total_documents}")

evaluator.monitor_batch(batch_id, callback=progress_callback)

# Get detailed results
results = evaluator.get_batch_results(batch_id)
for result in results.individual_results:
    print(f"Document {result.document_id}: Score {result.evaluation_result.overall_score}")
```

## Configuration Options

### Batch Configuration

```python
config = BatchConfiguration(
    max_concurrent_workers=5,      # Parallel processing limit
    max_retry_attempts=3,          # Retry failed documents
    timeout_seconds=300,           # Per-document timeout
    progress_update_interval=10,   # Progress update frequency
    enable_resumption=True,        # Allow batch resumption
    output_formats=["json", "summary"]  # Output formats
)
```

### Document Discovery

```python
# Supported file types
supported_types = [
    "text/plain",
    "application/pdf",
    "application/msword",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
]

# Folder discovery options
discovery_options = {
    "recursive": True,           # Include subdirectories
    "max_depth": 5,             # Maximum directory depth
    "file_size_limit": 50 * 1024 * 1024,  # 50MB limit
    "exclude_patterns": ["*.tmp", "*.log"]  # Exclude patterns
}
```

## Error Handling

### Common Error Scenarios

#### Document Processing Failures

```python
# Handle individual document failures
results = evaluator.get_batch_results(batch_id)

if results.failed_evaluations > 0:
    print(f"Failed documents: {results.failed_evaluations}")

    for error in results.error_summary:
        print(f"Document {error.document_id}: {error.error_message}")

        # Retry specific document
        evaluator.retry_document(batch_id, error.document_id)
```

#### Batch Processing Failures

```python
# Handle batch-level failures
batch = evaluator.get_batch(batch_id)

if batch.status == "failed":
    print(f"Batch failed: {batch.error_summary}")

    # Resume batch if possible
    if batch.status in ["failed", "partially_failed"]:
        evaluator.resume_batch(batch_id)
```

### Retry Strategies

```python
# Configure retry behavior
config = BatchConfiguration(
    max_retry_attempts=3,
    retry_delay_seconds=30,      # Wait between retries
    exponential_backoff=True     # Increase delay with each retry
)

# Manual retry of failed documents
failed_docs = evaluator.get_failed_documents(batch_id)
for doc in failed_docs:
    evaluator.retry_document(batch_id, doc.document_id)
```

## Output Formats

### JSON Output

```json
{
  "batch_id": "batch_20250104_143022",
  "total_documents": 10,
  "successful_evaluations": 8,
  "failed_evaluations": 2,
  "success_rate": 0.8,
  "processing_duration": "00:02:30",
  "individual_results": [
    {
      "document_id": "doc_001",
      "evaluation_result": {
        "overall_score": 3,
        "criteria_evaluations": [...]
      }
    }
  ]
}
```

### Summary Output

```
# バッチ評価結果

## バッチ情報
- **バッチID**: batch_20250104_143022
- **総文書数**: 10
- **成功**: 8
- **失敗**: 2
- **成功率**: 80%
- **処理時間**: 00:02:30

## 各文書の評価結果

### ✅ document1.txt
- **スコア**: 4/5
- **主要な不開示事由**: 個人情報保護

### ❌ document2.pdf
- **エラー**: ファイル読み込み失敗
- **再試行回数**: 3
```

## Performance Optimization

### Batch Size Guidelines

```python
# Recommended batch sizes
small_batch = 5-10 documents      # Quick processing
medium_batch = 10-50 documents   # Balanced performance
large_batch = 50-100 documents   # Maximum efficiency

# Memory considerations
max_memory_usage = 2 * 1024 * 1024 * 1024  # 2GB limit
documents_per_batch = max_memory_usage // average_document_size
```

### Concurrency Tuning

```python
# Adjust based on system resources
config = BatchConfiguration(
    max_concurrent_workers=min(10, cpu_count * 2),  # CPU-based limit
    timeout_seconds=300,                              # Per-document timeout
    progress_update_interval=5                       # Frequent updates
)
```

## Troubleshooting

### Common Issues

#### Memory Exhaustion

```bash
# Reduce batch size
python evaluator.py --batch --max-documents 20 --documents ...

# Increase system memory
export BATCH_MEMORY_LIMIT=4GB
```

#### API Rate Limits

```bash
# Reduce concurrency
python evaluator.py --batch --max-workers 2 --documents ...

# Add delays between requests
python evaluator.py --batch --request-delay 1.0 --documents ...
```

#### File Access Issues

```bash
# Check file permissions
ls -la /path/to/documents

# Use absolute paths
python evaluator.py --batch --documents "/absolute/path/document.txt"
```

### Debug Mode

```bash
# Enable debug logging
python evaluator.py --batch --debug --documents ...

# Verbose output
python evaluator.py --batch --verbose --documents ...

# Save debug logs
python evaluator.py --batch --log-file debug.log --documents ...
```

## Integration Examples

### Automated Processing

```python
import schedule
import time

def process_daily_documents():
    """Process documents from a daily folder"""
    evaluator = BatchEvaluator()

    # Process yesterday's documents
    yesterday = datetime.now() - timedelta(days=1)
    folder_path = f"/documents/{yesterday.strftime('%Y-%m-%d')}"

    batch_id = evaluator.create_batch_from_folder(folder_path)
    evaluator.start_batch(batch_id)

    # Wait for completion
    while evaluator.get_batch_status(batch_id) == "processing":
        time.sleep(30)

    # Send results
    results = evaluator.get_batch_results(batch_id)
    send_notification(f"Processed {results.total_documents} documents")

# Schedule daily processing
schedule.every().day.at("09:00").do(process_daily_documents)

while True:
    schedule.run_pending()
    time.sleep(60)
```

### Webhook Integration

```python
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/webhook/batch-complete', methods=['POST'])
def handle_batch_complete():
    data = request.json
    batch_id = data['batch_id']

    # Get results
    results = evaluator.get_batch_results(batch_id)

    # Send to external system
    send_to_external_system(results)

    return jsonify({"status": "success"})

if __name__ == '__main__':
    app.run(debug=True)
```

## Next Steps

1. **Test with Small Batches**: Start with 5-10 documents to verify setup
2. **Monitor Performance**: Use progress monitoring to optimize batch sizes
3. **Implement Error Handling**: Add retry logic for production use
4. **Scale Gradually**: Increase batch sizes as system performance is validated

---

**Quickstart Status**: Complete  
**Examples**: Provided  
**Troubleshooting**: Covered  
**Ready for Implementation**: ✅
