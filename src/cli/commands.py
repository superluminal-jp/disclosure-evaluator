"""
CLI command handlers for disclosure evaluator.
"""

import sys
from pathlib import Path
from typing import List, Optional

from ..models.batch import BatchConfiguration, DocumentInput
from ..batch import BatchEvaluator
from ..evaluators import evaluate_disclosure
from ..utils import format_structured_output
from ..config import ConfigManager


def print_usage():
    """Print usage information"""
    print(
        """
Usage: python evaluator.py [OPTIONS] [INPUT]

Single Document Evaluation:
  python evaluator.py <input_text> [context] [output_text] [--format json|summary] [--provider openai|anthropic|bedrock|bedrock_nova]

Batch Processing:
  python evaluator.py --batch --folder <folder_path> [OPTIONS]
  python evaluator.py --batch --documents <file1,file2,...> [OPTIONS]
  python evaluator.py --batch-status <batch_id>
  python evaluator.py --batch-results <batch_id> [--format json|summary|csv]
  python evaluator.py --resume-batch <batch_id>
  python evaluator.py --retry-documents <batch_id> <document_id1,document_id2,...>

Batch Options:
  --folder <path>              Process all documents in folder
  --documents <files>          Process specific documents (comma-separated)
  --recursive                  Include subdirectories (default: true)
  --file-types <types>         Filter by MIME types (comma-separated)
  --exclude <patterns>         Exclude file patterns (comma-separated)
  --max-workers <n>            Maximum parallel workers (default: 5)
  --timeout <seconds>          Timeout per document (default: 300)
  --retry-attempts <n>          Maximum retry attempts (default: 3)
  --file-size-limit <bytes>    Maximum file size (default: 50MB)
  --context <text>             Additional context for all documents
  --output-formats <formats>   Output formats (default: json,summary)

Examples:
  # Single document
  python evaluator.py "Personal information: John Doe" --format summary

  # Batch from folder
  python evaluator.py --batch --folder ./documents --recursive --max-workers 3

  # Batch specific files
  python evaluator.py --batch --documents file1.txt,file2.pdf --context "Legal review"

  # Check batch status
  python evaluator.py --batch-status batch_20250104_143022

  # Get batch results
  python evaluator.py --batch-results batch_20250104_143022 --format csv
"""
    )


def handle_single_document_evaluation():
    """Handle single document evaluation"""
    input_text = sys.argv[1]
    context = sys.argv[2] if len(sys.argv) > 2 else ""
    output_text = sys.argv[3] if len(sys.argv) > 3 else ""

    # Parse format argument
    format_type = "json"  # default
    if "--format" in sys.argv:
        format_idx = sys.argv.index("--format")
        if format_idx + 1 < len(sys.argv):
            format_type = sys.argv[format_idx + 1]

    # Parse provider argument
    provider = None
    if "--provider" in sys.argv:
        provider_idx = sys.argv.index("--provider")
        if provider_idx + 1 < len(sys.argv):
            provider = sys.argv[provider_idx + 1]

    try:
        # Initialize config manager
        config_manager = ConfigManager()
        result = evaluate_disclosure(
            input_text,
            context,
            output_text,
            provider=provider,
            config_manager=config_manager,
        )
        formatted_output = format_structured_output(result, format_type)
        print(formatted_output)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def handle_batch_command():
    """Handle batch processing command"""
    try:
        # Parse batch arguments
        folder_path = None
        documents = None
        recursive = True
        file_types = None
        exclude_patterns = None
        max_workers = 5
        timeout = 300
        retry_attempts = 3
        file_size_limit = 50 * 1024 * 1024  # 50MB
        context = ""
        output_formats = ["json", "summary"]

        # Parse arguments
        if "--folder" in sys.argv:
            folder_idx = sys.argv.index("--folder")
            if folder_idx + 1 < len(sys.argv):
                folder_path = sys.argv[folder_idx + 1]

        if "--documents" in sys.argv:
            docs_idx = sys.argv.index("--documents")
            if docs_idx + 1 < len(sys.argv):
                documents = sys.argv[docs_idx + 1].split(",")

        if "--recursive" in sys.argv:
            recursive = True

        if "--file-types" in sys.argv:
            types_idx = sys.argv.index("--file-types")
            if types_idx + 1 < len(sys.argv):
                file_types = sys.argv[types_idx + 1].split(",")

        if "--exclude" in sys.argv:
            exclude_idx = sys.argv.index("--exclude")
            if exclude_idx + 1 < len(sys.argv):
                exclude_patterns = sys.argv[exclude_idx + 1].split(",")

        if "--max-workers" in sys.argv:
            workers_idx = sys.argv.index("--max-workers")
            if workers_idx + 1 < len(sys.argv):
                max_workers = int(sys.argv[workers_idx + 1])

        if "--timeout" in sys.argv:
            timeout_idx = sys.argv.index("--timeout")
            if timeout_idx + 1 < len(sys.argv):
                timeout = int(sys.argv[timeout_idx + 1])

        if "--retry-attempts" in sys.argv:
            retry_idx = sys.argv.index("--retry-attempts")
            if retry_idx + 1 < len(sys.argv):
                retry_attempts = int(sys.argv[retry_idx + 1])

        if "--file-size-limit" in sys.argv:
            size_idx = sys.argv.index("--file-size-limit")
            if size_idx + 1 < len(sys.argv):
                file_size_limit = int(sys.argv[size_idx + 1])

        if "--context" in sys.argv:
            context_idx = sys.argv.index("--context")
            if context_idx + 1 < len(sys.argv):
                context = sys.argv[context_idx + 1]

        if "--output-formats" in sys.argv:
            formats_idx = sys.argv.index("--output-formats")
            if formats_idx + 1 < len(sys.argv):
                output_formats = sys.argv[formats_idx + 1].split(",")

        # Create batch configuration
        config = BatchConfiguration(
            max_concurrent_workers=max_workers,
            timeout_seconds=timeout,
            max_retry_attempts=retry_attempts,
            file_size_limit=file_size_limit,
            output_formats=output_formats,
        )

        # Initialize config manager
        config_manager = ConfigManager()

        # Create batch evaluator
        evaluator = BatchEvaluator(config=config, config_manager=config_manager)

        # Create batch
        if folder_path:
            batch_id = evaluator.create_batch_from_folder(
                folder_path=folder_path,
                context=context,
                recursive=recursive,
                file_types=file_types,
                exclude_patterns=exclude_patterns,
                file_size_limit=file_size_limit,
            )
        elif documents:
            # Create document inputs
            doc_inputs = []
            for doc_path in documents:
                doc_inputs.append(
                    DocumentInput(
                        file_path=doc_path,
                        file_name=Path(doc_path).name,
                        context=context,
                    )
                )
            batch_id = evaluator.create_batch(doc_inputs, config)
        else:
            print("Error: Must specify either --folder or --documents", file=sys.stderr)
            sys.exit(1)

        # Start batch processing
        evaluator.start_batch(batch_id)

        print(f"Batch processing started: {batch_id}")
        print(f"Monitor progress with: python evaluator.py --batch-status {batch_id}")
        print(f"Get results with: python evaluator.py --batch-results {batch_id}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def handle_batch_status_command():
    """Handle batch status command"""
    try:
        if "--batch-status" not in sys.argv:
            print("Error: --batch-status requires a batch_id", file=sys.stderr)
            sys.exit(1)

        status_idx = sys.argv.index("--batch-status")
        if status_idx + 1 >= len(sys.argv):
            print("Error: --batch-status requires a batch_id", file=sys.stderr)
            sys.exit(1)

        batch_id = sys.argv[status_idx + 1]

        # Initialize config manager
        config_manager = ConfigManager()

        # Create batch evaluator
        evaluator = BatchEvaluator(config_manager=config_manager)

        # Get batch status
        batch = evaluator.get_batch(batch_id)
        if not batch:
            print(f"Error: Batch {batch_id} not found", file=sys.stderr)
            sys.exit(1)

        progress = evaluator.get_batch_progress(batch_id)

        print(f"Batch ID: {batch_id}")
        print(f"Status: {batch.status.value}")
        print(f"Total Documents: {batch.total_documents}")
        print(f"Processed: {batch.processed_documents}")
        print(f"Successful: {batch.successful_documents}")
        print(f"Failed: {batch.failed_documents}")
        if progress:
            print(f"Progress: {progress.progress_percentage:.1f}%")
            print(f"Current Phase: {progress.current_phase.value}")
            print(f"Last Updated: {progress.last_updated}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def handle_batch_results_command():
    """Handle batch results command"""
    try:
        if "--batch-results" not in sys.argv:
            print("Error: --batch-results requires a batch_id", file=sys.stderr)
            sys.exit(1)

        results_idx = sys.argv.index("--batch-results")
        if results_idx + 1 >= len(sys.argv):
            print("Error: --batch-results requires a batch_id", file=sys.stderr)
            sys.exit(1)

        batch_id = sys.argv[results_idx + 1]

        # Parse format argument
        format_type = "json"
        if "--format" in sys.argv:
            format_idx = sys.argv.index("--format")
            if format_idx + 1 < len(sys.argv):
                format_type = sys.argv[format_idx + 1]

        # Initialize config manager
        config_manager = ConfigManager()

        # Create batch evaluator
        evaluator = BatchEvaluator(config_manager=config_manager)

        # Get batch results
        results = evaluator.get_batch_results(batch_id, format=format_type)
        if not results:
            print(f"Error: No results found for batch {batch_id}", file=sys.stderr)
            sys.exit(1)

        # Format and print results
        if format_type == "json":
            print(results.model_dump_json(indent=2))
        elif format_type == "summary":
            print(f"Batch Results for {batch_id}")
            print(f"Total Documents: {results.total_documents}")
            print(f"Successful: {results.successful_evaluations}")
            print(f"Failed: {results.failed_evaluations}")
            print(f"Success Rate: {results.success_rate:.1%}")
            print(f"Processing Duration: {results.processing_duration}")
        elif format_type == "csv":
            # CSV output would be implemented here
            print("CSV output not yet implemented")
        else:
            print(f"Error: Unsupported format {format_type}", file=sys.stderr)
            sys.exit(1)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def handle_resume_batch_command():
    """Handle resume batch command"""
    try:
        if "--resume-batch" not in sys.argv:
            print("Error: --resume-batch requires a batch_id", file=sys.stderr)
            sys.exit(1)

        resume_idx = sys.argv.index("--resume-batch")
        if resume_idx + 1 >= len(sys.argv):
            print("Error: --resume-batch requires a batch_id", file=sys.stderr)
            sys.exit(1)

        batch_id = sys.argv[resume_idx + 1]

        # Initialize config manager
        config_manager = ConfigManager()

        # Create batch evaluator
        evaluator = BatchEvaluator(config_manager=config_manager)

        # Resume batch
        success = evaluator.resume_batch(batch_id)
        if success:
            print(f"Batch {batch_id} resumed successfully")
        else:
            print(f"Error: Failed to resume batch {batch_id}", file=sys.stderr)
            sys.exit(1)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def handle_retry_documents_command():
    """Handle retry documents command"""
    try:
        if "--retry-documents" not in sys.argv:
            print(
                "Error: --retry-documents requires batch_id and document_ids",
                file=sys.stderr,
            )
            sys.exit(1)

        retry_idx = sys.argv.index("--retry-documents")
        if retry_idx + 2 >= len(sys.argv):
            print(
                "Error: --retry-documents requires batch_id and document_ids",
                file=sys.stderr,
            )
            sys.exit(1)

        batch_id = sys.argv[retry_idx + 1]
        document_ids = sys.argv[retry_idx + 2].split(",")

        # Initialize config manager
        config_manager = ConfigManager()

        # Create batch evaluator
        evaluator = BatchEvaluator(config_manager=config_manager)

        # Retry documents
        for doc_id in document_ids:
            success = evaluator.retry_document(batch_id, doc_id.strip())
            if success:
                print(f"Document {doc_id.strip()} retry initiated")
            else:
                print(
                    f"Error: Failed to retry document {doc_id.strip()}", file=sys.stderr
                )

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
