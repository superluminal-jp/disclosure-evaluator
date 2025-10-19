#!/usr/bin/env python3
"""
Disclosure Evaluator - CLI Entry Point

A modular system for evaluating information disclosure under Japanese law.
"""

import sys
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import CLI commands
from src.cli.commands import (
    print_usage,
    handle_single_document_evaluation,
    handle_batch_command,
    handle_batch_status_command,
    handle_batch_results_command,
    handle_resume_batch_command,
    handle_retry_documents_command,
)


def main():
    """Main CLI entry point with batch processing support"""
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)

    # Check for help commands first
    if "--help" in sys.argv or "-h" in sys.argv:
        print_usage()
        sys.exit(0)

    # Check for batch processing commands
    if "--batch" in sys.argv:
        handle_batch_command()
    elif "--batch-status" in sys.argv:
        handle_batch_status_command()
    elif "--batch-results" in sys.argv:
        handle_batch_results_command()
    elif "--resume-batch" in sys.argv:
        handle_resume_batch_command()
    elif "--retry-documents" in sys.argv:
        handle_retry_documents_command()
    else:
        # Single document evaluation
        handle_single_document_evaluation()


if __name__ == "__main__":
    main()
