#!/usr/bin/env python3
"""
Test runner for FastAPI tests.

This script sets up the correct Python path and runs the FastAPI tests.
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set environment variables for testing
os.environ.setdefault("DEBUG", "true")
os.environ.setdefault("ENVIRONMENT", "testing")

if __name__ == "__main__":
    import pytest

    # Run the tests
    test_args = ["tests/api/test_fastapi_health.py", "-v", "--tb=short"]

    exit_code = pytest.main(test_args)
    sys.exit(exit_code)
