"""
Shared fixtures and mocks for disclosure evaluator tests.
"""

import json
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import Mock, patch, MagicMock
import pytest
from pydantic import BaseModel

# Import main module components
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from main import (
    LLMProvider,
    OpenAIProvider,
    AnthropicProvider,
    BedrockAnthropicProvider,
    ConfigManager,
    EvaluationStep,
    CriterionEvaluation,
    DisclosureEvaluationResult,
    BatchStatus,
    DocumentStatus,
    ProcessingPhase,
    BatchConfiguration,
    BatchSummaryStatistics,
    DocumentResult,
    DocumentError,
    BatchEvaluation,
    BatchDocument,
    BatchProgress,
    BatchResult,
    DocumentInput,
    StepEvaluator,
    CriterionEvaluator,
    ResultAggregator,
    DisclosureEvaluator,
    DocumentDiscoveryService,
    BatchStatePersistenceService,
    ParallelDocumentProcessingService,
    BatchEvaluator,
)


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client with responses."""
    mock_client = Mock()
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = "Mock OpenAI response"
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


@pytest.fixture
def mock_anthropic_client():
    """Mock Anthropic client with responses."""
    mock_client = Mock()
    mock_response = Mock()
    mock_response.content = [Mock()]
    mock_response.content[0].text = "Mock Anthropic response"
    mock_client.messages.create.return_value = mock_response
    return mock_client


@pytest.fixture
def mock_bedrock_client():
    """Mock AWS Bedrock client with responses."""
    mock_client = Mock()
    mock_response = {"body": Mock(), "contentType": "application/json"}
    mock_response["body"].read.return_value = json.dumps(
        {"content": [{"text": "Mock Bedrock response"}]}
    ).encode()
    mock_client.invoke_model.return_value = mock_response
    return mock_client


@pytest.fixture
def mock_config_manager():
    """Mock configuration manager."""
    config = {
        "application": {"name": "Test", "version": "1.0.0"},
        "logging": {"level": "INFO", "format": "%(message)s"},
        "llm": {
            "provider": "openai",
            "openai": {"model": "gpt-4", "temperature": 0.1, "max_tokens": 2000},
            "anthropic": {"model": "claude-3", "temperature": 0.1, "max_tokens": 2000},
            "bedrock": {"model": "claude-3", "temperature": 0.1, "max_tokens": 2000},
        },
        "evaluation": {"parallel": {"enabled": True, "max_workers": 3}},
        "output": {"format": "json"},
    }
    return Mock(spec=ConfigManager, **{k: v for k, v in config.items()})


@pytest.fixture
def sample_evaluation_step():
    """Sample evaluation step for testing."""
    return EvaluationStep(step="Test step", result="YES", reasoning="Test reasoning")


@pytest.fixture
def sample_criterion():
    """Sample criterion configuration."""
    return {
        "id": "article_5_1",
        "name": "個人情報保護",
        "article": "第5条第1号",
        "evaluation_steps": ["個人に関する情報か", "特定の個人を識別できるか"],
    }


@pytest.fixture
def sample_criteria():
    """Sample criteria list for testing."""
    return {
        "criteria": [
            {
                "id": "article_5_1",
                "name": "個人情報保護",
                "article": "第5条第1号",
                "evaluation_steps": ["個人に関する情報か", "特定の個人を識別できるか"],
            },
            {
                "id": "article_5_2",
                "name": "法人等情報保護",
                "article": "第5条第2号",
                "evaluation_steps": ["法人等の情報か", "競争上の地位に影響するか"],
            },
        ]
    }


@pytest.fixture
def sample_document_input():
    """Sample document input for testing."""
    return DocumentInput(
        file_path="/test/path/document.txt",
        file_name="document.txt",
        context="Test context",
        output_text="Test output",
    )


@pytest.fixture
def temp_config_file(tmp_path):
    """Create temporary config.json file."""
    config = {
        "application": {"name": "Test", "version": "1.0.0"},
        "logging": {"level": "INFO", "format": "%(message)s"},
        "llm": {
            "provider": "openai",
            "openai": {"model": "gpt-4", "temperature": 0.1, "max_tokens": 2000},
        },
        "evaluation": {"parallel": {"enabled": True, "max_workers": 3}},
        "output": {"format": "json"},
    }
    config_file = tmp_path / "config.json"
    with open(config_file, "w") as f:
        json.dump(config, f)
    return str(config_file)


@pytest.fixture
def temp_batch_state_dir(tmp_path):
    """Create temporary batch state directory."""
    batch_dir = tmp_path / "batch_state"
    active_dir = batch_dir / "active_batches"
    completed_dir = batch_dir / "completed_batches"
    active_dir.mkdir(parents=True)
    completed_dir.mkdir(parents=True)
    return str(batch_dir)


@pytest.fixture
def mock_llm_response():
    """Mock LLM response for testing."""
    return {
        "step_response": "結果: YES\n理由: テスト理由",
        "reasoning_response": "スコア理由: 各段階の評価結果を総合した理由",
        "anthropic_response": "結果: NO\n理由: アンソロピック理由",
        "bedrock_response": "結果: YES\n理由: ベッドロック理由",
    }


@pytest.fixture
def mock_env_vars():
    """Mock environment variables."""
    with patch.dict(
        os.environ,
        {
            "OPENAI_API_KEY": "test-openai-key",
            "ANTHROPIC_API_KEY": "test-anthropic-key",
            "AWS_ACCESS_KEY_ID": "test-aws-key",
            "AWS_SECRET_ACCESS_KEY": "test-aws-secret",
            "AWS_REGION": "us-east-1",
        },
    ):
        yield


@pytest.fixture
def sample_batch_evaluation():
    """Sample batch evaluation for testing."""
    return BatchEvaluation(
        batch_id="test_batch_001",
        created_at=datetime.now(),
        status=BatchStatus.PENDING,
        total_documents=5,
        correlation_id="test_correlation",
        configuration=BatchConfiguration(),
    )


@pytest.fixture
def sample_batch_document():
    """Sample batch document for testing."""
    return BatchDocument(
        document_id="doc_001",
        batch_id="test_batch_001",
        file_path="/test/path/document.txt",
        file_name="document.txt",
        file_size=1024,
        mime_type="text/plain",
        status=DocumentStatus.PENDING,
        correlation_id="test_correlation",
        context="Test context",
        output_text="Test output",
    )


@pytest.fixture
def sample_document_result():
    """Sample document result for testing."""
    return DocumentResult(
        document_id="doc_001",
        evaluation_result=DisclosureEvaluationResult(
            input_text="Test input",
            context="Test context",
            output_text="Test output",
            criterion_evaluations=[],
            evaluation_timestamp=datetime.now().isoformat(),
        ),
        processing_time=timedelta(seconds=30),
        success=True,
    )


@pytest.fixture
def mock_criteria_file(tmp_path):
    """Create mock criteria JSON file."""
    criteria = {
        "name": "テスト評価基準",
        "version": "1.0",
        "criteria": [
            {
                "id": "article_5_1",
                "name": "個人情報保護",
                "article": "第5条第1号",
                "evaluation_steps": ["個人に関する情報か", "特定の個人を識別できるか"],
            }
        ],
    }
    criteria_file = tmp_path / "criteria" / "disclosure_evaluation_criteria.json"
    criteria_file.parent.mkdir(parents=True)
    with open(criteria_file, "w", encoding="utf-8") as f:
        json.dump(criteria, f, ensure_ascii=False, indent=2)
    return str(criteria_file)


@pytest.fixture
def sample_documents_dir(tmp_path):
    """Create sample documents directory for testing."""
    docs_dir = tmp_path / "documents"
    docs_dir.mkdir()

    # Create test files
    (docs_dir / "doc1.txt").write_text("Test document 1")
    (docs_dir / "doc2.txt").write_text("Test document 2")
    (docs_dir / "subdir").mkdir()
    (docs_dir / "subdir" / "doc3.txt").write_text("Test document 3")

    return str(docs_dir)


@pytest.fixture
def mock_file_operations():
    """Mock file operations for testing."""
    with patch("builtins.open", create=True) as mock_open:
        with patch("pathlib.Path.exists") as mock_exists:
            with patch("pathlib.Path.stat") as mock_stat:
                mock_stat.return_value.st_size = 1024
                mock_exists.return_value = True
                yield mock_open


@pytest.fixture
def mock_threading():
    """Mock threading components for testing."""
    with patch("threading.Lock") as mock_lock:
        with patch("concurrent.futures.ThreadPoolExecutor") as mock_executor:
            mock_lock.return_value.__enter__ = Mock()
            mock_lock.return_value.__exit__ = Mock()
            mock_executor.return_value.__enter__ = Mock(
                return_value=mock_executor.return_value
            )
            mock_executor.return_value.__exit__ = Mock()
            yield mock_executor


@pytest.fixture
def mock_logging():
    """Mock logging for testing."""
    with patch("logging.getLogger") as mock_get_logger:
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        yield mock_logger


@pytest.fixture
def mock_datetime():
    """Mock datetime for consistent testing."""
    fixed_time = datetime(2025, 1, 1, 12, 0, 0)
    with patch("main.datetime") as mock_dt:
        mock_dt.now.return_value = fixed_time
        mock_dt.strftime = fixed_time.strftime
        yield mock_dt


@pytest.fixture
def mock_json_operations():
    """Mock JSON operations for testing."""
    with patch("json.load") as mock_load:
        with patch("json.dump") as mock_dump:
            mock_load.return_value = {"test": "data"}
            yield mock_load, mock_dump


@pytest.fixture
def mock_os_operations():
    """Mock OS operations for testing."""
    with patch("os.makedirs") as mock_makedirs:
        with patch("os.getenv") as mock_getenv:
            mock_getenv.return_value = "test-value"
            yield mock_makedirs, mock_getenv


@pytest.fixture
def mock_mimetypes():
    """Mock mimetypes for testing."""
    with patch("mimetypes.guess_type") as mock_guess:
        mock_guess.return_value = ("text/plain", None)
        yield mock_guess


@pytest.fixture
def mock_sys_argv():
    """Mock sys.argv for CLI testing."""
    with patch("sys.argv", ["main.py", "test_input", "test_context", "test_output"]):
        yield


@pytest.fixture
def mock_sys_exit():
    """Mock sys.exit for CLI testing."""
    with patch("sys.exit") as mock_exit:
        yield mock_exit


@pytest.fixture
def mock_print():
    """Mock print function for CLI testing."""
    with patch("builtins.print") as mock_print:
        yield mock_print


@pytest.fixture
def mock_subprocess():
    """Mock subprocess for integration testing."""
    with patch("subprocess.run") as mock_run:
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "Test output"
        mock_result.stderr = ""
        mock_run.return_value = mock_result
        yield mock_run


# Test data fixtures
@pytest.fixture
def test_evaluation_context():
    """Test evaluation context data."""
    return {
        "input_text": "個人情報: 田中太郎, 住所: 東京都渋谷区",
        "context": "情報公開請求に対する検討",
        "output_text": "開示判断の結果",
    }


@pytest.fixture
def test_batch_configuration():
    """Test batch configuration."""
    return BatchConfiguration(
        max_concurrent_workers=2,
        max_retry_attempts=2,
        timeout_seconds=60,
        file_size_limit=1024,
        output_formats=["json", "summary"],
    )


@pytest.fixture
def test_llm_messages():
    """Test LLM messages for mocking."""
    return [
        {"role": "system", "content": "You are a legal expert."},
        {"role": "user", "content": "Evaluate this document."},
    ]


@pytest.fixture
def test_parallel_config():
    """Test parallel processing configuration."""
    return {"enabled": True, "max_workers": 3, "timeout": 300}


@pytest.fixture
def test_error_scenarios():
    """Test error scenarios for exception handling."""
    return {
        "api_error": Exception("API request failed"),
        "file_not_found": FileNotFoundError("File not found"),
        "json_decode_error": json.JSONDecodeError("Invalid JSON", "", 0),
        "timeout_error": TimeoutError("Request timeout"),
        "permission_error": PermissionError("Access denied"),
    }
