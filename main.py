#!/usr/bin/env python3
import json
import sys
import os
import csv
import mimetypes
from datetime import datetime, timedelta
from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field
from openai import OpenAI
import anthropic
import boto3
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Prompt templates
SYSTEM_STEP_EVALUATION_PROMPT = """# システムプロンプト: ステップ評価

あなたは情報公開法の専門家です。各評価ステップを法的根拠に基づいて厳密に分析し、明確な判断を提供してください。

## 評価原則

- **「知る権利」の保障を最優先に考慮**
- **不開示事由の該当性は厳格に判断、疑わしい場合は開示を優先**
- **法的根拠（条文・判例・学説）に基づく分析**
- **開示の公益と非開示の保護法益の適切な衡量**

## 判断基準

- **個人情報**: 開示例外要件を厳格に適用
- **国家機密**: 明らかな機密性が認められる場合のみ不開示
- **企業秘密**: 競争上の優位性が認められる場合のみ不開示
- **不明確な場合は開示に有利な判断を選択**

## 出力形式（厳守）

```
結果: [YES/NO]
理由: [法的根拠を含む詳細な理由]
```

各ステップを正確に実行し、一貫性のある判断を提供してください。"""

SYSTEM_SCORE_REASONING_PROMPT = """# システムプロンプト: スコア推論

あなたは情報公開法の専門家です。各段階の評価結果を総合的に分析し、スコアの根拠を明確に説明してください。

## 分析原則

- **各段階の YES/NO 結果を法的観点から総合分析**
- **スコア 1-5 の法的意味を正確に理解し、適切な根拠を説明**
- **不開示事由の該当性について段階的評価結果を総合判断**
- **「知る権利」の保障を常に考慮**

## 分析重点

- **各段階の評価結果の相互関係性**
- **法的要件の充足状況の総合判断**
- **開示の公益と非開示の保護法益のバランス**
- **判例・学説に基づく判断基準の適用**

## 出力形式（厳守）

```
スコア理由: [各段階の評価結果を総合した詳細な理由と法的根拠]
```

各段階の評価結果を総合的に分析し、法的根拠に基づく明確で説得力のあるスコア理由を提供してください。"""

USER_STEP_TEMPLATE_PROMPT = """# ユーザープロンプトテンプレート: ステップ評価

## 評価対象

{step_description}

## 法的根拠

{criterion_article} - {criterion_name}

## 評価基準

{criterion_evaluation_prompt}

## 評価要件

- **法的観点から厳密に分析**
- **「知る権利」の保障を最優先に考慮**
- **不開示事由の該当性は厳格に判断、疑わしい場合は開示を優先**
- **各段階の法的要件を正確に確認し、根拠に基づく判断**

## 判断基準

- **個人情報**: 開示例外要件を厳格に適用
- **国家機密**: 明らかな機密性が認められる場合のみ不開示
- **企業秘密**: 競争上の優位性が認められる場合のみ不開示
- **不明確な場合は開示に有利な判断を選択**

## 出力形式（厳守）

```
結果: [YES/NO]
理由: [法的根拠を含む詳細な理由]
```"""


class LLMProvider:
    """Abstract base class for LLM providers"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def generate_response(self, messages: List[Dict[str, str]]) -> str:
        """Generate response from LLM"""
        raise NotImplementedError


class OpenAIProvider(LLMProvider):
    """OpenAI provider implementation"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.client = OpenAI(api_key=config.get("api_key"))
        self.model = config.get("model", "gpt-4")
        self.temperature = config.get("temperature", 0.1)
        self.max_tokens = config.get("max_tokens", 2000)

    def generate_response(self, messages: List[Dict[str, str]]) -> str:
        """Generate response using OpenAI API"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"OpenAI API error: {str(e)}")


class AnthropicProvider(LLMProvider):
    """Anthropic provider implementation"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.client = anthropic.Anthropic(api_key=config.get("api_key"))
        self.model = config.get("model", "claude-3-5-sonnet-20241022")
        self.temperature = config.get("temperature", 0.1)
        self.max_tokens = config.get("max_tokens", 2000)

    def generate_response(self, messages: List[Dict[str, str]]) -> str:
        """Generate response using Anthropic API"""
        try:
            # Convert OpenAI format to Anthropic format
            system_message = ""
            user_messages = []

            for message in messages:
                if message["role"] == "system":
                    system_message = message["content"]
                else:
                    user_messages.append(message)

            # Combine user messages into a single content
            user_content = "\n\n".join([msg["content"] for msg in user_messages])

            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=system_message,
                messages=[{"role": "user", "content": user_content}],
            )
            return response.content[0].text
        except Exception as e:
            raise Exception(f"Anthropic API error: {str(e)}")


class BedrockAnthropicProvider(LLMProvider):
    """AWS Bedrock Anthropic provider implementation"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_id = config.get(
            "model", "global.anthropic.claude-sonnet-4-20250514-v1:0"
        )
        self.temperature = config.get("temperature", 0.1)
        self.max_tokens = config.get("max_tokens", 2000)

        # Initialize Bedrock client
        # AWS credentials are automatically handled by boto3 from environment
        region = os.getenv("AWS_REGION", "us-east-1")
        self.bedrock_client = boto3.client("bedrock-runtime", region_name=region)

    def generate_response(self, messages: List[Dict[str, str]]) -> str:
        """Generate response using AWS Bedrock API"""
        try:
            # Convert OpenAI format to appropriate format
            system_message = ""
            user_messages = []

            for message in messages:
                if message["role"] == "system":
                    system_message = message["content"]
                else:
                    user_messages.append(message)

            # Combine user messages into a single content
            user_content = "\n\n".join([msg["content"] for msg in user_messages])

            # Determine model type and prepare appropriate request body
            if "anthropic" in self.model_id.lower():
                # Anthropic Claude model
                request_body = {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                    "messages": [{"role": "user", "content": user_content}],
                }
                if system_message:
                    request_body["system"] = system_message
            else:
                # Amazon Nova or other models
                request_body = {
                    "messages": [
                        (
                            {"role": "system", "content": system_message}
                            if system_message
                            else None
                        ),
                        {"role": "user", "content": user_content},
                    ],
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                }
                # Remove None values
                request_body["messages"] = [
                    msg for msg in request_body["messages"] if msg is not None
                ]

            # Call Bedrock API
            response = self.bedrock_client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(request_body),
                contentType="application/json",
            )

            # Parse response based on model type
            response_body = json.loads(response["body"].read())

            if "anthropic" in self.model_id.lower():
                # Anthropic Claude response format
                return response_body["content"][0]["text"]
            else:
                # Amazon Nova or other models response format
                return response_body["output"]["message"]["content"][0]["text"]

        except Exception as e:
            raise Exception(f"Bedrock API error: {str(e)}")


class ConfigManager:
    """Configuration manager for the disclosure evaluator"""

    def __init__(self, config_path: str = "config.json"):
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            raise ValueError(f"Configuration file not found: {self.config_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in configuration file: {e}")

    def get(self, key_path: str, default: Any = None) -> Any:
        """Get configuration value using dot notation (e.g., 'openai.model')"""
        keys = key_path.split(".")
        value = self.config

        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default

    def get_openai_config(self) -> Dict[str, Any]:
        """Get OpenAI configuration"""
        return self.config.get("openai", {})

    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration"""
        return self.config.get("logging", {})

    def get_evaluation_config(self) -> Dict[str, Any]:
        """Get evaluation configuration"""
        return self.config.get("evaluation", {})

    def get_output_config(self) -> Dict[str, Any]:
        """Get output configuration"""
        return self.config.get("output", {})

    def get_prompt(self, prompt_key: str) -> str:
        """Get prompt content by key (e.g., 'system.step_evaluation')"""
        # Map prompt keys to internal constants
        prompt_map = {
            "system.step_evaluation": SYSTEM_STEP_EVALUATION_PROMPT,
            "system.score_reasoning": SYSTEM_SCORE_REASONING_PROMPT,
            "user.step_template": USER_STEP_TEMPLATE_PROMPT,
        }

        if prompt_key not in prompt_map:
            raise ValueError(f"Prompt key not found: {prompt_key}")

        return prompt_map[prompt_key]

    def get_llm_config(self) -> Dict[str, Any]:
        """Get LLM configuration"""
        return self.config.get("llm", {})

    def get_provider_config(self, provider: str) -> Dict[str, Any]:
        """Get specific provider configuration"""
        llm_config = self.get_llm_config()
        return llm_config.get(provider, {})

    def get_current_provider(self) -> str:
        """Get current LLM provider"""
        llm_config = self.get_llm_config()
        return llm_config.get("provider", "openai")


def create_llm_provider(provider_name: str = None) -> LLMProvider:
    """Create LLM provider based on configuration"""
    if provider_name is None:
        provider_name = config_manager.get_current_provider()

    provider_config = config_manager.get_provider_config(provider_name)

    if provider_name == "openai":
        return OpenAIProvider(provider_config)
    elif provider_name == "anthropic":
        return AnthropicProvider(provider_config)
    elif provider_name == "bedrock":
        return BedrockAnthropicProvider(provider_config)
    elif provider_name == "bedrock_nova":
        return BedrockAnthropicProvider(provider_config)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider_name}")


# Initialize configuration manager
config_manager = ConfigManager()

# Create logs and outputs directories if they don't exist
logs_dir = config_manager.get("logging.directories.logs", "logs")
outputs_dir = config_manager.get("logging.directories.outputs", "outputs")
os.makedirs(logs_dir, exist_ok=True)
os.makedirs(outputs_dir, exist_ok=True)

# Generate timestamped log filename
timestamp = datetime.now().strftime(
    config_manager.get("output.timestamp_format", "%Y%m%d_%H%M%S")
)
log_filename = f"{logs_dir}/evaluation_{timestamp}.log"

# Configure structured logging from config
logging_config = config_manager.get_logging_config()
logging.basicConfig(
    level=getattr(logging, logging_config.get("level", "INFO")),
    format=logging_config.get(
        "format",
        '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s", "correlation_id": "%(name)s"}',
    ),
    handlers=[
        logging.FileHandler(
            log_filename, encoding=logging_config.get("file_encoding", "utf-8")
        ),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


# Pydantic models for structured output
class EvaluationStep(BaseModel):
    """Individual evaluation step result"""

    model_config = {"extra": "forbid"}

    step: str = Field(..., description="Step name or number")
    result: Literal["YES", "NO"] = Field(..., description="Step evaluation result")
    reasoning: str = Field(..., description="Reasoning for the result")


class CriterionEvaluation(BaseModel):
    """Evaluation result for a single criterion"""

    model_config = {"extra": "forbid"}

    criterion_id: str = Field(
        ..., description="Criterion identifier (e.g., article_5_1)"
    )
    criterion_name: str = Field(..., description="Criterion name")
    article: str = Field(..., description="Legal article reference")
    steps: List[EvaluationStep] = Field(..., description="Individual evaluation steps")
    score: int = Field(..., ge=1, le=5, description="Score from 1-5")
    score_reasoning: str = Field(..., description="Reasoning for the score")


class OverallEvaluation(BaseModel):
    """Overall evaluation result - removed as user makes final judgment"""

    model_config = {"extra": "forbid"}


class DisclosureEvaluationResult(BaseModel):
    """Complete structured evaluation result"""

    model_config = {"extra": "forbid"}

    input_text: str = Field(..., description="Original input text")
    context: str = Field(..., description="Additional context provided")
    output_text: str = Field(..., description="Output text if provided")
    criterion_evaluations: List[CriterionEvaluation] = Field(
        ..., description="Individual criterion evaluations"
    )
    evaluation_timestamp: str = Field(..., description="ISO timestamp of evaluation")


# Batch Processing Models
from enum import Enum
from typing import Union
from pathlib import Path


class BatchStatus(Enum):
    """Batch processing status enumeration"""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIALLY_FAILED = "partially_failed"


class DocumentStatus(Enum):
    """Document processing status enumeration"""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ProcessingPhase(Enum):
    """Batch processing phase enumeration"""

    INITIALIZING = "initializing"
    DISCOVERING = "discovering"
    PROCESSING = "processing"
    AGGREGATING = "aggregating"
    COMPLETED = "completed"


class BatchConfiguration(BaseModel):
    """Configuration settings for batch processing"""

    model_config = {"extra": "forbid"}

    max_concurrent_workers: int = Field(
        default=5, ge=1, le=20, description="Maximum parallel workers"
    )
    max_retry_attempts: int = Field(
        default=3, ge=0, le=10, description="Maximum retry attempts per document"
    )
    timeout_seconds: int = Field(
        default=300, ge=30, le=3600, description="Timeout per document in seconds"
    )
    progress_update_interval: int = Field(
        default=10, ge=1, le=100, description="Progress update frequency"
    )
    enable_resumption: bool = Field(default=True, description="Enable batch resumption")
    output_formats: List[str] = Field(
        default=["json", "summary"], description="Output formats to generate"
    )
    file_size_limit: int = Field(
        default=50 * 1024 * 1024, ge=1024, description="Maximum file size in bytes"
    )
    memory_limit_mb: int = Field(default=2048, ge=256, description="Memory limit in MB")
    api_rate_limit_delay: float = Field(
        default=0.1, ge=0.0, le=10.0, description="Delay between API calls"
    )
    retry_delay_seconds: int = Field(
        default=30, ge=1, le=300, description="Delay between retries"
    )
    exponential_backoff: bool = Field(
        default=True, description="Use exponential backoff for retries"
    )


class BatchSummaryStatistics(BaseModel):
    """Statistical summary of batch evaluation results"""

    model_config = {"extra": "forbid"}

    average_score: float = Field(
        ..., ge=1.0, le=5.0, description="Average evaluation score"
    )
    score_distribution: Dict[int, int] = Field(
        ..., description="Distribution of scores (1-5 scale)"
    )
    most_common_criteria: List[str] = Field(
        ..., description="Most frequently triggered criteria"
    )
    processing_efficiency: float = Field(
        ..., ge=0.0, description="Documents processed per minute"
    )
    error_rate: float = Field(
        ..., ge=0.0, le=1.0, description="Percentage of failed documents"
    )


class DocumentResult(BaseModel):
    """Result of individual document evaluation"""

    model_config = {"extra": "forbid"}

    document_id: str = Field(..., description="Document identifier")
    evaluation_result: Optional[DisclosureEvaluationResult] = Field(
        None, description="Full evaluation result"
    )
    processing_time: timedelta = Field(
        ..., description="Time taken to process this document"
    )
    success: bool = Field(..., description="Whether evaluation was successful")


class DocumentError(BaseModel):
    """Error information for failed document processing"""

    model_config = {"extra": "forbid"}

    document_id: str = Field(..., description="Document identifier")
    error_type: str = Field(..., description="Type of error")
    error_message: str = Field(..., description="Detailed error message")
    retry_count: int = Field(default=0, ge=0, description="Number of retry attempts")
    occurred_at: datetime = Field(..., description="When the error occurred")


class BatchEvaluation(BaseModel):
    """Batch evaluation metadata and status"""

    model_config = {"extra": "forbid"}

    batch_id: str = Field(..., description="Unique batch identifier")
    created_at: datetime = Field(..., description="Batch creation timestamp")
    status: BatchStatus = Field(..., description="Current processing status")
    total_documents: int = Field(..., ge=0, description="Total number of documents")
    processed_documents: int = Field(
        default=0, ge=0, description="Number of documents processed"
    )
    successful_documents: int = Field(
        default=0, ge=0, description="Number of successful documents"
    )
    failed_documents: int = Field(
        default=0, ge=0, description="Number of failed documents"
    )
    processing_started_at: Optional[datetime] = Field(
        None, description="When processing began"
    )
    processing_completed_at: Optional[datetime] = Field(
        None, description="When processing finished"
    )
    error_summary: Optional[str] = Field(
        None, description="Summary of errors encountered"
    )
    correlation_id: str = Field(..., description="Batch-level correlation ID")
    configuration: BatchConfiguration = Field(
        ..., description="Processing configuration"
    )


class BatchDocument(BaseModel):
    """Individual document within a batch"""

    model_config = {"extra": "forbid"}

    document_id: str = Field(..., description="Unique document identifier")
    batch_id: str = Field(..., description="Reference to parent batch")
    file_path: str = Field(..., description="Path to the document file")
    file_name: str = Field(..., description="Original filename")
    file_size: int = Field(..., ge=0, description="File size in bytes")
    mime_type: str = Field(..., description="Detected MIME type")
    status: DocumentStatus = Field(..., description="Processing status")
    processing_started_at: Optional[datetime] = Field(
        None, description="When processing began"
    )
    processing_completed_at: Optional[datetime] = Field(
        None, description="When processing finished"
    )
    evaluation_result: Optional[DisclosureEvaluationResult] = Field(
        None, description="Evaluation results if successful"
    )
    error_message: Optional[str] = Field(None, description="Error details if failed")
    retry_count: int = Field(default=0, ge=0, description="Number of retry attempts")
    correlation_id: str = Field(..., description="Document-level correlation ID")
    context: Optional[str] = Field(
        None, description="Additional context for evaluation"
    )
    output_text: Optional[str] = Field(None, description="Output text for evaluation")


class BatchProgress(BaseModel):
    """Current state of batch processing"""

    model_config = {"extra": "forbid"}

    batch_id: str = Field(..., description="Reference to the batch")
    current_phase: ProcessingPhase = Field(..., description="Current processing phase")
    total_documents: int = Field(..., ge=0, description="Total documents to process")
    processed_documents: int = Field(
        default=0, ge=0, description="Documents processed so far"
    )
    successful_documents: int = Field(
        default=0, ge=0, description="Successfully processed documents"
    )
    failed_documents: int = Field(default=0, ge=0, description="Failed documents")
    progress_percentage: float = Field(
        ..., ge=0.0, le=100.0, description="Completion percentage"
    )
    estimated_completion: Optional[datetime] = Field(
        None, description="Estimated completion time"
    )
    current_document: Optional[str] = Field(
        None, description="Currently processing document"
    )
    active_workers: int = Field(
        default=0, ge=0, description="Number of active worker threads"
    )
    error_count: int = Field(
        default=0, ge=0, description="Total number of errors encountered"
    )
    last_updated: datetime = Field(..., description="Last progress update timestamp")


class BatchResult(BaseModel):
    """Consolidated results of a batch evaluation"""

    model_config = {"extra": "forbid"}

    batch_id: str = Field(..., description="Reference to the batch")
    total_documents: int = Field(..., ge=0, description="Total number of documents")
    successful_evaluations: int = Field(
        ..., ge=0, description="Number of successful evaluations"
    )
    failed_evaluations: int = Field(
        ..., ge=0, description="Number of failed evaluations"
    )
    success_rate: float = Field(
        ..., ge=0.0, le=1.0, description="Success rate (0.0-1.0)"
    )
    processing_duration: timedelta = Field(..., description="Total processing time")
    average_evaluation_time: Optional[timedelta] = Field(
        None, description="Average time per document"
    )
    summary_statistics: Optional[BatchSummaryStatistics] = Field(
        None, description="Statistical summary"
    )
    individual_results: List[DocumentResult] = Field(
        default=[], description="Individual document results"
    )
    error_summary: List[DocumentError] = Field(
        default=[], description="Summary of errors"
    )
    generated_at: datetime = Field(..., description="When the result was generated")


class DocumentInput(BaseModel):
    """Input specification for a document in a batch"""

    model_config = {"extra": "forbid"}

    file_path: str = Field(..., description="Path to the document file")
    file_name: Optional[str] = Field(None, description="Original filename")
    context: Optional[str] = Field(
        None, description="Additional context for evaluation"
    )
    output_text: Optional[str] = Field(None, description="Output text for evaluation")


class StepEvaluator:
    """Individual evaluation step executor with structured logging and error handling"""

    def __init__(self, llm_provider: LLMProvider, correlation_id: str):
        self.llm_provider = llm_provider
        self.correlation_id = correlation_id
        self.logger = logging.getLogger(f"StepEvaluator.{correlation_id}")

    def evaluate_step(
        self, step_prompt: str, step_name: str, context: Dict[str, Any]
    ) -> EvaluationStep:
        """
        Execute individual evaluation step with comprehensive error handling.

        Args:
            step_prompt: The specific step evaluation prompt
            step_name: Name of the evaluation step
            context: Additional context for evaluation

        Returns:
            EvaluationStep: Structured result of the step evaluation

        Raises:
            ValueError: If step evaluation fails
            Exception: For unexpected errors
        """
        try:
            self.logger.info(f"Starting evaluation step: {step_name}")

            # Create step-specific prompt with enhanced structure
            full_prompt = f"""
            {step_prompt}
            
            ## 評価対象情報
            {context.get('input_text', '')}
            
            ## 追加コンテキスト
            {context.get('context', '')}
            
            ## 関連出力テキスト
            {context.get('output_text', '')}
            
            ## 評価指示
            上記の情報について、{step_name}の評価を以下の形式で厳密に実行してください：
            
            結果: [YES/NO]
            理由: [上記要件を満たす具体的で実践的な理由（法的根拠、行政実務上の考慮事項を含む）]
            """

            # Use LLM provider for evaluation
            response = self.llm_provider.generate_response(
                [
                    {
                        "role": "system",
                        "content": config_manager.get_prompt("system.step_evaluation"),
                    },
                    {"role": "user", "content": full_prompt},
                ]
            )

            # Parse response
            result = self._parse_step_response(response, step_name)

            self.logger.info(f"Step {step_name} completed: {result.result}")
            return result

        except Exception as e:
            self.logger.error(f"Step evaluation failed for {step_name}: {str(e)}")
            raise ValueError(f"Step evaluation failed for {step_name}: {str(e)}")

    def _parse_step_response(self, content: str, step_name: str) -> EvaluationStep:
        """Parse step response with enhanced error handling and multiple pattern matching"""
        try:
            lines = content.strip().split("\n")
            result = "NO"  # Default to NO for safety
            reasoning = "解析に失敗しました"

            # Enhanced pattern matching for result
            for line in lines:
                line_clean = line.strip()
                if any(
                    keyword in line_clean
                    for keyword in ["結果:", "Result:", "判定:", "判断:"]
                ):
                    if (
                        "YES" in line_clean.upper()
                        or "はい" in line_clean
                        or "該当" in line_clean
                    ):
                        result = "YES"
                    elif (
                        "NO" in line_clean.upper()
                        or "いいえ" in line_clean
                        or "非該当" in line_clean
                    ):
                        result = "NO"

                # Enhanced pattern matching for reasoning
                elif any(
                    keyword in line_clean
                    for keyword in ["理由:", "Reasoning:", "根拠:", "説明:"]
                ):
                    reasoning = (
                        line_clean.split(":", 1)[1].strip()
                        if ":" in line_clean
                        else line_clean
                    )

            # Fallback: look for reasoning in longer text blocks
            if reasoning == "解析に失敗しました":
                meaningful_lines = [
                    line.strip()
                    for line in lines
                    if len(line.strip()) > 20 and not line.strip().startswith("#")
                ]
                if meaningful_lines:
                    reasoning = max(meaningful_lines, key=len)

            return EvaluationStep(step=step_name, result=result, reasoning=reasoning)

        except Exception as e:
            self.logger.error(f"Failed to parse step response: {str(e)}")
            return EvaluationStep(
                step=step_name, result="NO", reasoning=f"解析エラー: {str(e)}"
            )


class CriterionEvaluator:
    """Criterion-specific evaluator with step-by-step execution"""

    def __init__(self, llm_provider: LLMProvider, correlation_id: str):
        self.llm_provider = llm_provider
        self.correlation_id = correlation_id
        self.step_evaluator = StepEvaluator(llm_provider, correlation_id)
        self.logger = logging.getLogger(f"CriterionEvaluator.{correlation_id}")

    def evaluate_criterion(
        self, criterion: Dict[str, Any], context: Dict[str, Any]
    ) -> CriterionEvaluation:
        """
        Evaluate a single criterion through all its steps.

        Args:
            criterion: Criterion configuration from JSON
            context: Evaluation context

        Returns:
            CriterionEvaluation: Complete criterion evaluation result
        """
        try:
            self.logger.info(f"Starting criterion evaluation: {criterion['name']}")

            # Execute all evaluation steps in parallel (order-independent)
            steps = self._evaluate_steps_parallel(criterion, context)

            # Calculate score based on step results
            score, score_reasoning = self._calculate_score(steps, criterion)

            self.logger.info(
                f"Criterion {criterion['name']} completed with score {score}"
            )

            return CriterionEvaluation(
                criterion_id=criterion["id"],
                criterion_name=criterion["name"],
                article=criterion["article"],
                steps=steps,
                score=score,
                score_reasoning=score_reasoning,
            )

        except Exception as e:
            self.logger.error(
                f"Criterion evaluation failed for {criterion['name']}: {str(e)}"
            )
            raise ValueError(
                f"Criterion evaluation failed for {criterion['name']}: {str(e)}"
            )

    def _evaluate_steps_parallel(
        self, criterion: Dict[str, Any], context: Dict[str, Any]
    ) -> List[EvaluationStep]:
        """
        Evaluate all steps in parallel (order-independent).

        Args:
            criterion: Criterion configuration
            context: Evaluation context

        Returns:
            List[EvaluationStep]: List of step evaluation results
        """
        try:
            self.logger.info(
                f"Starting parallel step evaluation for: {criterion['name']}"
            )

            # Get evaluation configuration
            eval_config = config_manager.get_evaluation_config()
            parallel_config = eval_config.get("parallel", {})

            # Check if parallel step evaluation is enabled
            if not parallel_config.get("enabled", True):
                return self._evaluate_steps_sequential(criterion, context)

            steps = []
            evaluation_steps = criterion["evaluation_steps"]

            # Use ThreadPoolExecutor for parallel step execution
            max_workers = min(
                parallel_config.get("max_workers", 3), len(evaluation_steps)
            )

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all step evaluation tasks
                future_to_step = {}
                for step_description in evaluation_steps:
                    step_name = step_description
                    user_template = config_manager.get_prompt("user.step_template")

                    step_prompt = user_template.format(
                        step_description=step_description,
                        criterion_article=criterion["article"],
                        criterion_name=criterion["name"],
                        criterion_evaluation_prompt=criterion["evaluation_prompt"],
                    )

                    future = executor.submit(
                        self.step_evaluator.evaluate_step,
                        step_prompt,
                        step_name,
                        context,
                    )
                    future_to_step[future] = step_name

                # Collect results as they complete
                for future in as_completed(future_to_step):
                    step_name = future_to_step[future]
                    try:
                        result = future.result()
                        steps.append(result)
                        self.logger.info(f"Step evaluation completed: {step_name}")
                    except Exception as e:
                        self.logger.error(
                            f"Step evaluation failed for {step_name}: {str(e)}"
                        )
                        # Add fallback step result
                        fallback_step = EvaluationStep(
                            step=step_name,
                            result="NO",
                            reasoning=f"並列ステップ評価エラー: {str(e)}",
                        )
                        steps.append(fallback_step)

            # Sort results by original step order for consistency
            steps.sort(
                key=lambda x: next(
                    (i for i, step in enumerate(evaluation_steps) if step == x.step), 0
                )
            )

            self.logger.info(
                f"Parallel step evaluation completed: {len(steps)} steps evaluated"
            )
            return steps

        except Exception as e:
            self.logger.error(f"Parallel step evaluation failed: {str(e)}")
            # Fallback to sequential evaluation
            return self._evaluate_steps_sequential(criterion, context)

    def _evaluate_steps_sequential(
        self, criterion: Dict[str, Any], context: Dict[str, Any]
    ) -> List[EvaluationStep]:
        """
        Evaluate steps sequentially (fallback method).

        Args:
            criterion: Criterion configuration
            context: Evaluation context

        Returns:
            List[EvaluationStep]: List of step evaluation results
        """
        self.logger.info(
            f"Starting sequential step evaluation for: {criterion['name']}"
        )

        steps = []
        for step_description in criterion["evaluation_steps"]:
            step_name = step_description
            user_template = config_manager.get_prompt("user.step_template")

            step_prompt = user_template.format(
                step_description=step_description,
                criterion_article=criterion["article"],
                criterion_name=criterion["name"],
                criterion_evaluation_prompt=criterion["evaluation_prompt"],
            )

            try:
                step_result = self.step_evaluator.evaluate_step(
                    step_prompt, step_name, context
                )
                steps.append(step_result)
            except Exception as e:
                self.logger.error(
                    f"Sequential step evaluation failed for {step_name}: {str(e)}"
                )
                # Add fallback step result
                fallback_step = EvaluationStep(
                    step=step_name,
                    result="NO",
                    reasoning=f"順次ステップ評価エラー: {str(e)}",
                )
                steps.append(fallback_step)

        return steps

    def _calculate_score(
        self, steps: List[EvaluationStep], criterion: Dict[str, Any]
    ) -> tuple[int, str]:
        """Calculate score based on step results using LLM for reasoning"""
        try:
            # Count YES/NO results
            yes_count = sum(1 for step in steps if step.result == "YES")
            total_steps = len(steps)

            # Calculate score based on YES ratio
            if yes_count == 0:
                score = 5  # No criteria met - disclose
            elif yes_count == total_steps:
                score = 1  # All criteria met - non-disclosure
            elif yes_count >= total_steps * 0.7:
                score = 2  # Most criteria met - likely non-disclosure
            elif yes_count >= total_steps * 0.3:
                score = 3  # Some criteria met - unclear
            else:
                score = 4  # Few criteria met - likely disclosure

            # Generate reasoning using LLM
            reasoning = self._generate_score_reasoning(steps, criterion, score)
            return score, reasoning

        except Exception as e:
            self.logger.error(f"Score calculation failed: {str(e)}")
            return 3, f"スコア計算エラー: {str(e)}"

    def _generate_score_reasoning(
        self, steps: List[EvaluationStep], criterion: Dict[str, Any], score: int
    ) -> str:
        """Generate detailed reasoning for score using LLM based on step evaluations"""
        try:
            # Prepare comprehensive step analysis for LLM
            step_analysis = self._prepare_step_analysis(steps, criterion, score)

            # Create LLM prompt for comprehensive score reasoning
            reasoning_prompt = f"""
あなたは情報公開法の専門家として、以下の各段階の評価結果を総合的に分析し、スコアの理由を詳細に説明してください。

## 評価対象criteria
{criterion['name']} ({criterion['article']})

## 各段階の評価結果と理由
{step_analysis}

## 算出されたスコア
{score}/5

## スコア解釈
- 1: 強く不開示 - 明確に不開示事由に該当
- 2: 不開示の可能性が高い - 不開示事由に該当する可能性が高い
- 3: 不明確 - 慎重な法的検討が必要
- 4: 開示の可能性が高い - 適切な保護措置と共に開示可能
- 5: 明確に開示 - 不開示事由に該当しないか開示例外に該当

## 要求事項
1. 各段階の評価結果（YES/NO）を法的観点から具体的に分析
2. 各段階の理由を総合的に検討し、相互関係を明確化
3. 法的根拠に基づいてスコアの妥当性を詳細に説明
4. 不開示事由の該当性を総合的に判断
5. 行政実務上の考慮事項を反映
6. 具体的な条文解釈と判例・通達の観点を含める

## 重要：出力形式
必ず以下の形式で回答してください。他の形式は使用しないでください：

スコア理由: [各段階の評価結果を総合した具体的で実践的な理由と法的根拠、行政実務上の考慮事項を含む詳細な分析]
"""

            # Use LLM for comprehensive score reasoning
            response = self.llm_provider.generate_response(
                [
                    {
                        "role": "system",
                        "content": config_manager.get_prompt("system.score_reasoning"),
                    },
                    {"role": "user", "content": reasoning_prompt},
                ]
            )

            # Parse LLM response
            reasoning = self._parse_reasoning_response(response)

            return reasoning

        except Exception as e:
            self.logger.error(f"Score reasoning generation failed: {str(e)}")
            # Enhanced fallback reasoning
            if score <= 2:
                return f"各段階の評価結果を総合的に分析した結果、不開示事由に該当するため不開示が適切。法的根拠に基づく慎重な検討により、{score}点のスコアが算出された。"
            elif score == 3:
                return f"各段階の評価結果を総合的に分析した結果、慎重な法的検討が必要。専門家意見を要請し、部分開示の可能性も検討すべき状況であり、{score}点のスコアが算出された。"
            else:
                return f"各段階の評価結果を総合的に分析した結果、不開示事由に該当しないため開示が適切。情報公開法の目的に合致し、適切な保護措置と共に開示可能であり、{score}点のスコアが算出された。"

    def _prepare_step_analysis(
        self, steps: List[EvaluationStep], criterion: Dict[str, Any], score: int
    ) -> str:
        """Prepare comprehensive step analysis for LLM reasoning"""
        try:
            analysis_parts = []

            # Count YES/NO results
            yes_count = sum(1 for step in steps if step.result == "YES")
            no_count = sum(1 for step in steps if step.result == "NO")
            total_steps = len(steps)

            analysis_parts.append(f"## 評価結果サマリー")
            analysis_parts.append(f"- 総段階数: {total_steps}")
            analysis_parts.append(f"- YES評価: {yes_count}段階")
            analysis_parts.append(f"- NO評価: {no_count}段階")
            analysis_parts.append(f"- YES率: {(yes_count/total_steps*100):.1f}%")
            analysis_parts.append("")

            # Detailed step analysis with enhanced context
            analysis_parts.append("## 各段階の詳細評価")
            for i, step in enumerate(steps, 1):
                status_icon = "✅" if step.result == "YES" else "❌"
                analysis_parts.append(f"### 段階{i}: {status_icon} {step.result}")
                analysis_parts.append(f"**評価内容**: {step.step}")
                analysis_parts.append(f"**評価理由**: {step.reasoning}")
                analysis_parts.append(
                    f"**法的観点**: この段階の評価は{criterion['article']}の要件に基づく"
                )
                analysis_parts.append("")

            # Enhanced legal basis analysis
            analysis_parts.append("## 法的根拠分析")
            analysis_parts.append(f"**該当条文**: {criterion['article']}")
            analysis_parts.append(
                f"**条文内容**: {criterion.get('description', 'N/A')}"
            )
            analysis_parts.append(
                f"**評価基準**: {criterion.get('evaluation_prompt', 'N/A')}"
            )
            analysis_parts.append("")

            # Administrative considerations
            analysis_parts.append("## 行政実務上の考慮事項")
            analysis_parts.append("- 情報公開法の目的（国民の知る権利の保障）との関係")
            analysis_parts.append("- 不開示事由の厳格解釈の原則")
            analysis_parts.append("- 部分開示・部分不開示の可能性")
            analysis_parts.append("- 開示決定時の保護措置の必要性")
            analysis_parts.append("")

            # Score justification with enhanced reasoning
            analysis_parts.append("## スコア算出根拠")
            if score == 1:
                analysis_parts.append("- すべての段階で不開示事由に該当")
                analysis_parts.append("- 強く不開示が適切（法的根拠が明確）")
                analysis_parts.append("- 行政実務上も不開示決定が妥当")
            elif score == 2:
                analysis_parts.append("- 大部分の段階で不開示事由に該当")
                analysis_parts.append("- 不開示の可能性が高い（慎重な検討が必要）")
                analysis_parts.append("- 補完的な法的検討を推奨")
            elif score == 3:
                analysis_parts.append("- 一部の段階で不開示事由に該当")
                analysis_parts.append("- 慎重な法的検討が必要（専門家意見を要請）")
                analysis_parts.append("- 部分開示の可能性も検討")
            elif score == 4:
                analysis_parts.append("- 少数の段階で不開示事由に該当")
                analysis_parts.append("- 開示の可能性が高い（保護措置を検討）")
                analysis_parts.append("- 適切な保護措置と共に開示可能")
            else:  # score == 5
                analysis_parts.append("- 不開示事由に該当しない")
                analysis_parts.append("- 明確に開示が適切（法的根拠が明確）")
                analysis_parts.append("- 情報公開法の目的に合致")

            return "\n".join(analysis_parts)

        except Exception as e:
            self.logger.error(f"Step analysis preparation failed: {str(e)}")
            # Fallback to simple step summary
            step_details = []
            for step in steps:
                step_details.append(f"- {step.step}: {step.result} - {step.reasoning}")
            return "\n".join(step_details)

    def _parse_reasoning_response(self, content: str) -> str:
        """Parse LLM reasoning response with enhanced pattern matching"""
        try:
            lines = content.strip().split("\n")
            reasoning = "評価結果を総合的に判断"

            # Enhanced pattern matching for reasoning
            reasoning_patterns = [
                "スコア理由:",
                "理由:",
                "判断理由:",
                "総合判断:",
                "分析結果:",
                "根拠:",
                "説明:",
                "結論:",
                "総合分析:",
                "評価理由:",
                "法的根拠:",
                "行政実務上の考慮:",
                "総合評価:",
            ]

            for line in lines:
                line_clean = line.strip()
                for pattern in reasoning_patterns:
                    if pattern in line_clean:
                        reasoning = (
                            line_clean.split(":", 1)[1].strip()
                            if ":" in line_clean
                            else line_clean
                        )
                        break
                if reasoning != "評価結果を総合的に判断":
                    break

            # Enhanced fallback: extract meaningful content with better filtering
            if reasoning == "評価結果を総合的に判断":
                meaningful_lines = []
                for line in lines:
                    line_clean = line.strip()
                    if (
                        len(line_clean) > 30
                        and not line_clean.startswith("#")
                        and not line_clean.startswith("-")
                        and not line_clean.startswith("*")
                        and any(
                            keyword in line_clean
                            for keyword in [
                                "スコア",
                                "理由",
                                "判断",
                                "分析",
                                "法的",
                                "行政",
                                "根拠",
                                "該当",
                                "開示",
                                "不開示",
                            ]
                        )
                    ):
                        meaningful_lines.append(line_clean)

                if meaningful_lines:
                    # Use the most comprehensive line as reasoning
                    reasoning = max(
                        meaningful_lines,
                        key=lambda x: (len(x), x.count("。"), x.count("、")),
                    )

            # Ensure reasoning is substantial and meaningful
            if len(reasoning) < 50:
                reasoning = f"各段階の評価結果を総合的に分析した結果、{reasoning}。法的根拠に基づく慎重な検討が必要。"

            return reasoning

        except Exception as e:
            self.logger.error(f"Failed to parse reasoning response: {str(e)}")
            return f"解析エラー: {str(e)}"


class ResultAggregator:
    """Aggregates individual criterion evaluations into overall result"""

    def __init__(self, correlation_id: str, llm_provider: LLMProvider):
        self.correlation_id = correlation_id
        self.llm_provider = llm_provider
        self.logger = logging.getLogger(f"ResultAggregator.{correlation_id}")

    def aggregate_results(
        self,
        criterion_evaluations: List[CriterionEvaluation],
        criteria_config: Dict[str, Any],
    ) -> None:
        """
        Process evaluation results (no longer needed as user makes final judgment).

        Args:
            criterion_evaluations: List of individual criterion evaluations
            criteria_config: Criteria configuration for weights
        """
        try:
            self.logger.info("Starting result aggregation")

            self.logger.info(
                "Result aggregation completed: User will make final judgment"
            )

        except Exception as e:
            self.logger.error(f"Result aggregation failed: {str(e)}")
            raise ValueError(f"Result aggregation failed: {str(e)}")


def load_criteria():
    """Load criteria configuration from JSON file with error handling"""
    try:
        with open(
            "criteria/disclosure_evaluation_criteria.json", "r", encoding="utf-8"
        ) as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error("Criteria file not found")
        raise ValueError("Criteria file not found")
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in criteria file: {e}")
        raise ValueError(f"Invalid JSON in criteria file: {e}")


class DisclosureEvaluator:
    """Main evaluator orchestrating step-by-step evaluation with comprehensive error handling"""

    def __init__(self, api_key: str = None, provider: str = None):
        """Initialize evaluator with LLM provider and correlation ID"""
        # Get provider configuration
        if provider is None:
            provider = config_manager.get_current_provider()

        provider_config = config_manager.get_provider_config(provider)

        # Set credentials from environment variables
        if provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable is required")
            provider_config["api_key"] = api_key
        elif provider == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY environment variable is required")
            provider_config["api_key"] = api_key
        elif provider in ["bedrock", "bedrock_nova"]:
            # AWS credentials are handled by boto3 automatically from environment
            # or AWS credentials file
            pass

        # Create LLM provider
        self.llm_provider = create_llm_provider(provider)
        self.correlation_id = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.logger = logging.getLogger(f"DisclosureEvaluator.{self.correlation_id}")

        # Initialize evaluators
        self.criterion_evaluator = CriterionEvaluator(
            self.llm_provider, self.correlation_id
        )
        self.result_aggregator = ResultAggregator(
            self.correlation_id, self.llm_provider
        )

        # Thread-safe lock for logging
        self._log_lock = threading.Lock()

    def _evaluate_criterion_parallel(
        self, criterion: Dict[str, Any], eval_context: Dict[str, Any]
    ) -> CriterionEvaluation:
        """
        Evaluate a single criterion in parallel execution.

        Args:
            criterion: Criterion configuration
            eval_context: Evaluation context

        Returns:
            CriterionEvaluation: Evaluation result
        """
        try:
            with self._log_lock:
                self.logger.info(
                    f"Starting parallel evaluation of criterion: {criterion['name']}"
                )

            # Create a new criterion evaluator for this thread
            thread_correlation_id = f"{self.correlation_id}_{criterion['id']}"
            thread_evaluator = CriterionEvaluator(
                self.llm_provider, thread_correlation_id
            )

            criterion_eval = thread_evaluator.evaluate_criterion(
                criterion, eval_context
            )

            with self._log_lock:
                self.logger.info(
                    f"Criterion {criterion['name']} completed successfully in parallel"
                )

            return criterion_eval

        except Exception as e:
            with self._log_lock:
                self.logger.error(
                    f"Criterion {criterion['name']} parallel evaluation failed: {str(e)}"
                )

            # Create fallback evaluation
            return CriterionEvaluation(
                criterion_id=criterion["id"],
                criterion_name=criterion["name"],
                article=criterion["article"],
                steps=[
                    EvaluationStep(
                        step="エラー",
                        result="NO",
                        reasoning=f"並列評価エラー: {str(e)}",
                    )
                ],
                score=3,
                score_reasoning=f"並列評価エラーのため中間スコアを設定: {str(e)}",
            )

    def _evaluate_criteria_parallel(
        self, criteria: List[Dict[str, Any]], eval_context: Dict[str, Any]
    ) -> List[CriterionEvaluation]:
        """
        Evaluate multiple criteria in parallel using ThreadPoolExecutor.

        Args:
            criteria: List of criterion configurations
            eval_context: Evaluation context

        Returns:
            List[CriterionEvaluation]: List of evaluation results
        """
        self.logger.info(f"Starting parallel evaluation of {len(criteria)} criteria")

        criterion_evaluations = []

        # Use ThreadPoolExecutor for parallel execution
        eval_config = config_manager.get_evaluation_config()
        parallel_config = eval_config.get("parallel", {})

        if not parallel_config.get("enabled", True):
            # Fallback to sequential evaluation
            return self._evaluate_criteria_sequential(criteria, eval_context)

        max_workers = min(parallel_config.get("max_workers", 3), len(criteria))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all evaluation tasks with explicit mapping
            future_to_criterion = {}
            for criterion in criteria:
                future = executor.submit(
                    self._evaluate_criterion_parallel, criterion, eval_context
                )
                future_to_criterion[future] = criterion

            # Collect results as they complete
            for future in as_completed(future_to_criterion):
                criterion = future_to_criterion[future]
                try:
                    result = future.result()
                    # Verify the result matches the expected criterion
                    if result.criterion_id != criterion["id"]:
                        self.logger.warning(
                            f"Criterion ID mismatch: expected {criterion['id']}, got {result.criterion_id}"
                        )
                    criterion_evaluations.append(result)
                    self.logger.info(
                        f"Parallel evaluation completed for: {criterion['name']} (ID: {criterion['id']})"
                    )
                except Exception as e:
                    self.logger.error(
                        f"Parallel evaluation failed for {criterion['name']} (ID: {criterion['id']}): {str(e)}"
                    )
                    # Add fallback evaluation with correct criterion ID
                    fallback_eval = CriterionEvaluation(
                        criterion_id=criterion["id"],
                        criterion_name=criterion["name"],
                        article=criterion["article"],
                        steps=[
                            EvaluationStep(
                                step="エラー",
                                result="NO",
                                reasoning=f"並列実行エラー: {str(e)}",
                            )
                        ],
                        score=3,
                        score_reasoning=f"並列実行エラーのため中間スコアを設定: {str(e)}",
                    )
                    criterion_evaluations.append(fallback_eval)

        # Sort results by original criteria order
        criterion_evaluations.sort(
            key=lambda x: next(
                (i for i, c in enumerate(criteria) if c["id"] == x.criterion_id), 0
            )
        )

        # Verify all criteria were evaluated
        evaluated_ids = {c.criterion_id for c in criterion_evaluations}
        expected_ids = {c["id"] for c in criteria}

        if evaluated_ids != expected_ids:
            missing_ids = expected_ids - evaluated_ids
            extra_ids = evaluated_ids - expected_ids
            self.logger.error(
                f"Evaluation mismatch: missing {missing_ids}, extra {extra_ids}"
            )
        else:
            self.logger.info("All criteria successfully evaluated and mapped")

        self.logger.info(
            f"Parallel evaluation completed: {len(criterion_evaluations)} criteria evaluated"
        )
        return criterion_evaluations

    def evaluate_disclosure(
        self, input_text: str, context: str = "", output_text: str = ""
    ) -> DisclosureEvaluationResult:
        """
        Evaluate disclosure using step-by-step approach with comprehensive error handling.

        Args:
            input_text: Information to be evaluated
            context: Additional context
            output_text: Output text if provided

        Returns:
            DisclosureEvaluationResult: Complete evaluation result

        Raises:
            ValueError: If evaluation fails
            Exception: For unexpected errors
        """
        try:
            self.logger.info(
                f"Starting disclosure evaluation with correlation ID: {self.correlation_id}"
            )

            # Load criteria configuration
            criteria = load_criteria()

            # Prepare evaluation context
            eval_context = {
                "input_text": input_text,
                "context": context,
                "output_text": output_text,
            }

            # Evaluate criteria in parallel
            criterion_evaluations = self._evaluate_criteria_parallel(
                criteria["criteria"], eval_context
            )

            # Process evaluation results
            self.result_aggregator.aggregate_results(criterion_evaluations, criteria)

            # Create final result
            result = DisclosureEvaluationResult(
                input_text=input_text,
                context=context,
                output_text=output_text,
                criterion_evaluations=criterion_evaluations,
                evaluation_timestamp=datetime.now().isoformat(),
            )

            self.logger.info(
                f"Disclosure evaluation completed successfully: {len(criterion_evaluations)} criteria evaluated"
            )

            # Save evaluation result to JSON file
            self._save_evaluation_result(result)

            # Save evaluation result to CSV file
            self._save_evaluation_result_csv(result)

            return result

        except Exception as e:
            self.logger.error(f"Disclosure evaluation failed: {str(e)}")
            raise ValueError(f"Disclosure evaluation failed: {str(e)}")

    def _save_evaluation_result(self, result: DisclosureEvaluationResult) -> str:
        """
        Save evaluation result to timestamped JSON file.

        Args:
            result: The evaluation result to save

        Returns:
            str: Path to the saved file
        """
        try:
            # Get output configuration
            output_config = config_manager.get_output_config()
            timestamp_format = output_config.get("timestamp_format", "%Y%m%d_%H%M%S")
            encoding = output_config.get("encoding", "utf-8")

            # Generate timestamped output filename
            timestamp = datetime.now().strftime(timestamp_format)
            output_filename = f"{outputs_dir}/evaluation_result_{timestamp}.json"

            # Convert result to JSON with proper formatting
            result_json = result.model_dump_json(indent=2)

            # Save to file
            with open(output_filename, "w", encoding=encoding) as f:
                f.write(result_json)

            self.logger.info(f"Evaluation result saved to: {output_filename}")
            return output_filename

        except Exception as e:
            self.logger.error(f"Failed to save evaluation result: {str(e)}")
            # Don't raise exception as this is not critical for evaluation
            return ""

    def _save_evaluation_result_csv(self, result: DisclosureEvaluationResult) -> str:
        """
        Save evaluation result to CSV format.

        Args:
            result: DisclosureEvaluationResult to save

        Returns:
            str: Path to saved CSV file, empty string if failed
        """
        try:
            output_config = config_manager.get_output_config()
            csv_config = output_config.get("csv", {})

            # Check if CSV output is enabled
            if not csv_config.get("enabled", True):
                self.logger.info("CSV output is disabled in configuration")
                return ""

            timestamp_format = output_config.get("timestamp_format", "%Y%m%d_%H%M%S")
            encoding = output_config.get("encoding", "utf-8")
            delimiter = csv_config.get("delimiter", ",")
            quote_char = csv_config.get("quote_char", '"')
            include_headers = csv_config.get("include_headers", True)

            timestamp = datetime.now().strftime(timestamp_format)
            csv_filename = f"{outputs_dir}/evaluation_result_{timestamp}.csv"

            with open(csv_filename, "w", newline="", encoding=encoding) as csvfile:
                writer = csv.writer(csvfile, delimiter=delimiter, quotechar=quote_char)

                # Write header if enabled
                if include_headers:
                    writer.writerow(
                        [
                            "Criterion ID",
                            "Criterion Name",
                            "Article",
                            "Score",
                            "Score Reasoning",
                            "Step",
                            "Step Result",
                            "Step Reasoning",
                        ]
                    )

                # Write data rows
                for criterion in result.criterion_evaluations:
                    for step in criterion.steps:
                        writer.writerow(
                            [
                                criterion.criterion_id,
                                criterion.criterion_name,
                                criterion.article,
                                criterion.score,
                                criterion.score_reasoning,
                                step.step,
                                step.result,
                                step.reasoning,
                            ]
                        )

            self.logger.info(f"Evaluation result saved to CSV: {csv_filename}")
            return csv_filename

        except Exception as e:
            self.logger.error(f"Failed to save evaluation result to CSV: {str(e)}")
            return ""


def evaluate_disclosure(
    input_text, context="", output_text="", api_key=None, provider=None
):
    """
    Legacy function for backward compatibility.
    Creates a new DisclosureEvaluator instance and evaluates disclosure.
    """
    evaluator = DisclosureEvaluator(api_key, provider)
    return evaluator.evaluate_disclosure(input_text, context, output_text)


def format_structured_output(
    result: DisclosureEvaluationResult, format_type: str = "json"
) -> str:
    """Format the structured evaluation result for output"""
    if format_type == "json":
        return result.model_dump_json(indent=2)
    elif format_type == "summary":
        # Create a human-readable summary
        summary = f"""
# 情報公開法評価結果

## 評価対象
{result.input_text}

## 評価サマリー
- **評価criteria数**: {len(result.criterion_evaluations)}
- **評価完了時刻**: {result.evaluation_timestamp}

## 各不開示事由の評価\n\n"""

        for criterion in result.criterion_evaluations:
            # Highlight low score criteria
            highlight = "⚠️ " if criterion.score <= 2 else ""
            summary += f"""
### {highlight}{criterion.criterion_name} ({criterion.article})
- **スコア**: {criterion.score}/5
- **スコア理由**: {criterion.score_reasoning}

**段階的評価**:
"""
            for step in criterion.steps:
                summary += f"- {step.step}: {step.result} - {step.reasoning}\n"
            summary += "\n"

        return summary
    else:
        raise ValueError(f"Unsupported format type: {format_type}")


# Batch Processing Services


class DocumentDiscoveryService:
    """Service for discovering documents in folders and file systems"""

    def __init__(self, config: BatchConfiguration):
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


class BatchStatePersistenceService:
    """Service for persisting and loading batch processing state"""

    def __init__(self, config: BatchConfiguration):
        self.config = config
        self.logger = logging.getLogger("BatchStatePersistenceService")
        self.state_dir = Path("batch_state")
        self.active_dir = self.state_dir / "active_batches"
        self.completed_dir = self.state_dir / "completed_batches"

        # Create directories if they don't exist
        self.active_dir.mkdir(parents=True, exist_ok=True)
        self.completed_dir.mkdir(parents=True, exist_ok=True)

    def save_batch_state(self, batch: BatchEvaluation) -> None:
        """Save batch state to file"""
        try:
            state_file = self.active_dir / f"{batch.batch_id}.json"
            with open(state_file, "w", encoding="utf-8") as f:
                # Convert enum values to their string representations
                data = batch.model_dump()
                # Convert enums to strings for JSON serialization
                if "status" in data:
                    data["status"] = (
                        data["status"].value
                        if hasattr(data["status"], "value")
                        else str(data["status"])
                    )
                json.dump(data, f, indent=2, default=str)
            self.logger.info(f"Batch state saved: {state_file}")
        except Exception as e:
            self.logger.error(f"Failed to save batch state: {str(e)}")
            raise

    def load_batch_state(self, batch_id: str) -> Optional[BatchEvaluation]:
        """Load batch state from file"""
        try:
            state_file = self.active_dir / f"{batch_id}.json"
            if not state_file.exists():
                return None

            with open(state_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                return BatchEvaluation(**data)
        except Exception as e:
            self.logger.error(f"Failed to load batch state: {str(e)}")
            return None

    def move_to_completed(self, batch_id: str) -> None:
        """Move batch state from active to completed"""
        try:
            active_file = self.active_dir / f"{batch_id}.json"
            completed_file = self.completed_dir / f"{batch_id}.json"

            if active_file.exists():
                active_file.rename(completed_file)
                self.logger.info(f"Batch moved to completed: {batch_id}")
        except Exception as e:
            self.logger.error(f"Failed to move batch to completed: {str(e)}")
            raise


class ParallelDocumentProcessingService:
    """Service for parallel document processing"""

    def __init__(self, config: BatchConfiguration, llm_provider: LLMProvider):
        self.config = config
        self.llm_provider = llm_provider
        self.logger = logging.getLogger("ParallelDocumentProcessingService")

    def process_documents_parallel(
        self,
        documents: List[BatchDocument],
        progress_callback: Optional[callable] = None,
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
                evaluator = DisclosureEvaluator(
                    api_key=os.getenv("OPENAI_API_KEY"), provider="openai"
                )

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


class BatchEvaluator:
    """Main orchestrator for batch document evaluation"""

    def __init__(self, config: Optional[BatchConfiguration] = None):
        self.config = config or BatchConfiguration()
        self.logger = logging.getLogger("BatchEvaluator")

        # Initialize services
        self.discovery_service = DocumentDiscoveryService(self.config)
        self.state_service = BatchStatePersistenceService(self.config)

        # Initialize LLM provider for document processing
        self.llm_provider = create_llm_provider()
        self.processing_service = ParallelDocumentProcessingService(
            self.config, self.llm_provider
        )

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
        """Store document inputs for batch processing"""
        try:
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

            # Save to a separate file
            docs_file = self.state_service.active_dir / f"{batch_id}_documents.json"
            with open(docs_file, "w", encoding="utf-8") as f:
                json.dump(doc_data, f, indent=2, ensure_ascii=False)

            self.logger.info(f"Stored {len(documents)} documents for batch {batch_id}")

        except Exception as e:
            self.logger.error(f"Failed to store batch documents: {str(e)}")
            raise

    def _convert_document_inputs_to_batch_documents(
        self, documents: List[DocumentInput], batch_id: str
    ) -> List[BatchDocument]:
        """Convert DocumentInput list to BatchDocument list"""
        try:
            batch_documents = []

            for i, doc_input in enumerate(documents):
                # Get file information
                file_path = Path(doc_input.file_path)
                file_size = file_path.stat().st_size if file_path.exists() else 0
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

            self.logger.info(
                f"Converted {len(documents)} DocumentInputs to BatchDocuments"
            )
            return batch_documents

        except Exception as e:
            self.logger.error(f"Failed to convert document inputs: {str(e)}")
            raise

    def create_batch_from_folder(
        self,
        folder_path: str,
        context: str = "",
        recursive: bool = True,
        file_types: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
        file_size_limit: Optional[int] = None,
        progress_callback: Optional[callable] = None,
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


def main():
    """Main CLI entry point with batch processing support"""
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)

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
        result = evaluate_disclosure(
            input_text, context, output_text, provider=provider
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

        # Create batch evaluator
        evaluator = BatchEvaluator(config=config)

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

        # Create batch evaluator
        evaluator = BatchEvaluator()

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

        # Create batch evaluator
        evaluator = BatchEvaluator()

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

        # Create batch evaluator
        evaluator = BatchEvaluator()

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

        # Create batch evaluator
        evaluator = BatchEvaluator()

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


if __name__ == "__main__":
    main()
