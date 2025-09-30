#!/usr/bin/env python3
import json
import sys
import os
import csv
from datetime import datetime
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

    def get_prompts_config(self) -> Dict[str, Any]:
        """Get prompts configuration"""
        return self.config.get("prompts", {})

    def get_output_config(self) -> Dict[str, Any]:
        """Get output configuration"""
        return self.config.get("output", {})

    def load_prompt(self, prompt_path: str) -> str:
        """Load prompt content from file"""
        try:
            with open(prompt_path, "r", encoding="utf-8") as f:
                return f.read().strip()
        except FileNotFoundError:
            raise ValueError(f"Prompt file not found: {prompt_path}")
        except Exception as e:
            raise ValueError(f"Error reading prompt file {prompt_path}: {e}")

    def get_prompt(self, prompt_key: str) -> str:
        """Get prompt content by key (e.g., 'system.step_evaluation')"""
        prompts_config = self.config.get("prompts", {})
        prompt_path = self.get(f"prompts.{prompt_key}")

        if not prompt_path:
            raise ValueError(f"Prompt key not found: {prompt_key}")

        return self.load_prompt(prompt_path)

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


def main():
    if len(sys.argv) < 2:
        print(
            "Usage: python evaluator.py <input_text> [context] [output_text] [--format json|summary] [--provider openai|anthropic|bedrock|bedrock_nova]"
        )
        sys.exit(1)

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


if __name__ == "__main__":
    main()
