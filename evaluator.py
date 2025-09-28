#!/usr/bin/env python3
import json
import sys
import os
from datetime import datetime
from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field
from openai import OpenAI
import logging

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='{"timestamp": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s", "correlation_id": "%(name)s"}',
    handlers=[
        logging.FileHandler("logs/evaluation.log", encoding="utf-8"),
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

    def __init__(self, client: OpenAI, correlation_id: str):
        self.client = client
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

            # Create step-specific prompt
            full_prompt = f"""
            {step_prompt}
            
            ## 評価対象情報
            {context.get('input_text', '')}
            
            ## コンテキスト
            {context.get('context', '')}
            
            ## 出力テキスト
            {context.get('output_text', '')}
            
            上記の情報について、{step_name}の評価を以下の形式で行ってください：
            
            結果: [YES/NO]
            理由: [詳細な理由]
            """

            response = self.client.chat.completions.create(
                model="gpt-5",
                messages=[
                    {
                        "role": "system",
                        "content": "あなたは情報公開法の専門家です。各評価ステップを正確に実行してください。",
                    },
                    {"role": "user", "content": full_prompt},
                ],
            )

            # Parse response
            content = response.choices[0].message.content
            result = self._parse_step_response(content, step_name)

            self.logger.info(f"Step {step_name} completed: {result.result}")
            return result

        except Exception as e:
            self.logger.error(f"Step evaluation failed for {step_name}: {str(e)}")
            raise ValueError(f"Step evaluation failed for {step_name}: {str(e)}")

    def _parse_step_response(self, content: str, step_name: str) -> EvaluationStep:
        """Parse step response with defensive error handling"""
        try:
            lines = content.strip().split("\n")
            result = "NO"  # Default to NO for safety
            reasoning = "解析に失敗しました"

            for line in lines:
                if "結果:" in line or "Result:" in line:
                    if "YES" in line.upper():
                        result = "YES"
                    elif "NO" in line.upper():
                        result = "NO"
                elif "理由:" in line or "Reasoning:" in line:
                    reasoning = line.split(":", 1)[1].strip()

            return EvaluationStep(step=step_name, result=result, reasoning=reasoning)

        except Exception as e:
            self.logger.error(f"Failed to parse step response: {str(e)}")
            return EvaluationStep(
                step=step_name, result="NO", reasoning=f"解析エラー: {str(e)}"
            )


class CriterionEvaluator:
    """Criterion-specific evaluator with step-by-step execution"""

    def __init__(self, client: OpenAI, correlation_id: str):
        self.client = client
        self.correlation_id = correlation_id
        self.step_evaluator = StepEvaluator(client, correlation_id)
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

            # Execute all evaluation steps
            steps = []
            for step_description in criterion["evaluation_steps"]:
                step_name = step_description
                step_prompt = f"""
                以下の評価ステップを実行してください：
                
                {step_description}
                
                評価基準: {criterion['evaluation_prompt']}
                """

                step_result = self.step_evaluator.evaluate_step(
                    step_prompt, step_name, context
                )
                steps.append(step_result)

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
1. 各段階の評価結果（YES/NO）を分析
2. 各段階の理由を総合的に検討
3. 法的根拠に基づいてスコアの妥当性を説明
4. 不開示事由の該当性を総合的に判断

## 重要：出力形式
必ず以下の形式で回答してください。他の形式は使用しないでください：

スコア理由: [各段階の評価結果を総合した詳細な理由と法的根拠]

例：
スコア理由: 段階1でYES、段階2でYES、段階3でNOと評価され、個人情報保護の観点から不開示事由に該当する可能性が高いと判断されたため、スコア3（慎重な法的検討が必要）としました。
"""

            # Use LLM for comprehensive score reasoning
            response = self.client.chat.completions.create(
                model="gpt-5",
                messages=[
                    {
                        "role": "system",
                        "content": "あなたは情報公開法の専門家です。各段階の評価結果を総合的に分析し、スコアの理由を詳細に説明してください。各段階の理由を踏まえた総合的な判断を行ってください。",
                    },
                    {"role": "user", "content": reasoning_prompt},
                ],
            )

            # Parse LLM response
            content = response.choices[0].message.content
            reasoning = self._parse_reasoning_response(content)

            return reasoning

        except Exception as e:
            self.logger.error(f"Score reasoning generation failed: {str(e)}")
            # Fallback to simple reasoning
            if score <= 2:
                return "不開示事由に該当するため不開示が適切"
            elif score == 3:
                return "慎重な法的検討が必要"
            else:
                return "不開示事由に該当しないため開示が適切"

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

            # Detailed step analysis
            analysis_parts.append("## 各段階の詳細評価")
            for i, step in enumerate(steps, 1):
                status_icon = "✅" if step.result == "YES" else "❌"
                analysis_parts.append(f"### 段階{i}: {status_icon} {step.result}")
                analysis_parts.append(f"**評価内容**: {step.step}")
                analysis_parts.append(f"**評価理由**: {step.reasoning}")
                analysis_parts.append("")

            # Legal basis analysis
            analysis_parts.append("## 法的根拠分析")
            analysis_parts.append(f"**該当条文**: {criterion['article']}")
            analysis_parts.append(
                f"**条文内容**: {criterion.get('description', 'N/A')}"
            )
            analysis_parts.append("")

            # Score justification
            analysis_parts.append("## スコア算出根拠")
            if score == 1:
                analysis_parts.append("- すべての段階で不開示事由に該当")
                analysis_parts.append("- 強く不開示が適切")
            elif score == 2:
                analysis_parts.append("- 大部分の段階で不開示事由に該当")
                analysis_parts.append("- 不開示の可能性が高い")
            elif score == 3:
                analysis_parts.append("- 一部の段階で不開示事由に該当")
                analysis_parts.append("- 慎重な法的検討が必要")
            elif score == 4:
                analysis_parts.append("- 少数の段階で不開示事由に該当")
                analysis_parts.append("- 開示の可能性が高い")
            else:  # score == 5
                analysis_parts.append("- 不開示事由に該当しない")
                analysis_parts.append("- 明確に開示が適切")

            return "\n".join(analysis_parts)

        except Exception as e:
            self.logger.error(f"Step analysis preparation failed: {str(e)}")
            # Fallback to simple step summary
            step_details = []
            for step in steps:
                step_details.append(f"- {step.step}: {step.result} - {step.reasoning}")
            return "\n".join(step_details)

    def _parse_reasoning_response(self, content: str) -> str:
        """Parse LLM reasoning response with defensive error handling"""
        try:
            lines = content.strip().split("\n")
            reasoning = "評価結果を総合的に判断"

            # Try multiple patterns to find reasoning
            for line in lines:
                line = line.strip()
                if "スコア理由:" in line:
                    reasoning = line.split(":", 1)[1].strip()
                    break
                elif "理由:" in line:
                    reasoning = line.split(":", 1)[1].strip()
                    break
                elif "判断理由:" in line:
                    reasoning = line.split(":", 1)[1].strip()
                    break
                elif "総合判断:" in line:
                    reasoning = line.split(":", 1)[1].strip()
                    break

            # If no specific pattern found, try to extract meaningful content
            if reasoning == "評価結果を総合的に判断":
                # Look for the longest meaningful line
                meaningful_lines = []
                for line in lines:
                    line = line.strip()
                    if (
                        len(line) > 20
                        and not line.startswith("#")
                        and not line.startswith("-")
                    ):
                        meaningful_lines.append(line)

                if meaningful_lines:
                    # Use the longest meaningful line as reasoning
                    reasoning = max(meaningful_lines, key=len)

            return reasoning

        except Exception as e:
            self.logger.error(f"Failed to parse reasoning response: {str(e)}")
            return f"解析エラー: {str(e)}"


class ResultAggregator:
    """Aggregates individual criterion evaluations into overall result"""

    def __init__(self, correlation_id: str, client: OpenAI):
        self.correlation_id = correlation_id
        self.client = client
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

    def __init__(self, api_key: str = None):
        """Initialize evaluator with OpenAI client and correlation ID"""
        if not api_key:
            api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise ValueError("OpenAI API key is required")

        self.client = OpenAI(api_key=api_key)
        self.correlation_id = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.logger = logging.getLogger(f"DisclosureEvaluator.{self.correlation_id}")

        # Initialize evaluators
        self.criterion_evaluator = CriterionEvaluator(self.client, self.correlation_id)
        self.result_aggregator = ResultAggregator(self.correlation_id, self.client)

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

            # Evaluate each criterion individually
            criterion_evaluations = []
            for criterion in criteria["criteria"]:
                try:
                    self.logger.info(f"Evaluating criterion: {criterion['name']}")
                    criterion_eval = self.criterion_evaluator.evaluate_criterion(
                        criterion, eval_context
                    )
                    criterion_evaluations.append(criterion_eval)
                    self.logger.info(
                        f"Criterion {criterion['name']} completed successfully"
                    )
                except Exception as e:
                    self.logger.error(
                        f"Criterion {criterion['name']} evaluation failed: {str(e)}"
                    )
                    # Create fallback evaluation
                    fallback_eval = CriterionEvaluation(
                        criterion_id=criterion["id"],
                        criterion_name=criterion["name"],
                        article=criterion["article"],
                        steps=[
                            EvaluationStep(
                                step="エラー",
                                result="NO",
                                reasoning=f"評価エラー: {str(e)}",
                            )
                        ],
                        score=3,
                        score_reasoning=f"評価エラーのため中間スコアを設定: {str(e)}",
                    )
                    criterion_evaluations.append(fallback_eval)

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
            return result

        except Exception as e:
            self.logger.error(f"Disclosure evaluation failed: {str(e)}")
            raise ValueError(f"Disclosure evaluation failed: {str(e)}")


def evaluate_disclosure(input_text, context="", output_text="", api_key=None):
    """
    Legacy function for backward compatibility.
    Creates a new DisclosureEvaluator instance and evaluates disclosure.
    """
    evaluator = DisclosureEvaluator(api_key)
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
            "Usage: python evaluator.py <input_text> [context] [output_text] [--format json|summary]"
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

    try:
        result = evaluate_disclosure(input_text, context, output_text)
        formatted_output = format_structured_output(result, format_type)
        print(formatted_output)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
