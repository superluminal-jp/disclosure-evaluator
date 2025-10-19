"""
Individual evaluation step executor with structured logging and error handling.
"""

import logging
from typing import Dict, Any
from ..models.evaluation import EvaluationStep
from ..llm import LLMProvider


class StepEvaluator:
    """Individual evaluation step executor with structured logging and error handling"""

    def __init__(self, llm_provider: LLMProvider, correlation_id: str, config_manager):
        self.llm_provider = llm_provider
        self.correlation_id = correlation_id
        self.config_manager = config_manager
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
                        "content": self.config_manager.get_prompt(
                            "system.step_evaluation"
                        ),
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
