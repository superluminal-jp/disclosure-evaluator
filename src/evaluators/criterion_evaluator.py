"""
Criterion-specific evaluator with step-by-step execution.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any
from ..models.evaluation import EvaluationStep, CriterionEvaluation
from ..llm import LLMProvider
from .step_evaluator import StepEvaluator


class CriterionEvaluator:
    """Criterion-specific evaluator with step-by-step execution"""

    def __init__(self, llm_provider: LLMProvider, correlation_id: str, config_manager):
        self.llm_provider = llm_provider
        self.correlation_id = correlation_id
        self.config_manager = config_manager
        self.step_evaluator = StepEvaluator(
            llm_provider, correlation_id, config_manager
        )
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
            eval_config = self.config_manager.get_evaluation_config()
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
                    user_template = self.config_manager.get_prompt("user.step_template")

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
            user_template = self.config_manager.get_prompt("user.step_template")

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
                        "content": self.config_manager.get_prompt(
                            "system.score_reasoning"
                        ),
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
