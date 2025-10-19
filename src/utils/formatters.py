"""
Output formatting utilities.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..models.evaluation import DisclosureEvaluationResult


def format_structured_output(
    result: "DisclosureEvaluationResult", format_type: str = "json"
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

## 評価コンテキスト
{result.context}

## 評価出力
{result.output_text}

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
