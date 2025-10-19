"""
Evaluation models for disclosure assessment.
"""

from typing import List, Literal
from pydantic import BaseModel, Field


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
