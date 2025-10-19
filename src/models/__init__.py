"""
Models package for disclosure evaluator.
"""

from .evaluation import (
    EvaluationStep,
    CriterionEvaluation,
    OverallEvaluation,
    DisclosureEvaluationResult,
)

from .batch import (
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
)

__all__ = [
    # Evaluation models
    "EvaluationStep",
    "CriterionEvaluation",
    "OverallEvaluation",
    "DisclosureEvaluationResult",
    # Batch models
    "BatchStatus",
    "DocumentStatus",
    "ProcessingPhase",
    "BatchConfiguration",
    "BatchSummaryStatistics",
    "DocumentResult",
    "DocumentError",
    "BatchEvaluation",
    "BatchDocument",
    "BatchProgress",
    "BatchResult",
    "DocumentInput",
]


# Resolve forward references after all imports
def _resolve_all_forward_references():
    """Resolve all forward references for circular imports"""
    try:
        from .batch import DocumentResult, BatchDocument
        from .evaluation import DisclosureEvaluationResult

        # Rebuild models that have forward references
        DocumentResult.model_rebuild()
        BatchDocument.model_rebuild()
    except Exception:
        # This will be resolved when modules are imported
        pass


# Call the resolution function
_resolve_all_forward_references()
