"""
Evaluation sub-package.

Tools for benchmarking the MedQA system:
- runner: MedQAEvaluator for running 5-variant evaluation on the MedQA-USMLE test set
"""

from .runner import (
    MedQAEvaluator,
    EvaluationMetrics,
    EvaluationReport,
    run_evaluation,
)

__all__ = [
    "MedQAEvaluator",
    "EvaluationMetrics",
    "EvaluationReport",
    "run_evaluation",
]