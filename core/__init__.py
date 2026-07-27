"""
Core sub-package.

System-level orchestration:
- system: MedQASystem with 5 ablation variants (V0-V4)
"""

from .system import MedQASystem, SolveResult, Variant

__all__ = [
    "MedQASystem",
    "SolveResult",
    "Variant",
]