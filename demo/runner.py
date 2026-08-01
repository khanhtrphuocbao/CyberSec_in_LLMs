"""Adapter that runs the existing single-question variant CLI wrappers."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional

from .data import VARIANTS


def build_variant_command(
    python: str,
    repository_root: Path,
    variant: str,
    *,
    question_index: int,
    top_k: int,
    two_step_retrieval: bool,
) -> list[str]:
    """Build the exact command for an existing `run_v*.py` wrapper."""
    normalized_variant = variant.upper()
    if normalized_variant not in VARIANTS:
        raise ValueError(f"Unknown variant: {variant}")
    if question_index < 0:
        raise ValueError("question_index must be non-negative")
    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    command = [
        python,
        f"run_{normalized_variant.lower()}.py",
        "--question-index",
        str(question_index),
        "--top-k",
        str(top_k),
    ]
    if two_step_retrieval:
        command.append("--two-step-retrieval")
    return command


def run_variant(
    python: str,
    repository_root: Path,
    variant: str,
    *,
    question_index: int,
    top_k: int,
    two_step_retrieval: bool,
    environment: Optional[Mapping[str, str]] = None,
) -> subprocess.CompletedProcess[str]:
    """Run one variant sequentially, preserving the configured `.env` behaviour."""
    command = build_variant_command(
        python,
        repository_root,
        variant,
        question_index=question_index,
        top_k=top_k,
        two_step_retrieval=two_step_retrieval,
    )
    child_environment = os.environ.copy()
    child_environment["HF_HUB_OFFLINE"] = "1"
    if environment:
        child_environment.update(environment)
    return subprocess.run(
        command,
        cwd=Path(repository_root),
        env=child_environment,
        capture_output=True,
        text=True,
        check=False,
    )


def run_custom_variant(
    variant: str,
    *,
    question: str,
    options: Dict[str, str],
    top_k: int,
    two_step_retrieval: bool,
    environment: Optional[Mapping[str, str]] = None,
    config_loader: Optional[Callable[[], None]] = None,
    system_factory: Optional[Callable[..., Any]] = None,
) -> Dict[str, Any]:
    """Run one existing variant for a user-provided question in the UI session."""
    normalized_variant = variant.upper()
    if normalized_variant not in VARIANTS:
        raise ValueError(f"Unknown variant: {variant}")
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    if not question.strip():
        raise ValueError("Question must not be empty")
    if len(options) < 2:
        raise ValueError("At least two answer options are required")

    if config_loader is None:
        package_parent = Path(__file__).resolve().parents[2]
        if str(package_parent) not in sys.path:
            sys.path.insert(0, str(package_parent))
        from medqa_rag.config import load_config
        config_loader = load_config
    config_loader()

    source_environment = environment if environment is not None else os.environ
    api_key = source_environment.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is required to run a custom question")

    if system_factory is None:
        from medqa_rag.core.system import MedQASystem
        system_factory = MedQASystem
    system = system_factory(
        api_key=api_key,
        use_two_step_retrieval=two_step_retrieval,
    )
    result = system.solve(
        question=question.strip(),
        options=options,
        correct_answer="",
        question_id="custom",
        variant=normalized_variant,
        top_k=top_k,
        use_two_step_retrieval=two_step_retrieval,
    )
    return result.to_dict()
