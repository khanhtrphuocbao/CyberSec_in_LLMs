"""Pure data access helpers for the benchmark dashboard."""

from __future__ import annotations

import json
import re
from pathlib import Path
from statistics import fmean
from typing import Any, Callable, Dict, Iterable, List, Optional


VARIANTS = ("V0", "V1", "V2", "V3", "V4")


def parse_answer_options(raw_options: str) -> Dict[str, str]:
    """Turn pasted answer lines into the A/B/C… option mapping used by MedQA."""
    parsed: Dict[str, str] = {}
    for line in raw_options.splitlines():
        text = line.strip()
        if not text:
            continue
        matched = re.match(r"^(?:\(?([A-Za-z])\)?[.:)])\s*(.+)$", text)
        provided_label = matched.group(1).upper() if matched else None
        option_text = matched.group(2).strip() if matched else text
        label = provided_label or chr(ord("A") + len(parsed))
        if label in parsed:
            label = chr(ord("A") + len(parsed))
        parsed[label] = option_text
    return parsed


def load_variant_results(results_root: Path) -> Dict[str, List[Dict[str, Any]]]:
    """Load available V0–V4 JSON artifacts without failing on missing variants."""
    loaded: Dict[str, List[Dict[str, Any]]] = {}
    for variant in VARIANTS:
        path = Path(results_root) / variant / f"results_{variant}.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            loaded[variant] = payload
    return loaded


def variant_summary(rows: Iterable[Dict[str, Any]]) -> Dict[str, float | int]:
    """Calculate display-safe metrics from individual result rows."""
    results = list(rows)
    total = len(results)
    valid = sum(result.get("is_valid") is not False for result in results)
    correct = sum(bool(result.get("is_correct")) for result in results)
    confidences = [float(result["confidence"]) for result in results if result.get("confidence") is not None]
    latencies = [float(result["latency_seconds"]) for result in results if result.get("latency_seconds") is not None]
    return {
        "total": total,
        "valid": valid,
        "correct": correct,
        "invalid": total - valid,
        "accuracy": correct / total if total else 0.0,
        "valid_rate": valid / total if total else 0.0,
        "average_confidence": fmean(confidences) if confidences else 0.0,
        "average_latency_seconds": fmean(latencies) if latencies else 0.0,
    }


def select_question_result(
    variant: str,
    question_id: str,
    results_root: Path,
    single_root: Path,
) -> Optional[Dict[str, Any]]:
    """Prefer a fresh one-question artifact, then fall back to benchmark output."""
    variant = variant.upper()
    single_path = Path(single_root) / variant / f"{question_id}.json"
    if single_path.exists():
        payload = json.loads(single_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return payload

    for result in load_variant_results(results_root).get(variant, []):
        if result.get("question_id") == question_id:
            return result
    return None


def answer_comparison_rows(
    *,
    question_id: str,
    correct_answer: str,
    variants: Iterable[str],
    results_root: Path,
    single_root: Path,
    result_lookup: Callable[[str, str, Path, Path], Optional[Dict[str, Any]]] = select_question_result,
) -> List[Dict[str, Any]]:
    """Build display rows comparing each saved prediction to the test-set key."""
    rows: List[Dict[str, Any]] = []
    for variant in variants:
        result = result_lookup(variant, question_id, results_root, single_root)
        if result is None:
            continue
        predicted = result.get("predicted_answer")
        rows.append({
            "Variant": variant.upper(),
            "Predicted answer": predicted,
            "Corrected answer": correct_answer,
            "Correct?": predicted == correct_answer,
            "Confidence": result.get("confidence"),
            "Latency (s)": result.get("latency_seconds"),
        })
    return rows
