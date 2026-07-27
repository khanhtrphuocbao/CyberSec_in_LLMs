"""
MedQA-USMLE Evaluation Script
===========================

Evaluates all 1273 test questions across 5 variants and calculates:
- Accuracy per variant
- Invalid response rate
- Accuracy gain (V3 - V0)
- Error analysis with ≥20 extracted cases

Usage:
    python evaluate.py --data ./medqa_data/test.json --output ./results/
"""

import os
import json
import argparse
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime
from tqdm import tqdm

# Import the MedQA system
from ..core.system import MedQASystem, SolveResult, Variant
from ..rag.data_loader import MedQALoader, MedQAExporter, MedQAQuestion


@dataclass
class EvaluationMetrics:
    """Metrics for a single variant."""
    variant: str
    total: int
    correct: int
    invalid: int
    accuracy: float
    invalid_rate: float
    avg_confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvaluationReport:
    """Complete evaluation report."""
    timestamp: str
    total_questions: int
    variants_tested: List[str]
    metrics: Dict[str, EvaluationMetrics]
    accuracy_gain: Optional[float]  # V3 - V0
    error_analysis: List[Dict[str, Any]]
    total_time_seconds: float
    questions_per_variant: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "total_questions": self.total_questions,
            "variants_tested": self.variants_tested,
            "metrics": {k: v.to_dict() for k, v in self.metrics.items()},
            "accuracy_gain": self.accuracy_gain,
            "error_analysis": self.error_analysis,
            "total_time_seconds": self.total_time_seconds,
            "questions_per_variant": self.questions_per_variant
        }

    def summary(self) -> str:
        """Generate a human-readable summary."""
        lines = [
            "=" * 60,
            "MedQA-USMLE EVALUATION REPORT",
            "=" * 60,
            f"Timestamp: {self.timestamp}",
            f"Total Questions: {self.total_questions}",
            f"Questions per Variant: {self.questions_per_variant}",
            f"Time: {self.total_time_seconds:.1f}s",
            "",
            "ACCURACY BY VARIANT:",
            "-" * 40
        ]

        for variant, metrics in sorted(self.metrics.items()):
            gain = ""
            if variant == "V3" and "V0" in self.metrics:
                gain = f" (Δ={self.accuracy_gain:+.1%})" if self.accuracy_gain else ""
            lines.append(
                f"  {variant}: {metrics.accuracy:.1%} "
                f"(valid: {100-metrics.invalid_rate:.1f}%){gain}"
            )

        lines.extend([
            "",
            "ERROR ANALYSIS:",
            "-" * 40,
            f"  Total errors: {sum(1 for e in self.error_analysis)}",
            f"  Extracted cases: {len(self.error_analysis)}"
        ])

        return "\n".join(lines)


class MedQAEvaluator:
    """
    Evaluates the MedQA system across all variants.

    Features:
    - Run all 5 variants on the same questions
    - Calculate accuracy, invalid rate, confidence
    - Extract error cases for analysis
    - Generate comprehensive reports
    """

    def __init__(
        self,
        api_key: str,
        data_path: str,
        output_dir: str = "./results",
        model: str = "gpt-4o",
        max_questions: Optional[int] = None,
        variants: Optional[List[str]] = None
    ):
        """
        Initialize evaluator.

        Args:
            api_key: OpenAI API key
            data_path: Path to MedQA test data
            output_dir: Directory for results
            model: LLM model to use
            max_questions: Limit number of questions (for testing)
            variants: List of variants to test (default: all 5)
        """
        self.api_key = api_key
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Default to all variants
        self.variants = variants or ["V0", "V1", "V2", "V3", "V4"]

        # Load questions
        self.questions = self._load_questions(max_questions)

        # Initialize system
        self.system = MedQASystem(api_key, model=model)

        # Results storage
        self.results: Dict[str, List[SolveResult]] = {v: [] for v in self.variants}

    def _load_questions(self, max_questions: Optional[int]) -> List[MedQAQuestion]:
        """Load questions from data file."""
        loader = MedQALoader()
        questions = loader.load_json(self.data_path)

        if max_questions:
            questions = questions[:max_questions]

        print(f"[Evaluator] Loaded {len(questions)} questions")
        return questions

    def evaluate(self, use_tqdm: bool = True) -> EvaluationReport:
        """
        Run evaluation across all variants.

        Args:
            use_tqdm: Show progress bar

        Returns:
            EvaluationReport with all metrics
        """
        start_time = time.time()

        # For fair comparison, use the same questions and guidelines for all variants
        # Pre-retrieve guidelines for each question
        print("[Evaluator] Pre-retrieving RAG context for all questions...")
        question_contexts = {}

        for q in tqdm(self.questions, desc="Retrieving guidelines"):
            context = self.system.rag.get_relevant_context(
                q.question, q.options, top_k=5
            )
            question_contexts[q.question_id] = context

        # Run each variant
        for variant in self.variants:
            print(f"\n[Evaluator] Running variant {variant}...")
            variant_results = []

            iterator = tqdm(self.questions, desc=f"{variant}") if use_tqdm else self.questions

            for q in iterator:
                result = self.system.solve(
                    question=q.question,
                    options=q.options,
                    correct_answer=q.answer,
                    question_id=q.question_id,
                    variant=variant,
                    guidelines=question_contexts.get(q.question_id)
                )
                variant_results.append(result)

            self.results[variant] = variant_results

        total_time = time.time() - start_time

        # Calculate metrics
        metrics = self._calculate_metrics()

        # Extract error cases
        error_cases = self._extract_errors(min_cases=20)

        # Calculate accuracy gain
        accuracy_gain = None
        if "V3" in metrics and "V0" in metrics:
            accuracy_gain = metrics["V3"].accuracy - metrics["V0"].accuracy

        # Build report
        report = EvaluationReport(
            timestamp=datetime.now().isoformat(),
            total_questions=len(self.questions),
            variants_tested=self.variants,
            metrics=metrics,
            accuracy_gain=accuracy_gain,
            error_analysis=error_cases,
            total_time_seconds=total_time,
            questions_per_variant=len(self.questions)
        )

        return report

    def _calculate_metrics(self) -> Dict[str, EvaluationMetrics]:
        """Calculate metrics for each variant."""
        metrics = {}

        for variant, results in self.results.items():
            total = len(results)
            correct = sum(1 for r in results if r.is_correct)
            invalid = sum(1 for r in results if not r.is_valid)
            avg_conf = sum(r.confidence for r in results) / total if total else 0

            metrics[variant] = EvaluationMetrics(
                variant=variant,
                total=total,
                correct=correct,
                invalid=invalid,
                accuracy=correct / total if total else 0,
                invalid_rate=invalid / total if total else 0,
                avg_confidence=avg_conf
            )

        return metrics

    def _extract_errors(self, min_cases: int = 20) -> List[Dict[str, Any]]:
        """Extract error cases for analysis."""
        errors = []

        # Use V3 results for error analysis (our best system)
        v3_results = self.results.get("V3", [])

        for result in v3_results:
            if not result.is_correct:
                errors.append({
                    "question_id": result.question_id,
                    "question": self._get_question_text(result.question_id),
                    "options": self._get_question_options(result.question_id),
                    "predicted_answer": result.predicted_answer,
                    "correct_answer": result.correct_answer,
                    "confidence": result.confidence,
                    "reasoning": result.reasoning[:1000],  # Truncate
                    "metadata": result.metadata,
                    "variant": result.variant
                })

        # Sort by confidence (most confident wrong answers first - interesting cases)
        errors.sort(key=lambda x: x["confidence"], reverse=True)

        return errors[:max(min_cases, len(errors))]

    def _get_question_text(self, question_id: str) -> str:
        """Get question text by ID."""
        for q in self.questions:
            if q.question_id == question_id:
                return q.question
        return ""

    def _get_question_options(self, question_id: str) -> Dict[str, str]:
        """Get options by question ID."""
        for q in self.questions:
            if q.question_id == question_id:
                return q.options
        return {}

    def save_results(self, report: EvaluationReport) -> None:
        """Save all results to files."""
        # Save main report
        report_path = self.output_dir / "evaluation_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
        print(f"[Evaluator] Saved report to {report_path}")

        # Save per-question results for each variant
        for variant, results in self.results.items():
            variant_path = self.output_dir / f"results_{variant}.json"
            with open(variant_path, "w", encoding="utf-8") as f:
                json.dump([r.to_dict() for r in results], f, indent=2, ensure_ascii=False)
            print(f"[Evaluator] Saved {variant} results to {variant_path}")

        # Save error cases
        error_path = self.output_dir / "error_cases.json"
        with open(error_path, "w", encoding="utf-8") as f:
            json.dump(report.error_analysis, f, indent=2, ensure_ascii=False)
        print(f"[Evaluator] Saved {len(report.error_analysis)} error cases to {error_path}")

        # Save CSV for easy analysis
        csv_path = self.output_dir / "results_summary.csv"
        self._save_csv(csv_path)
        print(f"[Evaluator] Saved CSV summary to {csv_path}")

    def _save_csv(self, csv_path: Path) -> None:
        """Save a summary CSV with all results."""
        import csv

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            # Header
            writer.writerow([
                "question_id", "correct_answer",
                "V0_pred", "V0_correct", "V0_conf",
                "V1_pred", "V1_correct", "V1_conf",
                "V2_pred", "V2_correct", "V2_conf",
                "V3_pred", "V3_correct", "V3_conf",
                "V4_pred", "V4_correct", "V4_conf"
            ])

            # Get result by question_id for each variant
            results_by_qid = {v: {r.question_id: r for r in rs} for v, rs in self.results.items()}

            for q in self.questions:
                row = [q.question_id, q.answer]

                for variant in ["V0", "V1", "V2", "V3", "V4"]:
                    if variant in results_by_qid and q.question_id in results_by_qid[variant]:
                        r = results_by_qid[variant][q.question_id]
                        row.extend([r.predicted_answer, r.is_correct, r.confidence])
                    else:
                        row.extend(["N/A", "N/A", "N/A"])

                writer.writerow(row)


def run_evaluation(
    data_path: str,
    api_key: str = None,
    output_dir: str = "./results",
    max_questions: Optional[int] = None,
    variants: Optional[List[str]] = None,
    save_results: bool = True
) -> EvaluationReport:
    """
    Convenience function to run evaluation.

    Args:
        data_path: Path to MedQA test JSON
        api_key: OpenAI API key
        output_dir: Output directory
        max_questions: Limit questions (for testing)
        variants: Variants to test
        save_results: Whether to save to disk

    Returns:
        EvaluationReport
    """
    if api_key is None:
        api_key = os.environ.get("OPENAI_API_KEY", "")

    if not api_key:
        raise ValueError("OpenAI API key required. Set OPENAI_API_KEY or pass api_key.")

    evaluator = MedQAEvaluator(
        api_key=api_key,
        data_path=data_path,
        output_dir=output_dir,
        max_questions=max_questions,
        variants=variants
    )

    report = evaluator.evaluate()

    if save_results:
        evaluator.save_results(report)

    print("\n" + report.summary())

    return report


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="MedQA-USMLE Evaluation")
    parser.add_argument(
        "--data", "-d",
        required=True,
        help="Path to MedQA test data JSON"
    )
    parser.add_argument(
        "--output", "-o",
        default="./results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--api-key", "-k",
        default=None,
        help="OpenAI API key (or set OPENAI_API_KEY env)"
    )
    parser.add_argument(
        "--max-questions", "-m",
        type=int,
        default=None,
        help="Limit number of questions (for testing)"
    )
    parser.add_argument(
        "--variants", "-v",
        nargs="+",
        default=["V0", "V1", "V2", "V3", "V4"],
        choices=["V0", "V1", "V2", "V3", "V4"],
        help="Which variants to test"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save results to disk"
    )

    args = parser.parse_args()

    report = run_evaluation(
        data_path=args.data,
        api_key=args.api_key,
        output_dir=args.output,
        max_questions=args.max_questions,
        variants=args.variants,
        save_results=not args.no_save
    )

    return report


if __name__ == "__main__":
    main()
