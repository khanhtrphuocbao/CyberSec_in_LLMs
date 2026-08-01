import json
import tempfile
import unittest
from pathlib import Path

from demo.data import (
    answer_comparison_rows,
    load_variant_results,
    parse_answer_options,
    select_question_result,
    variant_summary,
)


class DemoDataTests(unittest.TestCase):
    def test_parse_answer_options_accepts_pasted_labeled_or_plain_lines(self):
        self.assertEqual(
            parse_answer_options("A. First choice\n(B) Second choice\nThird choice"),
            {"A": "First choice", "B": "Second choice", "C": "Third choice"},
        )

    def test_variant_summary_counts_correct_against_all_rows_and_tracks_validity(self):
        summary = variant_summary([
            {"is_valid": True, "is_correct": True, "confidence": 0.9, "latency_seconds": 2.0},
            {"is_valid": False, "is_correct": False, "confidence": 0.0, "latency_seconds": 4.0},
        ])

        self.assertEqual(summary["total"], 2)
        self.assertEqual(summary["valid"], 1)
        self.assertEqual(summary["correct"], 1)
        self.assertEqual(summary["accuracy"], 0.5)
        self.assertEqual(summary["valid_rate"], 0.5)
        self.assertEqual(summary["average_latency_seconds"], 3.0)

    def test_loads_variant_rows_and_prefers_single_question_artifact_for_trace(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            full_path = root / "results" / "V3"
            single_path = root / "single" / "V3"
            full_path.mkdir(parents=True)
            single_path.mkdir(parents=True)
            (full_path / "results_V3.json").write_text(json.dumps([
                {"question_id": "q0001", "variant": "V3", "predicted_answer": "A"}
            ]))
            (single_path / "q0001.json").write_text(json.dumps({
                "question_id": "q0001", "variant": "V3", "predicted_answer": "B"
            }))

            loaded = load_variant_results(root / "results")
            selected = select_question_result("V3", "q0001", root / "results", root / "single")

            self.assertEqual(loaded["V3"][0]["predicted_answer"], "A")
            self.assertEqual(selected["predicted_answer"], "B")

    def test_answer_comparison_uses_test_set_answer_and_marks_each_prediction(self):
        rows = answer_comparison_rows(
            question_id="q0010",
            correct_answer="C",
            variants=["V0", "V3"],
            results_root=Path("/unused/results"),
            single_root=Path("/unused/single"),
            result_lookup=lambda variant, question_id, _results, _single: {
                "V0": {"predicted_answer": "B", "confidence": 0.40, "latency_seconds": 1.2},
                "V3": {"predicted_answer": "C", "confidence": 0.95, "latency_seconds": 4.8},
            }[variant],
        )

        self.assertEqual(rows, [
            {
                "Variant": "V0", "Predicted answer": "B", "Corrected answer": "C",
                "Correct?": False, "Confidence": 0.40, "Latency (s)": 1.2,
            },
            {
                "Variant": "V3", "Predicted answer": "C", "Corrected answer": "C",
                "Correct?": True, "Confidence": 0.95, "Latency (s)": 4.8,
            },
        ])
