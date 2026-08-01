import subprocess
import sys
import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class DemoAppTests(unittest.TestCase):
    def test_demo_dependencies_include_watchdog_for_streamlit_file_watching(self):
        requirements = (REPOSITORY_ROOT / "demo" / "requirements.txt").read_text(encoding="utf-8")

        self.assertIn("watchdog", requirements)

    def test_streamlit_entry_point_compiles(self):
        completed = subprocess.run(
            [sys.executable, "-m", "py_compile", "demo/app.py"],
            cwd=REPOSITORY_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_question_runner_exposes_test_and_custom_question_inputs(self):
        app = AppTest.from_file(str(REPOSITORY_ROOT / "demo" / "app.py")).run(timeout=60)

        self.assertFalse(app.exception)
        self.assertIn("Nguồn câu hỏi", [control.label for control in app.segmented_control])
        app.segmented_control[0].set_value("Câu hỏi tuỳ ý").run(timeout=60)
        self.assertIn("Câu hỏi", [input.label for input in app.text_area])
        self.assertIn("Các đáp án", [input.label for input in app.text_area])

    def test_agent_trace_follows_the_custom_question_source_and_uses_session_result(self):
        app = AppTest.from_file(str(REPOSITORY_ROOT / "demo" / "app.py")).run(timeout=60)
        app.segmented_control[0].set_value("Câu hỏi tuỳ ý").run(timeout=60)
        app.session_state["custom_run_results"] = {
            "V3": {
                "predicted_answer": "CUSTOM_ANSWER",
                "confidence": 0.9,
                "latency_seconds": 1.5,
                "total_tokens": 123,
                "metadata": {"planner_trace": {"steps": []}},
            }
        }
        app.run(timeout=60)

        self.assertIn("CUSTOM_ANSWER", [metric.value for metric in app.metric])
