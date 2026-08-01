import unittest
from pathlib import Path

from demo.runner import build_variant_command, run_custom_variant


class DemoRunnerTests(unittest.TestCase):
    def test_builds_existing_variant_cli_command_with_retrieval_options(self):
        command = build_variant_command(
            "python",
            Path("/repo"),
            "V3",
            question_index=10,
            top_k=5,
            two_step_retrieval=True,
        )

        self.assertEqual(command, [
            "python", "run_v3.py", "--question-index", "10",
            "--top-k", "5", "--two-step-retrieval",
        ])

    def test_custom_question_is_forwarded_without_a_test_set_answer(self):
        calls = {}

        class FakeResult:
            def to_dict(self):
                return {"predicted_answer": "B", "is_valid": True}

        class FakeSystem:
            def __init__(self, **kwargs):
                calls["init"] = kwargs

            def solve(self, **kwargs):
                calls["solve"] = kwargs
                return FakeResult()

        result = run_custom_variant(
            "V3",
            question="Which finding is most likely?",
            options={"A": "First", "B": "Second"},
            top_k=3,
            two_step_retrieval=True,
            environment={"OPENAI_API_KEY": "test-key"},
            config_loader=lambda: calls.setdefault("config_loaded", True),
            system_factory=FakeSystem,
        )

        self.assertEqual(calls["init"], {"api_key": "test-key", "use_two_step_retrieval": True})
        self.assertEqual(calls["solve"], {
            "question": "Which finding is most likely?",
            "options": {"A": "First", "B": "Second"},
            "correct_answer": "",
            "question_id": "custom",
            "variant": "V3",
            "top_k": 3,
            "use_two_step_retrieval": True,
        })
        self.assertEqual(result["predicted_answer"], "B")
