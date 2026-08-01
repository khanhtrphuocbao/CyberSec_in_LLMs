import json
import os
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from medqa_rag.single_question_cli import run_single_question


class FakeSolveResult:
    def __init__(self, question_id, variant):
        self.question_id = question_id
        self.variant = variant

    def to_dict(self):
        return {
            "question_id": self.question_id,
            "variant": self.variant,
            "predicted_answer": "A",
            "is_valid": True,
        }


class FakeLoader:
    def __init__(self, questions):
        self.questions = questions
        self.loaded_path = None

    def load_json(self, path):
        self.loaded_path = path
        return self.questions


class FakeSystem:
    def __init__(self, api_key):
        self.api_key = api_key
        self.calls = []

    def solve(self, **kwargs):
        self.calls.append(kwargs)
        print(f"solved {kwargs['question_id']}")
        return FakeSolveResult(kwargs["question_id"], kwargs["variant"])


class SingleQuestionCliTests(unittest.TestCase):
    def test_runs_the_requested_zero_based_index_and_persists_result_and_log(self):
        questions = [
            SimpleNamespace(question_id="q0000", question="Question 0", options={"A": "A"}, answer="A"),
            SimpleNamespace(question_id="q0010", question="Question 10", options={"A": "A"}, answer="A"),
        ]
        loader = FakeLoader(questions)
        systems = []

        def system_factory(api_key):
            system = FakeSystem(api_key)
            systems.append(system)
            return system

        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ, {"OPENAI_API_KEY": "test-key"}, clear=False
        ):
            root = Path(directory)
            args = Namespace(
                data_path="test.jsonl",
                question_index=1,
                top_k=3,
                two_step_retrieval=False,
                valid_book_names=None,
                result_root=str(root / "results"),
                log_root=str(root / "logs"),
                api_key_env="OPENAI_API_KEY",
            )

            exit_code = run_single_question(
                args,
                "V3",
                loader=loader,
                system_factory=system_factory,
                config_loader=lambda: None,
            )

            self.assertEqual(exit_code, 0)
            self.assertEqual(loader.loaded_path, "test.jsonl")
            self.assertEqual(systems[0].api_key, "test-key")
            self.assertEqual(systems[0].calls[0]["question_id"], "q0010")
            self.assertEqual(systems[0].calls[0]["variant"], "V3")
            self.assertEqual(systems[0].calls[0]["top_k"], 3)

            result_path = root / "results" / "V3" / "q0010.json"
            log_path = root / "logs" / "V3" / "q0010.log"
            self.assertTrue(result_path.exists())
            self.assertTrue(log_path.exists())
            self.assertEqual(json.loads(result_path.read_text())["question_index"], 1)
            self.assertIn("solved q0010", log_path.read_text())
