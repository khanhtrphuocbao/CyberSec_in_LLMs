import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

from medqa_rag.evaluation.v2_benchmark import V3BenchmarkRunner
from run_v3 import build_v3_parser


class FakeSolveResult:
    def __init__(self, question_id): self.question_id = question_id
    def to_dict(self): return {"question_id": self.question_id, "variant": "V3", "is_valid": True, "is_correct": True, "error": None}


class FakeRAG:
    def __init__(self): self.batch_requests = []
    def get_relevant_context_batch(self, requests, top_k):
        self.batch_requests.append((requests, top_k))
        return [f"context for {question}" for question, _ in requests]


class FakeSystem:
    def __init__(self, barrier):
        self.rag, self.barrier, self.solve_thread_id = FakeRAG(), barrier, None
    def solve(self, *, question, options, correct_answer, question_id, variant, guidelines, **kwargs):
        self.solve_thread_id = threading.get_ident()
        self.barrier.wait(timeout=5)
        return FakeSolveResult(question_id)


class RecordingSystemFactory:
    def __init__(self): self.barrier, self.instances = threading.Barrier(2), []
    def __call__(self):
        system = FakeSystem(self.barrier)
        self.instances.append(system)
        return system


class V3BenchmarkRunnerTests(unittest.TestCase):
    def test_prefetches_rag_once_and_passes_context_to_isolated_v3_workers(self):
        questions = [
            SimpleNamespace(question_id="q1", question="Question 1", options={"A": "A1"}, answer="A"),
            SimpleNamespace(question_id="q2", question="Question 2", options={"A": "A2"}, answer="A"),
        ]
        factory = RecordingSystemFactory()
        with tempfile.TemporaryDirectory() as directory:
            results = V3BenchmarkRunner(questions, Path(directory), workers=2, system_factory=factory).run()
            self.assertEqual(factory.instances[0].rag.batch_requests, [([("Question 1", ["A1"]), ("Question 2", ["A2"])], 5)])
            self.assertTrue(all(system.rag.batch_requests == [] for system in factory.instances[1:]))
            self.assertEqual({result["question_id"] for result in results}, {"q1", "q2"})

    def test_v3_cli_defaults_to_two_workers_and_v3_paths(self):
        args = build_v3_parser().parse_args([])
        self.assertEqual(args.workers, 2)
        self.assertEqual(args.output_dir, "results_V3")
        self.assertEqual(args.cache_file, "results_V3/rag_cache.json")
