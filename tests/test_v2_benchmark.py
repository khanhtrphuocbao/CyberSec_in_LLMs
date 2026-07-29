import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

from medqa_rag.evaluation import v2_benchmark
from medqa_rag.evaluation.v2_benchmark import RAGContextCache, V2BenchmarkRunner


class RAGContextCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.cache_path = Path(self.temp_dir.name) / "rag_cache.json"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_returns_persisted_context_for_a_matching_question_and_config(self):
        key = RAGContextCache.make_key("Question", {"A": "one", "B": "two"}, {"top_k": 5, "two_step": False})
        RAGContextCache(self.cache_path).put("q1", key, "cached guidelines")
        self.assertEqual(RAGContextCache(self.cache_path).get("q1", key), "cached guidelines")

    def test_rejects_context_when_options_or_retrieval_config_change(self):
        original = RAGContextCache.make_key("Question", {"A": "one"}, {"top_k": 5, "two_step": False})
        changed_options = RAGContextCache.make_key("Question", {"A": "different"}, {"top_k": 5, "two_step": False})
        changed_config = RAGContextCache.make_key("Question", {"A": "one"}, {"top_k": 3, "two_step": False})
        cache = RAGContextCache(self.cache_path)
        cache.put("q1", original, "cached guidelines")
        self.assertIsNone(cache.get("q1", changed_options))
        self.assertIsNone(cache.get("q1", changed_config))


class FakeSolveResult:
    def __init__(self, question_id): self.question_id = question_id
    def to_dict(self): return {"question_id": self.question_id, "variant": "V2", "is_valid": True, "is_correct": True, "error": None}


class FakeRAG:
    def __init__(self): self.batch_requests = []
    def get_relevant_context_batch(self, requests, top_k):
        self.batch_requests.append((requests, top_k))
        return [f"context for {question}" for question, _ in requests]


class FakeSystem:
    def __init__(self, barrier=None):
        self.rag, self.barrier, self.solve_thread_id = FakeRAG(), barrier, None
    def solve(self, *, question, options, correct_answer, question_id, variant, guidelines, **kwargs):
        self.solve_thread_id = threading.get_ident()
        if self.barrier: self.barrier.wait(timeout=5)
        return FakeSolveResult(question_id)


class RecordingSystemFactory:
    def __init__(self, barrier=None): self.barrier, self.instances = barrier, []
    def __call__(self):
        system = FakeSystem(self.barrier)
        self.instances.append(system)
        return system


class V2BenchmarkRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.temp_dir.name) / "results_V2"
        self.questions = [
            SimpleNamespace(question_id="q1", question="Question 1", options={"A": "A1", "B": "B1"}, answer="A"),
            SimpleNamespace(question_id="q2", question="Question 2", options={"A": "A2", "B": "B2"}, answer="B"),
            SimpleNamespace(question_id="q3", question="Question 3", options={"A": "A3", "B": "B3"}, answer="A"),
        ]
    def tearDown(self): self.temp_dir.cleanup()

    def test_prefetch_retrieves_only_cache_misses_in_one_standard_batch(self):
        factory = RecordingSystemFactory()
        runner = V2BenchmarkRunner(self.questions[:2], self.output_dir, workers=1, system_factory=factory)
        runner.cache.put("q1", runner.cache_key(self.questions[0]), "already cached")
        contexts = runner.prefetch_contexts(self.questions[:2])
        self.assertEqual(contexts["q1"], "already cached")
        self.assertEqual(contexts["q2"], "context for Question 2")
        self.assertEqual(factory.instances[0].rag.batch_requests, [([("Question 2", ["A2", "B2"])], 5)])

    def test_workers_use_distinct_system_instances_and_main_thread_writes_results(self):
        factory = RecordingSystemFactory(threading.Barrier(3))
        results = V2BenchmarkRunner(self.questions, self.output_dir, workers=3, system_factory=factory).run()
        worker_systems = [system for system in factory.instances if system.solve_thread_id is not None]
        self.assertEqual({row["question_id"] for row in results}, {"q1", "q2", "q3"})
        self.assertEqual(len({id(system) for system in worker_systems}), 3)
        self.assertTrue((self.output_dir / "results_V2.json").exists())

    def test_result_includes_its_share_of_precomputed_retrieval_latency(self):
        runner = V2BenchmarkRunner(self.questions[:1], self.output_dir, workers=1, system_factory=RecordingSystemFactory())
        runner._prefetch_telemetry = {"q1": {"retrieval_seconds": 1.25, "context_source": "precomputed"}}
        enriched = runner._attach_prefetch_telemetry({"latency_seconds": 10.0, "metadata": {"rag_trace": {}, "latency_breakdown_seconds": {"retrieval": 0.0, "total": 10.0}}}, "q1")
        self.assertEqual(enriched["latency_seconds"], 11.25)
        self.assertEqual(enriched["metadata"]["latency_breakdown_seconds"]["retrieval"], 1.25)


class V2CliTests(unittest.TestCase):
    def test_parser_defaults_to_three_workers_and_the_v2_cache_file(self):
        args = v2_benchmark.build_v2_parser().parse_args([])
        self.assertEqual(args.workers, 3)
        self.assertEqual(args.cache_file, "results_V2/rag_cache.json")
