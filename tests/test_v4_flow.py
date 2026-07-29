import unittest
from unittest import mock

from medqa_rag.core.system import MedQASystem


class FakePlanner:
    total_tokens, prompt_tokens, completion_tokens = 101, 61, 40
    total_latency = 91.0

    def create_plan(self, *args):
        self.total_tokens += 10
        self.prompt_tokens += 6
        self.completion_tokens += 4
        return []


class FakeExaminer:
    total_tokens, prompt_tokens, completion_tokens = 202, 122, 80
    total_latency = 92.0

    def __init__(self):
        self.cleared = 0

    def clear_memory(self):
        self.cleared += 1

    def examine(self, *args, **kwargs):
        self.total_tokens += 20
        self.prompt_tokens += 12
        self.completion_tokens += 8
        return {"reasoning_steps": [], "option_analysis": {}, "final_answer": "A", "confidence": 0.8}

    def get_trace(self):
        return "trace"


class FakeRAG:
    def __init__(self):
        self.keyword_calls = 0

    def extract_keywords(self, *args, **kwargs):
        self.keyword_calls += 1
        return ["should not be requested"]


class V4FlowTests(unittest.TestCase):
    def _system(self):
        system = MedQASystem.__new__(MedQASystem)
        system._planner = FakePlanner()
        system._examiner = FakeExaminer()
        system._rag = FakeRAG()
        return system

    def test_resets_memory_and_does_not_repeat_keyword_extraction_for_cached_two_step_context(self):
        system = self._system()

        result = system._solve_v4(
            "Question", {"A": "answer"}, "A", "q1", "cached context", 5, use_two_step=True
        )

        self.assertEqual(system.examiner.cleared, 1)
        self.assertEqual(system.rag.keyword_calls, 0)
        self.assertEqual(result.metadata["rag_trace"]["keywords"], [])

    def test_records_per_question_latency_not_cumulative_agent_totals(self):
        system = self._system()
        ticks = iter([0.0, 0.0, 1.0, 1.0, 3.0, 3.0, 6.0, 6.0])

        with mock.patch("medqa_rag.core.system.time.perf_counter", side_effect=lambda: next(ticks)):
            result = system._solve_v4("Question", {"A": "answer"}, "A", "q1", "cached", 5)

        self.assertEqual(result.latency_seconds, 6.0)
        self.assertEqual((result.total_tokens, result.prompt_tokens, result.completion_tokens), (30, 18, 12))
        self.assertEqual(result.metadata["latency_breakdown_seconds"], {
            "retrieval": 1.0,
            "planner": 2.0,
            "examiner": 3.0,
            "evaluator": 0.0,
            "total": 6.0,
        })
