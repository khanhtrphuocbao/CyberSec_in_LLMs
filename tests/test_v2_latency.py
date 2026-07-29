import unittest
from types import SimpleNamespace
from unittest import mock

from medqa_rag.core.system import MedQASystem


class FakePlanner:
    total_tokens, prompt_tokens, completion_tokens, total_latency = 101, 61, 40, 91.0
    def create_plan(self, question, options, guidelines): return []


class FakeExaminer:
    total_tokens, prompt_tokens, completion_tokens, total_latency = 202, 122, 80, 92.0
    def clear_memory(self): pass
    def examine(self, question, options, guidelines, plan, use_memory):
        return {"reasoning_steps": [], "option_analysis": {}, "final_answer": "A", "confidence": 0.8}
    def get_trace(self): return "examiner trace"


class FakeEvaluator:
    total_tokens, prompt_tokens, completion_tokens, total_latency = 303, 183, 120, 93.0
    def evaluate(self, question, options, guidelines, result):
        return SimpleNamespace(status=SimpleNamespace(value="Continue"), confidence=0.9, feedback="verified")


class V2LatencyTests(unittest.TestCase):
    def _system(self):
        system = MedQASystem.__new__(MedQASystem)
        system._planner, system._examiner, system._evaluator = FakePlanner(), FakeExaminer(), FakeEvaluator()
        return system

    def test_records_only_this_questions_retrieval_and_agent_stage_durations(self):
        ticks = iter([0.0, 0.0, 1.0, 1.0, 3.0, 3.0, 6.0, 6.0, 10.0, 10.0])
        with mock.patch("medqa_rag.core.system.time.perf_counter", side_effect=lambda: next(ticks)):
            result = self._system()._solve_v2("Question", {"A": "answer"}, "A", "q1", "cached context", 5)
        self.assertEqual(result.metadata["latency_breakdown_seconds"], {"retrieval": 1.0, "planner": 2.0, "examiner": 3.0, "evaluator": 4.0, "total": 10.0})
        self.assertEqual(result.latency_seconds, 10.0)

    def test_cached_two_step_context_does_not_make_a_second_keyword_call(self):
        result = self._system()._solve_v2("Question", {"A": "answer"}, "A", "q1", "cached two-step context", 5, use_two_step=True)
        self.assertTrue(result.metadata["rag_trace"]["guidelines_supplied"])
        self.assertEqual(result.metadata["rag_trace"]["keywords"], [])
