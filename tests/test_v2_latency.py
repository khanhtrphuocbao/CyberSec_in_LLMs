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

    def test_usage_breakdown_contains_only_this_questions_agent_deltas(self):
        class CountingPlanner(FakePlanner):
            def create_plan(self, question, options, guidelines):
                self.total_tokens += 11
                self.prompt_tokens += 7
                self.completion_tokens += 4
                self.total_latency += 1.1
                return []

        class CountingExaminer(FakeExaminer):
            def examine(self, question, options, guidelines, plan, use_memory):
                self.total_tokens += 22
                self.prompt_tokens += 14
                self.completion_tokens += 8
                self.total_latency += 2.2
                return super().examine(question, options, guidelines, plan, use_memory)

        class CountingEvaluator(FakeEvaluator):
            def evaluate(self, question, options, guidelines, result):
                self.total_tokens += 33
                self.prompt_tokens += 21
                self.completion_tokens += 12
                self.total_latency += 3.3
                return super().evaluate(question, options, guidelines, result)

        system = MedQASystem.__new__(MedQASystem)
        system._planner, system._examiner, system._evaluator = CountingPlanner(), CountingExaminer(), CountingEvaluator()
        result = system._solve_v2("Question", {"A": "answer"}, "A", "q1", "cached context", 5)

        self.assertEqual(result.total_tokens, 66)
        self.assertEqual(result.prompt_tokens, 42)
        self.assertEqual(result.completion_tokens, 24)
        self.assertAlmostEqual(result.metadata["usage_breakdown"]["planner"]["latency"], 1.1)
        self.assertAlmostEqual(result.metadata["usage_breakdown"]["examiner"]["latency"], 2.2)
        self.assertAlmostEqual(result.metadata["usage_breakdown"]["evaluator"]["latency"], 3.3)
