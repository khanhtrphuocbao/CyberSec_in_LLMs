import unittest
from types import SimpleNamespace

from medqa_rag.core.system import MedQASystem


class FakePlanner:
    total_tokens = prompt_tokens = completion_tokens = 0
    total_latency = 0.0
    def create_plan(self, *args): return []


class FakeExaminer:
    total_tokens = prompt_tokens = completion_tokens = 0
    total_latency = 0.0
    def __init__(self): self.cleared, self.calls = 0, []
    def clear_memory(self): self.cleared += 1
    def examine(self, question, options, guidelines, plan, **kwargs):
        self.calls.append(kwargs)
        return {"reasoning_steps": [], "option_analysis": {}, "final_answer": "A", "confidence": 0.8}
    def get_trace(self): return "trace"


class FakeEvaluator:
    total_tokens = prompt_tokens = completion_tokens = 0
    total_latency = 0.0
    def __init__(self): self.cleared, self.evaluation_history = 0, []
    def clear_history(self): self.cleared += 1; self.evaluation_history = []
    def verify_with_iteration(self, question, options, guidelines, examine_fn, max_cycles):
        examine_fn(question, options, guidelines)
        revised = examine_fn(question, options, guidelines, feedback="correct the diagnosis", corrections=["use guideline"], prev_result={"final_answer": "B"})
        self.evaluation_history = [SimpleNamespace(status=SimpleNamespace(value="Complete"), confidence=0.9, feedback="ok", to_dict=lambda: {})]
        return revised


class V3FlowTests(unittest.TestCase):
    def test_resets_question_state_and_forwards_revision_feedback(self):
        system = MedQASystem.__new__(MedQASystem)
        system._planner, system._examiner, system._evaluator = FakePlanner(), FakeExaminer(), FakeEvaluator()
        system._solve_v3("Q", {"A": "a"}, "A", "q1", "cached", 5)
        self.assertEqual(system.examiner.cleared, 1)
        self.assertEqual(system.evaluator.cleared, 1)
        self.assertEqual(system.examiner.calls[1]["feedback"], "correct the diagnosis")
        self.assertEqual(system.examiner.calls[1]["corrections"], ["use guideline"])
        self.assertEqual(system.examiner.calls[1]["prev_result"], {"final_answer": "B"})

    def test_uses_examiner_highest_confidence_option_when_verifier_returns_null_answer(self):
        class NullAnswerExaminer(FakeExaminer):
            def examine(self, question, options, guidelines, plan, **kwargs):
                return {
                    "reasoning_steps": [],
                    "option_analysis": {
                        "A": {"confidence": 0.20},
                        "B": {"confidence": 0.91},
                        "C": {"confidence": 0.50},
                    },
                    "final_answer": None,
                    "confidence": 0.91,
                }

        class CompleteEvaluator(FakeEvaluator):
            def verify_with_iteration(self, question, options, guidelines, examine_fn, max_cycles):
                result = examine_fn(question, options, guidelines)
                self.evaluation_history = [SimpleNamespace(status=SimpleNamespace(value="Complete"), confidence=0.9, feedback="ok", to_dict=lambda: {})]
                return result

        system = MedQASystem.__new__(MedQASystem)
        system._planner, system._examiner, system._evaluator = FakePlanner(), NullAnswerExaminer(), CompleteEvaluator()
        result = system._solve_v3("Q", {"A": "a", "B": "b", "C": "c"}, "B", "q1", "cached", 5)

        self.assertEqual(result.predicted_answer, "B")
        self.assertTrue(result.is_valid)
        self.assertTrue(result.metadata["fallback_answer_used"])

    def test_v3_usage_breakdown_contains_only_this_questions_agent_deltas(self):
        class CountingPlanner(FakePlanner):
            total_tokens, prompt_tokens, completion_tokens, total_latency = 100, 60, 40, 10.0
            def create_plan(self, *args):
                self.total_tokens += 11
                self.prompt_tokens += 7
                self.completion_tokens += 4
                self.total_latency += 1.1
                return []

        class CountingExaminer(FakeExaminer):
            total_tokens, prompt_tokens, completion_tokens, total_latency = 200, 120, 80, 20.0
            def examine(self, question, options, guidelines, plan, **kwargs):
                self.total_tokens += 22
                self.prompt_tokens += 14
                self.completion_tokens += 8
                self.total_latency += 2.2
                return {"reasoning_steps": [], "option_analysis": {}, "final_answer": "A", "confidence": 0.8}

        class CountingEvaluator(FakeEvaluator):
            total_tokens, prompt_tokens, completion_tokens, total_latency = 300, 180, 120, 30.0
            def verify_with_iteration(self, question, options, guidelines, examine_fn, max_cycles):
                result = examine_fn(question, options, guidelines)
                self.total_tokens += 33
                self.prompt_tokens += 21
                self.completion_tokens += 12
                self.total_latency += 3.3
                self.evaluation_history = [SimpleNamespace(status=SimpleNamespace(value="Complete"), confidence=0.9, feedback="ok", to_dict=lambda: {})]
                return result

        system = MedQASystem.__new__(MedQASystem)
        system._planner, system._examiner, system._evaluator = CountingPlanner(), CountingExaminer(), CountingEvaluator()
        result = system._solve_v3("Q", {"A": "a"}, "A", "q1", "cached", 5)

        self.assertEqual(result.total_tokens, 66)
        self.assertEqual(result.prompt_tokens, 42)
        self.assertEqual(result.completion_tokens, 24)
        self.assertAlmostEqual(result.metadata["usage_breakdown"]["planner"]["latency"], 1.1)
        self.assertAlmostEqual(result.metadata["usage_breakdown"]["examiner"]["latency"], 2.2)
        self.assertAlmostEqual(result.metadata["usage_breakdown"]["evaluator"]["latency"], 3.3)
