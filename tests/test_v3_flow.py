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
