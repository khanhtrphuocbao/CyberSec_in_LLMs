"""
MedQA-USMLE 5-Variant Ablation Study System
==========================================

This module implements the 5 ablation variants for the MedQA-USMLE benchmark:

V0 (Direct LLM): Send the MedQA question directly to the LLM (no RAG, no multi-agent).
V1 (RAG-only): Send the question + RAG context directly to the LLM.
V2 (Multi-agent without memory): Run Planner -> Examiner -> Evaluator, but clear memory at each step.
V3 (Full system): Run the complete workflow: RAG + Planner -> Examiner (with memory) -> Evaluator.
V4 (Full system without verifier): Run V3 but bypass the Evaluator step.

Usage:
    from main import MedQASystem

    system = MedQASystem(api_key="your-key")
    result = system.solve(question, options, variant="V3")
"""

import os
import json
import openai
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum

# Import agents
from ..rag.retriever import MedQA_RAG
from ..agents.planner import MedQA_Planner, ReasoningStep
from ..agents.examiner import MedQA_Examiner
from ..agents.evaluator import MedQA_Evaluator, EvaluationStatus, VerificationResult
from ..rag.data_loader import MedQAQuestion


class Variant(Enum):
    """Ablation study variants."""
    V0_DIRECT = "V0"  # Direct LLM, no RAG, no agents
    V1_RAG_ONLY = "V1"  # RAG + direct LLM
    V2_NO_MEMORY = "V2"  # Multi-agent, clear memory each step
    V3_FULL = "V3"  # Full system with memory
    V4_NO_VERIFIER = "V4"  # V3 without Evaluator


@dataclass
class SolveResult:
    """Result of solving a single question."""
    question_id: str
    variant: str
    predicted_answer: Optional[str]
    correct_answer: str
    is_correct: bool
    is_valid: bool  # Did we get a valid answer format?
    confidence: float
    reasoning: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question_id": self.question_id,
            "variant": self.variant,
            "predicted_answer": self.predicted_answer,
            "correct_answer": self.correct_answer,
            "is_correct": self.is_correct,
            "is_valid": self.is_valid,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "metadata": self.metadata,
            "error": self.error
        }


class MedQASystem:
    """
    Multi-agent MedQA system with 5 ablation variants.

    Supports:
    - V0: Direct LLM baseline
    - V1: RAG + direct LLM
    - V2: Multi-agent without memory persistence
    - V3: Full multi-agent with memory
    - V4: V3 without evaluator verification
    """

    # System prompts for direct answer variants
    V0_SYSTEM_PROMPT = """You are a medical expert answering USMLE-style multiple choice questions.

Answer the question based on your medical knowledge.
Return your answer in this EXACT format:
ANSWER: [A/B/C/D]
CONF: [0.0-1.0]
REASONING: [brief explanation]"""

    V1_SYSTEM_PROMPT = """You are a medical expert answering USMLE-style multiple choice questions.

Use the provided medical guidelines to answer the question.
Return your answer in this EXACT format:
ANSWER: [A/B/C/D]
CONF: [0.0-1.0]
REASONING: [brief explanation based on guidelines]"""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        rag_persist_dir: str = "/Users/mac/Developers/MedQA_RAG/MedQA_ChromaDB_Injected",
        use_existing_rag: bool = True,
        api_base: Optional[str] = None,
        use_two_step_retrieval: bool = False,
        valid_book_names: Optional[List[str]] = None,
        keyword_model: str = "gpt-4o-mini"
    ):
        """
        Initialize the MedQA system.

        Args:
            api_key: OpenAI API key
            model: LLM model to use
            rag_persist_dir: Directory for RAG vector store
            use_existing_rag: Load existing vector store if available
            api_base: Custom API base URL (for OpenAI-compatible APIs)
            use_two_step_retrieval: Enable Two-step Retrieval (MedAgent-Pro style)
            valid_book_names: List of known book names for metadata filtering
            keyword_model: Model for keyword extraction (default gpt-4o-mini)
        """
        self.api_key = api_key
        self.model = model
        self.api_base = api_base
        openai.api_key = api_key
        if api_base:
            openai.api_base = api_base

        # Initialize RAG (lazy - only for variants that need it)
        self._rag: Optional[MedQA_RAG] = None
        self.rag_persist_dir = rag_persist_dir
        self.use_existing_rag = use_existing_rag

        # Two-step Retrieval config
        self.use_two_step_retrieval = use_two_step_retrieval
        self.valid_book_names = valid_book_names
        self.keyword_model = keyword_model

        # Initialize agents (lazy)
        self._planner: Optional[MedQA_Planner] = None
        self._examiner: Optional[MedQA_Examiner] = None
        self._evaluator: Optional[MedQA_Evaluator] = None

    # =========================================================================
    # Lazy Initialization
    # =========================================================================

    @property
    def rag(self) -> MedQA_RAG:
        """Lazy load RAG."""
        if self._rag is None:
            self._rag = MedQA_RAG(
                openai_api_key=self.api_key,
                persist_directory=self.rag_persist_dir,
                api_base=self.api_base
            )
            if self.use_existing_rag:
                self._rag._load_existing_store()
        return self._rag

    @property
    def planner(self) -> MedQA_Planner:
        """Lazy load Planner."""
        if self._planner is None:
            self._planner = MedQA_Planner(self.api_key, self.model, self.api_base)
        return self._planner

    @property
    def examiner(self) -> MedQA_Examiner:
        """Lazy load Examiner."""
        if self._examiner is None:
            self._examiner = MedQA_Examiner(self.api_key, self.model, self.api_base)
        return self._examiner

    @property
    def evaluator(self) -> MedQA_Evaluator:
        """Lazy load Evaluator."""
        if self._evaluator is None:
            self._evaluator = MedQA_Evaluator(self.api_key, self.model, self.api_base)
        return self._evaluator

    # =========================================================================
    # Core Solve Methods
    # =========================================================================

    def solve(
        self,
        question: str,
        options: Dict[str, str],
        correct_answer: str,
        question_id: str = "unknown",
        variant: str = "V3",
        guidelines: Optional[str] = None,
        top_k: int = 5,
        use_two_step_retrieval: Optional[bool] = None,
        valid_book_names: Optional[List[str]] = None
    ) -> SolveResult:
        """
        Solve a MedQA question using the specified variant.

        Args:
            question: The MedQA question text
            options: Dict of options {"A": "...", "B": "...", ...}
            correct_answer: The correct answer key
            question_id: Identifier for logging
            variant: Which variant to use ("V0", "V1", "V2", "V3", "V4")
            guidelines: Pre-retrieved guidelines (optional)
            top_k: Number of RAG results to retrieve

        Returns:
            SolveResult with prediction and metadata
        """
        # Resolve two-step config: per-call override > instance default
        use_two_step = use_two_step_retrieval if use_two_step_retrieval is not None else self.use_two_step_retrieval
        book_names = valid_book_names if valid_book_names is not None else self.valid_book_names

        # Normalize variant
        variant_upper = variant.upper()
        try:
            variant_enum = Variant(variant_upper)
        except ValueError:
            variant_enum = Variant.V3_FULL

        print(f"\n[{question_id}] Solving with {variant_enum.value}..." +
              (" [Two-step Retrieval]" if use_two_step else ""))

        try:
            # Route to appropriate method
            if variant_enum == Variant.V0_DIRECT:
                return self._solve_v0(question, options, correct_answer, question_id)
            elif variant_enum == Variant.V1_RAG_ONLY:
                return self._solve_v1(
                    question, options, correct_answer, question_id,
                    guidelines, top_k, use_two_step, book_names
                )
            elif variant_enum == Variant.V2_NO_MEMORY:
                return self._solve_v2(
                    question, options, correct_answer, question_id,
                    guidelines, top_k, use_two_step, book_names
                )
            elif variant_enum == Variant.V3_FULL:
                return self._solve_v3(
                    question, options, correct_answer, question_id,
                    guidelines, top_k, use_two_step, book_names
                )
            elif variant_enum == Variant.V4_NO_VERIFIER:
                return self._solve_v4(
                    question, options, correct_answer, question_id,
                    guidelines, top_k, use_two_step, book_names
                )
        except Exception as e:
            print(f"[{question_id}] Error in {variant}: {e}")
            return SolveResult(
                question_id=question_id,
                variant=variant,
                predicted_answer=None,
                correct_answer=correct_answer,
                is_correct=False,
                is_valid=False,
                confidence=0.0,
                reasoning="",
                error=str(e)
            )

    # =========================================================================
    # V0: Direct LLM (no RAG, no agents)
    # =========================================================================

    def _solve_v0(
        self,
        question: str,
        options: Dict[str, str],
        correct_answer: str,
        question_id: str
    ) -> SolveResult:
        """V0: Direct LLM without RAG or agents."""
        print(f"[{question_id}] V0: Direct LLM baseline")

        options_text = "\n".join(f"({k}) {v}" for k, v in options.items())

        messages = [
            {"role": "system", "content": self.V0_SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {question}\n\nOptions:\n{options_text}"}
        ]

        response = self._call_llm(messages)
        answer, confidence, reasoning = self._parse_direct_response(response)

        return SolveResult(
            question_id=question_id,
            variant="V0",
            predicted_answer=answer,
            correct_answer=correct_answer,
            is_correct=answer == correct_answer if answer else False,
            is_valid=answer in ["A", "B", "C", "D"] if answer else False,
            confidence=confidence,
            reasoning=reasoning,
            metadata={"method": "direct_llm"}
        )

    # =========================================================================
    # V1: RAG + Direct LLM
    # =========================================================================

    def _solve_v1(
        self,
        question: str,
        options: Dict[str, str],
        correct_answer: str,
        question_id: str,
        guidelines: Optional[str],
        top_k: int,
        use_two_step: bool = False,
        book_names: Optional[List[str]] = None
    ) -> SolveResult:
        """V1: RAG context + direct LLM (no agents)."""
        print(f"[{question_id}] V1: RAG + Direct LLM" +
              (" [Two-step]" if use_two_step else ""))

        # Get guidelines (standard RAG or Two-step)
        guidelines = self._get_guidelines(
            question, options, guidelines, top_k, use_two_step, book_names
        )

        options_text = "\n".join(f"({k}) {v}" for k, v in options.items())

        messages = [
            {"role": "system", "content": self.V1_SYSTEM_PROMPT},
            {"role": "user", "content": f"""Medical Guidelines:
{guidelines}

Question: {question}

Options:
{options_text}"""}
        ]

        response = self._call_llm(messages)
        answer, confidence, reasoning = self._parse_direct_response(response)

        return SolveResult(
            question_id=question_id,
            variant="V1",
            predicted_answer=answer,
            correct_answer=correct_answer,
            is_correct=answer == correct_answer if answer else False,
            is_valid=answer in ["A", "B", "C", "D"] if answer else False,
            confidence=confidence,
            reasoning=reasoning,
            metadata={
                "method": "rag_direct_llm",
                "top_k": top_k,
                "two_step_retrieval": use_two_step
            }
        )

    # =========================================================================
    # V2: Multi-agent without memory
    # =========================================================================

    def _solve_v2(
        self,
        question: str,
        options: Dict[str, str],
        correct_answer: str,
        question_id: str,
        guidelines: Optional[str],
        top_k: int,
        use_two_step: bool = False,
        book_names: Optional[List[str]] = None
    ) -> SolveResult:
        """V2: Multi-agent workflow WITHOUT memory persistence."""
        print(f"[{question_id}] V2: Multi-agent (no memory)" +
              (" [Two-step]" if use_two_step else ""))

        # Get guidelines (standard RAG or Two-step)
        guidelines = self._get_guidelines(
            question, options, guidelines, top_k, use_two_step, book_names
        )

        # Clear examiner memory
        self.examiner.clear_memory()

        # Create plan
        plan = self.planner.create_plan(question, options, guidelines)

        # Examine WITHOUT memory persistence
        # (each step would clear, but we run full examination)
        result = self.examiner.examine(
            question, options, guidelines, plan, use_memory=False
        )

        # Evaluate (with cleared memory)
        verification = self.evaluator.evaluate(
            question, options, guidelines, result
        )

        answer = result.get("final_answer")
        confidence = result.get("confidence", 0.5)

        return SolveResult(
            question_id=question_id,
            variant="V2",
            predicted_answer=answer,
            correct_answer=correct_answer,
            is_correct=answer == correct_answer if answer else False,
            is_valid=answer in ["A", "B", "C", "D"] if answer else False,
            confidence=confidence,
            reasoning=self.examiner.get_trace(),
            metadata={
                "method": "multi_agent_no_memory",
                "plan_steps": len(plan),
                "verification_status": verification.status.value,
                "evaluator_confidence": verification.confidence,
                "two_step_retrieval": use_two_step
            }
        )

    # =========================================================================
    # V3: Full system (RAG + Planner + Examiner + Evaluator with memory)
    # =========================================================================

    def _solve_v3(
        self,
        question: str,
        options: Dict[str, str],
        correct_answer: str,
        question_id: str,
        guidelines: Optional[str],
        top_k: int,
        use_two_step: bool = False,
        book_names: Optional[List[str]] = None
    ) -> SolveResult:
        """V3: Full multi-agent workflow WITH memory persistence."""
        print(f"[{question_id}] V3: Full system (with memory)" +
              (" [Two-step]" if use_two_step else ""))

        # Get guidelines (standard RAG or Two-step)
        guidelines = self._get_guidelines(
            question, options, guidelines, top_k, use_two_step, book_names
        )

        # Create plan
        plan = self.planner.create_plan(question, options, guidelines)

        # Examine WITH memory
        result = self.examiner.examine(
            question, options, guidelines, plan, use_memory=True
        )

        # Evaluate with memory trace
        verification = self.evaluator.evaluate(
            question, options, guidelines, result
        )

        answer = result.get("final_answer")
        confidence = result.get("confidence", 0.5)

        # Adjust confidence based on verification
        if verification.status != EvaluationStatus.COMPLETE:
            confidence *= verification.confidence

        return SolveResult(
            question_id=question_id,
            variant="V3",
            predicted_answer=answer,
            correct_answer=correct_answer,
            is_correct=answer == correct_answer if answer else False,
            is_valid=answer in ["A", "B", "C", "D"] if answer else False,
            confidence=confidence,
            reasoning=self.examiner.get_trace(),
            metadata={
                "method": "full_multi_agent",
                "plan_steps": len(plan),
                "memory_steps": len(self.examiner.get_memory()),
                "verification_status": verification.status.value,
                "evaluator_confidence": verification.confidence,
                "evaluation_feedback": verification.feedback,
                "two_step_retrieval": use_two_step
            }
        )

    # =========================================================================
    # V4: V3 without Evaluator
    # =========================================================================

    def _solve_v4(
        self,
        question: str,
        options: Dict[str, str],
        correct_answer: str,
        question_id: str,
        guidelines: Optional[str],
        top_k: int,
        use_two_step: bool = False,
        book_names: Optional[List[str]] = None
    ) -> SolveResult:
        """V4: V3 but skip the Evaluator verification step."""
        print(f"[{question_id}] V4: Full system (no evaluator)" +
              (" [Two-step]" if use_two_step else ""))

        # Get guidelines (standard RAG or Two-step)
        guidelines = self._get_guidelines(
            question, options, guidelines, top_k, use_two_step, book_names
        )

        # Create plan
        plan = self.planner.create_plan(question, options, guidelines)

        # Examine WITH memory
        result = self.examiner.examine(
            question, options, guidelines, plan, use_memory=True
        )

        # Skip evaluation - take examiner's answer directly
        answer = result.get("final_answer")
        confidence = result.get("confidence", 0.5)

        return SolveResult(
            question_id=question_id,
            variant="V4",
            predicted_answer=answer,
            correct_answer=correct_answer,
            is_correct=answer == correct_answer if answer else False,
            is_valid=answer in ["A", "B", "C", "D"] if answer else False,
            confidence=confidence,
            reasoning=self.examiner.get_trace(),
            metadata={
                "method": "full_no_evaluator",
                "plan_steps": len(plan),
                "memory_steps": len(self.examiner.get_memory()),
                "evaluation_skipped": True,
                "two_step_retrieval": use_two_step
            }
        )

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def _get_guidelines(
        self,
        question: str,
        options: Dict[str, str],
        guidelines: Optional[str],
        top_k: int,
        use_two_step: bool,
        book_names: Optional[List[str]]
    ) -> str:
        """
        Get guidelines via either standard RAG or Two-step Retrieval.
        """
        if guidelines is not None:
            return guidelines

        options_list = list(options.values()) if options else None

        if use_two_step:
            print(f"[{question}] Using Two-step Retrieval (keyword filtering)")
            return self.rag.get_relevant_context_two_step(
                question=question,
                options=options_list,
                top_k=top_k,
                valid_book_names=book_names,
                use_metadata_filter=book_names is not None
            )

        # Standard RAG
        return self.rag.get_relevant_context(question, options_list, top_k)

    def _call_llm(self, messages: List[Dict], temperature: float = 0.3) -> str:
        """Call the LLM and return the response."""
        call_kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 512
        }
        if self.api_base:
            call_kwargs["base_url"] = self.api_base
        response = openai.ChatCompletion.create(**call_kwargs)
        return response.choices[0].message.content

    def _parse_direct_response(self, response: str) -> tuple:
        """Parse V0/V1 direct response into answer, confidence, reasoning."""
        answer = None
        confidence = 0.5
        reasoning = ""

        # Extract answer
        for line in response.split("\n"):
            line = line.strip()
            if line.startswith("ANSWER:"):
                answer = line.split("ANSWER:")[1].strip()[0].upper()
            elif line.startswith("CONF:"):
                try:
                    confidence = float(line.split("CONF:")[1].strip())
                except ValueError:
                    confidence = 0.5
            elif line.startswith("REASONING:"):
                reasoning = line.split("REASONING:")[1].strip()

        # Fallback: try to find A/B/C/D in response
        if not answer:
            for char in ["A", "B", "C", "D"]:
                if char in response.upper():
                    answer = char
                    break

        return answer, confidence, reasoning or response[:500]

    def solve_batch(
        self,
        questions: List[MedQAQuestion],
        variant: str = "V3",
        **kwargs
    ) -> List[SolveResult]:
        """Solve a batch of questions."""
        results = []
        for q in questions:
            result = self.solve(
                question=q.question,
                options=q.options,
                correct_answer=q.answer,
                question_id=q.question_id,
                variant=variant,
                **kwargs
            )
            results.append(result)
        return results


# =============================================================================
# Standalone Usage
# =============================================================================

def solve_single_question(
    question: str,
    options: Dict[str, str],
    correct_answer: str,
    variant: str = "V3",
    api_key: str = None
) -> SolveResult:
    """
    Convenience function to solve a single question.

    Args:
        question: The MedQA question
        options: Dict of options
        correct_answer: The correct answer key
        variant: Which variant to use
        api_key: OpenAI API key

    Returns:
        SolveResult
    """
    if api_key is None:
        import os
        api_key = os.environ.get("OPENAI_API_KEY", "")

    system = MedQASystem(api_key)
    return system.solve(question, options, correct_answer, variant=variant)


if __name__ == "__main__":
    # Example usage
    import os

    api_key = os.environ.get("OPENAI_API_KEY", "your-key")

    question = """
    A 65-year-old man with hypertension and diabetes presents with progressive
    shortness of breath. Exam shows elevated JVP, bilateral crackles, and
    peripheral edema. EF is 30%. Which medication improves survival?
    """

    options = {
        "A": "Digoxin",
        "B": "ACE inhibitor",
        "C": "Calcium channel blocker",
        "D": "Nitrate alone"
    }

    system = MedQASystem(api_key)

    # Test all variants
    for variant in ["V0", "V1", "V2", "V3", "V4"]:
        result = system.solve(
            question=question,
            options=options,
            correct_answer="B",
            question_id="demo_001",
            variant=variant
        )
        print(f"\n{variant}: {result.predicted_answer} (correct: {result.is_correct})")
