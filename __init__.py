"""
MedQA-RAG: Multi-Agent Medical Question Answering System
=====================================================

A text-only adaptation of MedAgent-Pro for the MedQA-USMLE dataset.

Architecture
------------
::
    medqa_rag/
    ├── core/         - System orchestration (5-variant ablation)
    ├── agents/       - Planner, Examiner, Evaluator
    ├── rag/          - ChromaDB retriever + two-step retrieval + dataset loader
    ├── evaluation/   - Benchmark runner for 1273 MedQA questions
    └── scripts/      - Standalone ingestion utilities

Default paths on this machine
-----------------------------
- ChromaDB (with metadata injection):  ~/MedQA_ChromaDB_Injected
- MedQA-USMLE test set (JSONL):        ~/dataset_MedQA-USMLE/questions/US/test.jsonl

Usage
-----
    from medqa_rag import MedQASystem, MedQALoader, run_evaluation

    # Load questions (uses default test.jsonl path)
    loader = MedQALoader()
    questions = loader.load_test_set()  # 1273 questions

    # Initialize system (points to default ChromaDB)
    system = MedQASystem(api_key="your-key", use_two_step_retrieval=True)

    # Solve with a specific variant
    result = system.solve(question, options, correct_answer, variant="V3")

    # Run full benchmark
    report = run_evaluation()
"""

# Configuration
from .config import (
    Config,
    load_config,
    get_api_key,
    get_rag_config,
    get_model_config,
    require_api_key,
)

# Agents
from .agents import (
    MedQA_Planner,
    ReasoningStep,
    MedQA_Examiner,
    ReasoningResult,
    OptionAnalysis,
    MedQA_Evaluator,
    VerificationResult,
    EvaluationStatus,
)

# RAG + Data loading
from .rag import (
    MedQA_RAG,
    create_rag_from_documents,
    quick_retrieve,
    MedQALoader,
    MedQAQuestion,
    MedQAExporter,
    MedQABatchProcessor,
)

# Core system (5-variant ablation)
from .core import MedQASystem, SolveResult, Variant

# Evaluation runner
from .evaluation import (
    MedQAEvaluator,
    EvaluationMetrics,
    EvaluationReport,
    run_evaluation,
)

__version__ = "2.0.0"
__all__ = [
    # Configuration
    "Config",
    "load_config",
    "get_api_key",
    "get_rag_config",
    "get_model_config",
    "require_api_key",
    # Agents
    "MedQA_Planner",
    "ReasoningStep",
    "MedQA_Examiner",
    "ReasoningResult",
    "OptionAnalysis",
    "MedQA_Evaluator",
    "VerificationResult",
    "EvaluationStatus",
    # RAG + Data
    "MedQA_RAG",
    "create_rag_from_documents",
    "quick_retrieve",
    "MedQALoader",
    "MedQAQuestion",
    "MedQAExporter",
    "MedQABatchProcessor",
    # Core system
    "MedQASystem",
    "SolveResult",
    "Variant",
    # Evaluation
    "MedQAEvaluator",
    "EvaluationMetrics",
    "EvaluationReport",
    "run_evaluation",
]