"""
RAG sub-package.

Retrieval-Augmented Generation components:
- retriever: ChromaDB-backed retriever with two-step retrieval (MedAgent-Pro style)
- data_loader: MedQA dataset loading and processing utilities
"""

from .retriever import (
    MedQA_RAG,
    create_rag_from_documents,
    quick_retrieve,
)
from .data_loader import (
    MedQALoader,
    MedQAQuestion,
    MedQAExporter,
    MedQABatchProcessor,
)

__all__ = [
    # Retriever
    "MedQA_RAG",
    "create_rag_from_documents",
    "quick_retrieve",
    # Data loader
    "MedQALoader",
    "MedQAQuestion",
    "MedQAExporter",
    "MedQABatchProcessor",
]