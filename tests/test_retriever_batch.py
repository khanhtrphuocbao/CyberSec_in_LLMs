import unittest

from medqa_rag.rag.retriever import MedQA_RAG


class RecordingEmbeddings:
    def __init__(self):
        self.batches = []

    def embed_documents(self, queries):
        self.batches.append(queries)
        return [[float(index)] for index, _ in enumerate(queries)]


class FakeCollection:
    def __init__(self):
        self.query_embeddings = []

    def query(self, *, query_embeddings, n_results, include):
        self.query_embeddings.append((query_embeddings, n_results, include))
        return {
            "documents": [["cardiology"], ["ethics"]],
            "metadatas": [[{"source": "Book A"}], [{"source": "Book B"}]],
            "distances": [[0.1], [0.2]],
        }


class FakeVectorStore:
    def __init__(self):
        self._collection = FakeCollection()


class BatchRetrievalTests(unittest.TestCase):
    def test_embeds_queries_once_and_returns_contexts_in_request_order(self):
        rag = MedQA_RAG.__new__(MedQA_RAG)
        rag._initialized = True
        rag.vectorstore = FakeVectorStore()
        rag.embeddings = RecordingEmbeddings()

        contexts = rag.get_relevant_context_batch(
            [("Question 1", ["A1", "B1"]), ("Question 2", ["A2", "B2"])], top_k=1
        )

        self.assertEqual(rag.embeddings.batches, [["Question 1 (A) A1 (B) B1", "Question 2 (A) A2 (B) B2"]])
        self.assertEqual(contexts, ["[Source 1 - Book A]:\ncardiology", "[Source 1 - Book B]:\nethics"])
