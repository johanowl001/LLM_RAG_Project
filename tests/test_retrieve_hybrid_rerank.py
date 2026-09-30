"""Unit tests for hybrid candidate reranking."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from retrieve_hybrid_rerank import HybridRerankRetriever  # noqa: E402


class FakeHybridRetriever:
    def retrieve(self, question: str, **kwargs):
        return [
            {
                "document_id": "lower",
                "chunk_index": 0,
                "text": "lower-scoring text",
                "rrf_score": 0.04,
                "dense_rank": 1,
                "dense_score": 0.8,
            },
            {
                "document_id": "higher",
                "chunk_index": 2,
                "text": "higher-scoring text",
                "rrf_score": 0.03,
                "bm25_rank": 1,
                "bm25_score": 7.5,
            },
        ]


class FakeCrossEncoder:
    def predict(self, pairs, **kwargs):
        return np.array([-2.0, 3.0])


class HybridRerankRetrieverTest(unittest.TestCase):
    def setUp(self) -> None:
        self.retriever = HybridRerankRetriever.__new__(HybridRerankRetriever)
        self.retriever.hybrid = FakeHybridRetriever()
        self.retriever.reranker = FakeCrossEncoder()

    def test_reranks_and_preserves_component_metadata(self) -> None:
        results = self.retriever.retrieve("question", top_k=2)
        self.assertEqual([result["document_id"] for result in results], ["higher", "lower"])
        self.assertEqual([result["reranker_rank"] for result in results], [1, 2])
        self.assertEqual(results[0]["reranker_score"], 3.0)
        self.assertEqual(results[0]["bm25_rank"], 1)
        self.assertEqual(results[1]["dense_rank"], 1)

    def test_rejects_final_depth_larger_than_candidate_pool(self) -> None:
        with self.assertRaisesRegex(ValueError, "top_k"):
            self.retriever.retrieve("question", top_k=9, candidate_top_n=8)


if __name__ == "__main__":
    unittest.main()
