"""Tests for controlled retrieval experiment aggregation and ranking."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from run_retrieval_experiments import evaluate_index, ranking_key  # noqa: E402


class FakeRetriever:
    def retrieve(self, question: str, top_k: int):
        results = {
            "single": [{"document_id": "a"}],
            "multi": [{"document_id": "b"}, {"document_id": "other"}],
        }
        return results[question][:top_k], 0.25


class RetrievalExperimentTest(unittest.TestCase):
    def test_multi_document_rate_excludes_single_document_questions(self) -> None:
        questions = [
            {"id": "q1", "question": "single", "expected_documents": ["a"]},
            {"id": "q2", "question": "multi", "expected_documents": ["b", "c"]},
        ]
        _, summary = evaluate_index(FakeRetriever(), questions, 3)
        self.assertEqual(summary["document_hit_at_k_rate"], 1.0)
        self.assertEqual(summary["multi_document_all_expected_documents_present_rate"], 0.0)
        self.assertEqual(summary["mean_expected_document_recall_at_k"], 0.75)

    def test_ranking_prefers_hit_rate_before_latency(self) -> None:
        strong = {
            "document_hit_at_k_rate": 1.0,
            "multi_document_all_expected_documents_present_rate": 0.5,
            "mean_expected_document_recall_at_k": 0.8,
            "mean_retrieval_latency_seconds": 1.0,
        }
        fast_but_weaker = {**strong, "document_hit_at_k_rate": 0.9, "mean_retrieval_latency_seconds": 0.1}
        self.assertLess(ranking_key(strong), ranking_key(fast_but_weaker))


if __name__ == "__main__":
    unittest.main()
