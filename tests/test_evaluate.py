"""Tests for transparent evaluation metric calculations."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evaluate import aggregate_results, retrieval_metrics  # noqa: E402


class EvaluationMetricsTest(unittest.TestCase):
    def test_multi_document_metrics_distinguish_any_hit_from_all_present(self) -> None:
        metrics = retrieval_metrics(
            ["doc-a", "doc-b"],
            [{"document_id": "other"}, {"document_id": "doc-a"}],
        )
        self.assertTrue(metrics["document_hit_at_k"])
        self.assertEqual(metrics["expected_document_recall_at_k"], 0.5)
        self.assertFalse(metrics["all_expected_documents_present"])
        self.assertEqual(metrics["missing_documents"], ["doc-b"])
        self.assertEqual(metrics["first_expected_document_rank"], 2)

    def test_aggregate_uses_micro_recall_across_expected_documents(self) -> None:
        records = []
        for expected, found in [(["a"], ["a"]), (["b", "c"], ["b"])]:
            retrieved = [{"document_id": value} for value in found]
            metric = retrieval_metrics(expected, retrieved)
            metric["latency_seconds"] = 0.1
            records.append({"expected_documents": expected, "retrieval": metric})
        summary = aggregate_results(records)
        self.assertAlmostEqual(summary["micro_expected_document_recall_at_k"], 2 / 3)
        self.assertEqual(summary["incomplete_multi_document_count"], 1)


if __name__ == "__main__":
    unittest.main()
