"""Unit tests for grounded prompt construction and citation labels."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from generate import build_prompt, source_label  # noqa: E402


class GenerateHelpersTest(unittest.TestCase):
    def setUp(self) -> None:
        self.result = {
            "document_id": "AMZN-Q2-2025-Earnings-Release",
            "year": 2025,
            "quarter": "Q2",
            "chunk_index": 3,
            "text": "AWS sales increased year over year.",
            "score": 0.82,
        }

    def test_source_label_uses_retrieval_metadata(self) -> None:
        self.assertEqual(
            source_label(self.result, 1),
            "S1: AMZN-Q2-2025-Earnings-Release, 2025 Q2, chunk 3",
        )

    def test_prompt_contains_question_context_and_grounding_rules(self) -> None:
        prompt = build_prompt("How did AWS perform?", [self.result])
        self.assertIn("How did AWS perform?", prompt)
        self.assertIn("AWS sales increased year over year.", prompt)
        self.assertIn("[S1:", prompt)
        self.assertIn("Use only the retrieved context", prompt)
        self.assertIn("does not provide enough evidence", prompt)

    def test_prompt_requires_retrieved_context(self) -> None:
        with self.assertRaisesRegex(ValueError, "At least one"):
            build_prompt("Question?", [])


if __name__ == "__main__":
    unittest.main()
