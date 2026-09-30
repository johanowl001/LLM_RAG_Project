"""Unit tests for full-RAG scoring and bounded prompt construction."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evaluate_full_rag import (  # noqa: E402
    build_bounded_prompt,
    deterministic_answer_scores,
    parse_json_object,
)


class FullRAGEvaluationTest(unittest.TestCase):
    def test_reference_and_citations_are_scored_separately(self) -> None:
        scores = deterministic_answer_scores(
            "AWS sales were $30.9 billion, up 17.5% [S1].",
            {"reference_answer": "AWS sales increased 17.5% to $30.9 billion."},
            source_count=2,
        )
        self.assertGreater(scores["correctness"]["score_0_to_1"], 0.5)
        self.assertEqual(scores["citation_source_support"]["score_0_to_1"], 1.0)
        self.assertGreater(scores["completeness"]["score_0_to_1"], 0.5)

    def test_bounded_prompt_keeps_every_source_label(self) -> None:
        chunks = [
            {"document_id": f"doc-{number}", "chunk_index": 0, "text": "word " * 100}
            for number in range(1, 9)
        ]
        prompt = build_bounded_prompt("Question?", chunks, max_chars_per_chunk=40)
        for number in range(1, 9):
            self.assertIn(f"[S{number}:", prompt)
        self.assertLess(len(prompt), 2000)

    def test_judge_parser_enforces_component_range(self) -> None:
        valid = {
            component: {"score": 4, "reason": "supported"}
            for component in (
                "correctness", "groundedness", "citation_source_support", "completeness"
            )
        }
        self.assertEqual(parse_json_object(json_text(valid))["correctness"]["score"], 4)
        valid["correctness"]["score"] = 6
        with self.assertRaisesRegex(ValueError, "between 1 and 5"):
            parse_json_object(json_text(valid))


def json_text(value: object) -> str:
    import json

    return json.dumps(value)


if __name__ == "__main__":
    unittest.main()
