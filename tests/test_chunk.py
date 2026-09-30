"""Tests for the fixed-token chunking baseline."""

import importlib.util
import unittest
from pathlib import Path

import tiktoken


MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "chunk.py"
SPEC = importlib.util.spec_from_file_location("chunk", MODULE_PATH)
chunk = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(chunk)


class ChunkDocumentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.encoding = tiktoken.get_encoding("cl100k_base")
        self.metadata = {
            "company": "Amazon.com, Inc.",
            "ticker": "AMZN",
            "year": 2026,
            "quarter": "Q2",
            "document_type": "earnings_release",
            "source_file": "AMZN-Q2-2026-Earnings-Release.pdf",
            "text_file": "AMZN-Q2-2026-Earnings-Release.txt",
        }

    def test_uses_requested_size_and_overlap(self) -> None:
        chunks = chunk.chunk_document(
            self.metadata,
            "Amazon Web Services grew rapidly. " * 300,
            chunk_size=500,
            overlap=75,
            encoding=self.encoding,
        )

        self.assertGreater(len(chunks), 1)
        self.assertEqual(chunks[0]["token_count"], 500)
        self.assertEqual(chunks[1]["token_start"], 425)
        self.assertEqual(chunks[0]["token_end"] - chunks[1]["token_start"], 75)
        self.assertLessEqual(chunks[-1]["token_count"], 500)
        self.assertEqual(chunks[0]["year"], 2026)
        self.assertTrue(chunks[0]["source_filename"].endswith(".pdf"))

    def test_rejects_invalid_overlap(self) -> None:
        with self.assertRaisesRegex(ValueError, "smaller than chunk_size"):
            chunk.chunk_document(
                self.metadata,
                "sample",
                chunk_size=500,
                overlap=500,
                encoding=self.encoding,
            )


if __name__ == "__main__":
    unittest.main()
