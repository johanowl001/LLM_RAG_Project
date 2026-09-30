"""Small unit tests for the framework-light retrieval helpers."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from embed import load_chunks  # noqa: E402
from retrieve import load_metadata  # noqa: E402


class RetrievalHelpersTest(unittest.TestCase):
    def test_chunk_and_metadata_jsonl_stay_in_order(self) -> None:
        records = [
            {"chunk_id": "a_0000", "text": "first"},
            {"chunk_id": "b_0000", "text": "second"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.jsonl"
            path.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            self.assertEqual(load_chunks(path), records)
            self.assertEqual(load_metadata(path), records)

    def test_empty_chunk_text_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.jsonl"
            path.write_text('{"chunk_id": "empty", "text": ""}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "has no text"):
                load_chunks(path)


if __name__ == "__main__":
    unittest.main()
