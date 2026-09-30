"""Tests for the separate OpenAI generation path."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from generate_openai import DEFAULT_INDEX_DIR, generate_answer  # noqa: E402


class OpenAIGeneratorTest(unittest.TestCase):
    def test_missing_key_fails_before_import_or_network(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "OPENAI_API_KEY is not set"):
                generate_answer("prompt")

    def test_default_index_is_winning_800_100_artifact(self) -> None:
        self.assertIn("chunk_800_overlap_100", str(DEFAULT_INDEX_DIR))


if __name__ == "__main__":
    unittest.main()
