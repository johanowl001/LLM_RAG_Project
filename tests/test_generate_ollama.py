"""Unit tests for the local Ollama generation path."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from generate_ollama import generate_answer  # noqa: E402


class OllamaGenerationTest(unittest.TestCase):
    @patch("generate_ollama.urllib.request.urlopen")
    def test_generate_answer_uses_local_api(self, urlopen) -> None:
        response = urlopen.return_value.__enter__.return_value
        response.read.return_value = (
            b'{"model":"qwen2.5:7b","response":"Grounded answer [S1]",'
            b'"prompt_eval_count":100,"eval_count":12}'
        )

        answer, metrics = generate_answer("prompt")

        self.assertEqual(answer, "Grounded answer [S1]")
        self.assertEqual(metrics["model"], "qwen2.5:7b")
        self.assertEqual(metrics["input_tokens"], 100)
        self.assertEqual(metrics["output_tokens"], 12)
        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "http://localhost:11434/api/generate")


if __name__ == "__main__":
    unittest.main()
