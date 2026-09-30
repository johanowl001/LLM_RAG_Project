"""Run the 12-question full-RAG benchmark with OpenAI generation only."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

from evaluate_full_rag import (
    DEFAULT_INDEX_DIR,
    DEFAULT_MAX_CHARS_PER_CHUNK,
    DEFAULT_QUESTIONS,
    DEFAULT_RESULTS_DIR,
    DEFAULT_TOP_K,
    run,
)
from generate_openai import DEFAULT_MODEL, generate_answer

DEFAULT_OPENAI_RESULTS_DIR = DEFAULT_RESULTS_DIR / "openai"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_OPENAI_RESULTS_DIR)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", DEFAULT_MODEL))
    parser.add_argument(
        "--max-chars-per-chunk", type=int, default=DEFAULT_MAX_CHARS_PER_CHUNK
    )
    parser.add_argument("--limit", type=int, help="Run only the first N questions")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.max_chars_per_chunk <= 0:
        parser.error("--max-chars-per-chunk must be positive")
    args.top_k = DEFAULT_TOP_K
    args.generator = "openai"
    args.judge = "none"
    args.judge_model = None
    args.ollama_url = ""
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = args.results_dir / f"openai_{args.model}_{timestamp}.json"
    return args


def _openai_backend(backend: str, model: str | None, ollama_url: str):
    """Adapter matching evaluate_full_rag's backend factory contract."""
    selected = model or os.getenv("OPENAI_MODEL", DEFAULT_MODEL)
    return lambda prompt: generate_answer(prompt, selected), selected


if __name__ == "__main__":
    try:
        import evaluate_full_rag

        evaluate_full_rag.backend_generator = _openai_backend
        run(parse_args())
    except KeyboardInterrupt:
        sys.exit(130)
