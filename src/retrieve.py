"""Retrieve the most semantically similar Amazon chunks for a question."""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INDEX_DIR = PROJECT_ROOT / "data" / "processed" / "vector_index"


def load_metadata(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def retrieve(question: str, index_dir: Path, top_k: int = 5) -> list[dict[str, Any]]:
    """Embed a question and return the top-k chunks by cosine similarity."""
    if not question.strip():
        raise ValueError("question must not be empty")
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    config = json.loads((index_dir / "config.json").read_text(encoding="utf-8"))
    metadata = load_metadata(index_dir / "metadata.jsonl")
    index = faiss.read_index(str(index_dir / "chunks.faiss"))
    if index.ntotal != len(metadata):
        raise ValueError("FAISS index and metadata mapping have different lengths")

    model = SentenceTransformer(config["model_name"])
    query_text = config.get("query_prefix", "") + question
    query_vector = model.encode(
        [query_text], convert_to_numpy=True, normalize_embeddings=True
    ).astype(np.float32)
    scores, positions = index.search(query_vector, min(top_k, index.ntotal))

    results: list[dict[str, Any]] = []
    for score, position in zip(scores[0], positions[0]):
        if position < 0:
            continue
        results.append({"score": float(score), **metadata[int(position)]})
    return results


def print_results(question: str, results: list[dict[str, Any]], snippet_chars: int) -> None:
    print(f"Question: {question}\n")
    for rank, result in enumerate(results, start=1):
        metadata = ", ".join(
            str(value)
            for value in (
                result.get("document_id"),
                result.get("document_type"),
                result.get("year"),
                result.get("quarter"),
                f"chunk {result.get('chunk_index')}",
            )
            if value not in (None, "")
        )
        snippet = " ".join(result["text"].split())[:snippet_chars]
        print(f"{rank}. score={result['score']:.4f} | {metadata}")
        print(textwrap.fill(snippet, width=100, initial_indent="   ", subsequent_indent="   "))
        print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", help="Question to search for")
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--snippet-chars", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = retrieve(args.question, args.index_dir, args.top_k)
    print_results(args.question, results, args.snippet_chars)


if __name__ == "__main__":
    main()
