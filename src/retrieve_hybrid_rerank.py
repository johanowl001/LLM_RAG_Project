"""Rerank a broad hybrid retrieval candidate set with a local cross-encoder."""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path
from typing import Any

from sentence_transformers import CrossEncoder

from retrieve_hybrid import DEFAULT_BM25_INDEX, DEFAULT_INDEX_DIR, HybridRetriever


DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class HybridRerankRetriever:
    """Retrieve with hybrid RRF, then rerank question/chunk pairs locally."""

    def __init__(
        self,
        index_dir: Path = DEFAULT_INDEX_DIR,
        bm25_index_path: Path = DEFAULT_BM25_INDEX,
        reranker_model: str = DEFAULT_RERANKER_MODEL,
    ):
        self.hybrid = HybridRetriever(index_dir, bm25_index_path)
        self.reranker_model = reranker_model
        self.reranker = CrossEncoder(reranker_model)

    def retrieve(
        self,
        question: str,
        *,
        top_k: int = 8,
        candidate_top_n: int = 20,
        dense_top_n: int = 20,
        bm25_top_n: int = 20,
        rrf_k: int = 60,
        batch_size: int = 16,
    ) -> list[dict[str, Any]]:
        if not question.strip():
            raise ValueError("question must not be empty")
        if min(top_k, candidate_top_n, dense_top_n, bm25_top_n, rrf_k, batch_size) <= 0:
            raise ValueError("retrieval depths, rrf_k, and batch_size must be positive")
        if top_k > candidate_top_n:
            raise ValueError("top_k must not exceed candidate_top_n")

        candidates = self.hybrid.retrieve(
            question,
            top_k=candidate_top_n,
            dense_top_n=dense_top_n,
            bm25_top_n=bm25_top_n,
            rrf_k=rrf_k,
        )
        if not candidates:
            return []

        scores = self.reranker.predict(
            [(question, candidate["text"]) for candidate in candidates],
            batch_size=batch_size,
            show_progress_bar=False,
        )
        scored = [
            {**candidate, "reranker_score": float(score)}
            for candidate, score in zip(candidates, scores)
        ]
        scored.sort(
            key=lambda result: (
                -result["reranker_score"],
                -result["rrf_score"],
                result.get("document_id", ""),
                result.get("chunk_index", 0),
            )
        )
        for rank, result in enumerate(scored, start=1):
            result["reranker_rank"] = rank
        return scored[:top_k]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--bm25-index", type=Path, default=DEFAULT_BM25_INDEX)
    parser.add_argument("--reranker-model", default=DEFAULT_RERANKER_MODEL)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--candidate-top-n", type=int, default=20)
    parser.add_argument("--dense-top-n", type=int, default=20)
    parser.add_argument("--bm25-top-n", type=int, default=20)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--snippet-chars", type=int, default=400)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    retriever = HybridRerankRetriever(
        args.index_dir, args.bm25_index, args.reranker_model
    )
    results = retriever.retrieve(
        args.question,
        top_k=args.top_k,
        candidate_top_n=args.candidate_top_n,
        dense_top_n=args.dense_top_n,
        bm25_top_n=args.bm25_top_n,
        rrf_k=args.rrf_k,
        batch_size=args.batch_size,
    )
    print(f"Question: {args.question}\n")
    for rank, result in enumerate(results, start=1):
        component_ranks = (
            f"dense={result.get('dense_rank', '-')} "
            f"bm25={result.get('bm25_rank', '-')}"
        )
        print(
            f"{rank}. reranker={result['reranker_score']:.6f} "
            f"rrf={result['rrf_score']:.6f} {component_ranks} | "
            f"{result['document_id']} chunk {result['chunk_index']}"
        )
        snippet = " ".join(result["text"].split())[: args.snippet_chars]
        print(
            textwrap.fill(
                snippet, 100, initial_indent="   ", subsequent_indent="   "
            )
        )


if __name__ == "__main__":
    main()
