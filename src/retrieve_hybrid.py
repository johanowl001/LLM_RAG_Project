"""Hybrid dense/FAISS and BM25 retrieval with reciprocal-rank fusion."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import re
import textwrap
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INDEX_DIR = (
    PROJECT_ROOT / "evals" / "results" / "retrieval_experiments" / "artifacts"
    / "chunk_800_overlap_100" / "vector_index"
)
DEFAULT_BM25_INDEX = PROJECT_ROOT / "evals" / "results" / "retrieval_hybrid" / "bm25_index.json.gz"
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:[.'%-][A-Za-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    """Return simple case-folded terms suitable for financial prose."""
    return TOKEN_PATTERN.findall(text.casefold())


def _load_metadata(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


class BM25Index:
    """Small persistent BM25 implementation with no additional dependency."""

    def __init__(self, document_count: int, avgdl: float, doc_lengths: list[int],
                 postings: dict[str, list[list[int]]], *, k1: float = 1.5,
                 b: float = 0.75):
        self.document_count = document_count
        self.avgdl = avgdl
        self.doc_lengths = doc_lengths
        self.postings = postings
        self.k1 = k1
        self.b = b

    @classmethod
    def build(cls, texts: list[str], *, k1: float = 1.5, b: float = 0.75) -> "BM25Index":
        postings: dict[str, list[list[int]]] = defaultdict(list)
        doc_lengths: list[int] = []
        for position, text in enumerate(texts):
            terms = tokenize(text)
            doc_lengths.append(len(terms))
            for term, frequency in Counter(terms).items():
                postings[term].append([position, frequency])
        return cls(len(texts), sum(doc_lengths) / len(texts), doc_lengths, dict(postings), k1=k1, b=b)

    @classmethod
    def load_or_build(cls, path: Path, texts: list[str]) -> "BM25Index":
        if path.exists():
            with gzip.open(path, "rt", encoding="utf-8") as source:
                payload = json.load(source)
            if payload["document_count"] != len(texts):
                raise ValueError("BM25 index and chunk metadata have different lengths")
            return cls(**payload)
        index = cls.build(texts)
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wt", encoding="utf-8") as target:
            json.dump(index.to_dict(), target, separators=(",", ":"))
        return index

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_count": self.document_count,
            "avgdl": self.avgdl,
            "doc_lengths": self.doc_lengths,
            "postings": self.postings,
            "k1": self.k1,
            "b": self.b,
        }

    def search(self, query: str, top_n: int) -> list[tuple[int, float]]:
        scores: dict[int, float] = defaultdict(float)
        for term in set(tokenize(query)):
            entries = self.postings.get(term, [])
            df = len(entries)
            if not df:
                continue
            idf = math.log(1.0 + (self.document_count - df + 0.5) / (df + 0.5))
            for position, frequency in entries:
                length_norm = 1.0 - self.b + self.b * self.doc_lengths[position] / self.avgdl
                scores[position] += idf * frequency * (self.k1 + 1.0) / (frequency + self.k1 * length_norm)
        return sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:top_n]


class HybridRetriever:
    """Load dense and lexical indexes once and fuse their rankings with RRF."""

    def __init__(self, index_dir: Path = DEFAULT_INDEX_DIR,
                 bm25_index_path: Path = DEFAULT_BM25_INDEX):
        self.index_dir = index_dir
        self.config = json.loads((index_dir / "config.json").read_text(encoding="utf-8"))
        self.metadata = _load_metadata(index_dir / "metadata.jsonl")
        self.dense_index = faiss.read_index(str(index_dir / "chunks.faiss"))
        if self.dense_index.ntotal != len(self.metadata):
            raise ValueError("FAISS index and metadata mapping have different lengths")
        self.model = SentenceTransformer(self.config["model_name"])
        self.bm25 = BM25Index.load_or_build(
            bm25_index_path, [record["text"] for record in self.metadata]
        )

    def retrieve(self, question: str, *, top_k: int = 8, dense_top_n: int = 15,
                 bm25_top_n: int = 15, rrf_k: int = 60) -> list[dict[str, Any]]:
        if not question.strip():
            raise ValueError("question must not be empty")
        if min(top_k, dense_top_n, bm25_top_n, rrf_k) <= 0:
            raise ValueError("retrieval depths and rrf_k must be positive")

        query = self.config.get("query_prefix", "") + question
        vector = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
        dense_scores, dense_positions = self.dense_index.search(
            vector, min(dense_top_n, self.dense_index.ntotal)
        )
        dense = [(int(position), float(score)) for score, position in zip(dense_scores[0], dense_positions[0]) if position >= 0]
        lexical = self.bm25.search(question, min(bm25_top_n, len(self.metadata)))

        fused: dict[int, dict[str, Any]] = {}
        for component, ranking in (("dense", dense), ("bm25", lexical)):
            for rank, (position, score) in enumerate(ranking, start=1):
                entry = fused.setdefault(position, {"rrf_score": 0.0})
                entry["rrf_score"] += 1.0 / (rrf_k + rank)
                entry[f"{component}_rank"] = rank
                entry[f"{component}_score"] = score

        ordered = sorted(fused.items(), key=lambda item: (-item[1]["rrf_score"], item[0]))[:top_k]
        return [{**details, **self.metadata[position]} for position, details in ordered]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--bm25-index", type=Path, default=DEFAULT_BM25_INDEX)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--dense-top-n", type=int, default=15)
    parser.add_argument("--bm25-top-n", type=int, default=15)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--snippet-chars", type=int, default=400)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    retriever = HybridRetriever(args.index_dir, args.bm25_index)
    results = retriever.retrieve(args.question, top_k=args.top_k, dense_top_n=args.dense_top_n,
                                 bm25_top_n=args.bm25_top_n, rrf_k=args.rrf_k)
    print(f"Question: {args.question}\n")
    for rank, result in enumerate(results, start=1):
        ranks = f"dense={result.get('dense_rank', '-')} bm25={result.get('bm25_rank', '-')}"
        print(f"{rank}. rrf={result['rrf_score']:.6f} {ranks} | {result['document_id']} chunk {result['chunk_index']}")
        print(textwrap.fill(" ".join(result["text"].split())[:args.snippet_chars], 100, initial_indent="   ", subsequent_indent="   "))


if __name__ == "__main__":
    main()
