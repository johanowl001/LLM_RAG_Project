"""Run a transparent retrieval-and-generation baseline evaluation."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUESTIONS = PROJECT_ROOT / "evals" / "questions.json"
DEFAULT_INDEX_DIR = PROJECT_ROOT / "data" / "processed" / "vector_index"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "evals" / "results"
DEFAULT_TOP_K = 5


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_questions(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list) or not raw:
        raise ValueError("questions file must contain a non-empty JSON list")
    seen: set[str] = set()
    for position, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"question {position} must be an object")
        question_id = item.get("id")
        if not isinstance(question_id, str) or not question_id.strip():
            raise ValueError(f"question {position} needs a non-empty id")
        if question_id in seen:
            raise ValueError(f"duplicate question id: {question_id}")
        seen.add(question_id)
        if not isinstance(item.get("question"), str) or not item["question"].strip():
            raise ValueError(f"{question_id} needs a non-empty question")
        expected = item.get("expected_documents")
        if not isinstance(expected, list) or not expected or not all(
            isinstance(value, str) and value.strip() for value in expected
        ):
            raise ValueError(f"{question_id} needs expected_documents")
    return raw


class BaselineRetriever:
    """Load the baseline model and FAISS index once for the entire benchmark."""

    def __init__(self, index_dir: Path):
        self.index_dir = index_dir
        self.config = load_json(index_dir / "config.json")
        self.metadata = [
            json.loads(line)
            for line in (index_dir / "metadata.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.index = faiss.read_index(str(index_dir / "chunks.faiss"))
        if self.index.ntotal != len(self.metadata):
            raise ValueError("FAISS index and metadata mapping have different lengths")
        self.model = SentenceTransformer(self.config["model_name"])

    def retrieve(self, question: str, top_k: int) -> tuple[list[dict[str, Any]], float]:
        started = time.perf_counter()
        query = self.config.get("query_prefix", "") + question
        vector = self.model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        ).astype(np.float32)
        scores, positions = self.index.search(vector, min(top_k, self.index.ntotal))
        results = [
            {"score": float(score), **self.metadata[int(position)]}
            for score, position in zip(scores[0], positions[0])
            if position >= 0
        ]
        for result in results:
            result.pop("text", None)
        return results, time.perf_counter() - started


def retrieval_metrics(
    expected_documents: list[str], retrieved: list[dict[str, Any]]
) -> dict[str, Any]:
    retrieved_documents = list(dict.fromkeys(item["document_id"] for item in retrieved))
    expected_set = set(expected_documents)
    found = [name for name in expected_documents if name in retrieved_documents]
    missing = [name for name in expected_documents if name not in retrieved_documents]
    first_hit_rank = next(
        (rank for rank, item in enumerate(retrieved, start=1) if item["document_id"] in expected_set),
        None,
    )
    return {
        "document_hit_at_k": bool(found),
        "expected_document_recall_at_k": len(found) / len(expected_documents),
        "all_expected_documents_present": not missing,
        "found_documents": found,
        "missing_documents": missing,
        "retrieved_documents": retrieved_documents,
        "first_expected_document_rank": first_hit_rank,
    }


def aggregate_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    retrieval = [item["retrieval"] for item in results]
    generation = [item.get("generation") for item in results if item.get("generation")]
    complete_generation = [item for item in generation if item.get("status") == "completed"]
    expected_count = sum(len(item["expected_documents"]) for item in results)
    found_count = sum(len(item["retrieval"]["found_documents"]) for item in results)
    summary: dict[str, Any] = {
        "question_count": len(results),
        "document_hit_at_k_rate": sum(item["document_hit_at_k"] for item in retrieval) / len(retrieval),
        "micro_expected_document_recall_at_k": found_count / expected_count,
        "mean_expected_document_recall_at_k": statistics.mean(
            item["expected_document_recall_at_k"] for item in retrieval
        ),
        "all_expected_documents_present_rate": sum(
            item["all_expected_documents_present"] for item in retrieval
        ) / len(retrieval),
        "mean_retrieval_latency_seconds": statistics.mean(
            item["latency_seconds"] for item in retrieval
        ),
        "retrieval_failure_count": sum(not item["document_hit_at_k"] for item in retrieval),
        "incomplete_multi_document_count": sum(
            len(item["expected_documents"]) > 1
            and not item["retrieval"]["all_expected_documents_present"]
            for item in results
        ),
        "generation": {
            "requested_count": len(generation),
            "completed_count": len(complete_generation),
        },
    }
    if complete_generation:
        summary["generation"].update(
            {
                "mean_latency_seconds": statistics.mean(
                    item["latency_seconds"] for item in complete_generation
                ),
                "total_input_tokens": sum(
                    item.get("input_tokens") or 0 for item in complete_generation
                ),
                "total_output_tokens": sum(
                    item.get("output_tokens") or 0 for item in complete_generation
                ),
            }
        )
    return summary


def generator_for(name: str, model: str | None, ollama_url: str) -> tuple[Callable[..., Any] | None, str | None]:
    if name == "none":
        return None, None
    if name == "ollama":
        from generate_ollama import DEFAULT_MODEL, generate_answer

        selected_model = model or os.getenv("OLLAMA_MODEL", DEFAULT_MODEL)
        return lambda prompt: generate_answer(prompt, selected_model, ollama_url), selected_model
    from generate import DEFAULT_MODEL, generate_answer

    selected_model = model or os.getenv("OPENAI_MODEL", DEFAULT_MODEL)
    return lambda prompt: generate_answer(prompt, selected_model), selected_model


def run(args: argparse.Namespace) -> dict[str, Any]:
    questions = validate_questions(load_json(args.questions))
    retriever = BaselineRetriever(args.index_dir)
    generate, selected_model = generator_for(args.generator, args.model, args.ollama_url)
    if generate:
        from generate import build_prompt

    results: list[dict[str, Any]] = []
    for number, item in enumerate(questions, start=1):
        # Keep text for generation but omit it from the compact result artifact.
        started = time.perf_counter()
        query = retriever.config.get("query_prefix", "") + item["question"]
        vector = retriever.model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        ).astype(np.float32)
        scores, positions = retriever.index.search(vector, min(args.top_k, retriever.index.ntotal))
        full_results = [
            {"score": float(score), **retriever.metadata[int(position)]}
            for score, position in zip(scores[0], positions[0])
            if position >= 0
        ]
        retrieval_latency = time.perf_counter() - started
        compact_results = [
            {key: value for key, value in result.items() if key != "text"}
            for result in full_results
        ]
        metrics = retrieval_metrics(item["expected_documents"], compact_results)
        metrics["latency_seconds"] = retrieval_latency
        record: dict[str, Any] = {
            **item,
            "retrieval": metrics,
            "retrieved_chunks": compact_results,
            "answer_assessment": {
                "method": "manual",
                "status": "pending" if generate else "not_generated",
                "instructions": "Review the answer against answer_notes and retrieved evidence; set status to pass, partial, or fail and add reviewer_notes.",
                "reviewer_notes": "",
            },
        }
        if generate:
            try:
                answer, generation_metrics = generate(build_prompt(item["question"], full_results))
                record["answer"] = answer
                record["generation"] = {"status": "completed", **generation_metrics}
            except Exception as error:  # preserve retrieval results if one model call fails
                record["generation"] = {
                    "status": "error",
                    "model": selected_model,
                    "error": f"{type(error).__name__}: {error}",
                }
                if args.fail_fast:
                    raise
        results.append(record)
        print(
            f"[{number}/{len(questions)}] {item['id']}: "
            f"hit={metrics['document_hit_at_k']} all={metrics['all_expected_documents_present']}"
        )

    chunk_summary_path = PROJECT_ROOT / "data" / "processed" / "chunks" / "summary.json"
    artifact = {
        "run": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "questions_path": str(args.questions.resolve()),
            "index_dir": str(args.index_dir.resolve()),
            "top_k": args.top_k,
            "generator": args.generator,
            "generation_model": selected_model,
            "embedding": retriever.config,
            "chunking": load_json(chunk_summary_path) if chunk_summary_path.exists() else None,
        },
        "summary": {},
        "results": results,
    }
    artifact["summary"] = aggregate_results(results)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = args.output or args.results_dir / f"baseline_{timestamp}.json"
    output_path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved: {output_path.resolve()}")
    print(json.dumps(artifact["summary"], indent=2))
    return artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--generator", choices=("none", "ollama", "openai"), default="ollama")
    parser.add_argument("--model", help="Generation model; defaults to the selected backend's current default")
    parser.add_argument("--ollama-url", default=os.getenv("OLLAMA_URL", "http://localhost:11434"))
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()
    if args.top_k <= 0:
        parser.error("--top-k must be positive")
    return args


if __name__ == "__main__":
    run(parse_args())
