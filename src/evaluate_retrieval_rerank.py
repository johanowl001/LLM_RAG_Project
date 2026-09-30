"""Evaluate hybrid retrieval plus local cross-encoder reranking."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evaluate import load_json, retrieval_metrics, validate_questions
from retrieve_hybrid import DEFAULT_BM25_INDEX, DEFAULT_INDEX_DIR
from retrieve_hybrid_rerank import DEFAULT_RERANKER_MODEL, HybridRerankRetriever


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUESTIONS = PROJECT_ROOT / "evals" / "questions.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "evals" / "results" / "retrieval_rerank"
DEFAULT_DENSE_BASELINE = (
    PROJECT_ROOT / "evals" / "results" / "retrieval_experiments" / "artifacts"
    / "chunk_800_overlap_100" / "top_k_8.json"
)
DEFAULT_HYBRID_BASELINE = (
    PROJECT_ROOT / "evals" / "results" / "retrieval_hybrid"
    / "hybrid_20260930_135544.json"
)


def compact(result: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in result.items() if key != "text"}


def run(args: argparse.Namespace) -> dict[str, Any]:
    questions = validate_questions(load_json(args.questions))
    dense_artifact = load_json(args.dense_baseline)
    hybrid_artifact = load_json(args.hybrid_baseline)
    dense_by_id = {record["id"]: record for record in dense_artifact["results"]}
    hybrid_by_id = {record["id"]: record for record in hybrid_artifact["results"]}
    question_ids = {item["id"] for item in questions}
    if question_ids != dense_by_id.keys() or question_ids != hybrid_by_id.keys():
        raise ValueError("questions and baseline artifacts must contain the same question IDs")

    retriever = HybridRerankRetriever(
        args.index_dir, args.bm25_index, args.reranker_model
    )
    records: list[dict[str, Any]] = []

    retriever.retrieve(
        "retrieval timing warm-up",
        top_k=1,
        candidate_top_n=args.candidate_top_n,
        dense_top_n=args.dense_top_n,
        bm25_top_n=args.bm25_top_n,
        rrf_k=args.rrf_k,
        batch_size=args.batch_size,
    )
    for item in questions:
        started = time.perf_counter()
        retrieved = retriever.retrieve(
            item["question"],
            top_k=args.top_k,
            candidate_top_n=args.candidate_top_n,
            dense_top_n=args.dense_top_n,
            bm25_top_n=args.bm25_top_n,
            rrf_k=args.rrf_k,
            batch_size=args.batch_size,
        )
        latency = time.perf_counter() - started
        chunks = [compact(result) for result in retrieved]
        metrics = retrieval_metrics(item["expected_documents"], chunks)
        metrics["latency_seconds"] = latency
        dense_hit = dense_by_id[item["id"]]["retrieval"]["document_hit_at_k"]
        hybrid_hit = hybrid_by_id[item["id"]]["retrieval"]["document_hit_at_k"]
        records.append(
            {
                "id": item["id"],
                "question": item["question"],
                "expected_documents": item["expected_documents"],
                "retrieval": metrics,
                "retrieved_chunks": chunks,
                "dense_baseline_hit": dense_hit,
                "hybrid_baseline_hit": hybrid_hit,
                "hit_changed_vs_dense": metrics["document_hit_at_k"] != dense_hit,
                "hit_changed_vs_hybrid": metrics["document_hit_at_k"] != hybrid_hit,
            }
        )
        print(
            f"{item['id']}: rerank_hit={metrics['document_hit_at_k']} "
            f"hybrid_hit={hybrid_hit} dense_hit={dense_hit} "
            f"latency={latency:.3f}s"
        )

    rerank_hit_rate = statistics.mean(
        record["retrieval"]["document_hit_at_k"] for record in records
    )
    dense_hit_rate = dense_artifact["summary"]["document_hit_at_k_rate"]
    hybrid_hit_rate = hybrid_artifact["summary"]["document_hit_at_k_rate"]
    dense_latency = dense_artifact["summary"]["mean_retrieval_latency_seconds"]
    hybrid_latency = hybrid_artifact["summary"]["mean_retrieval_latency_seconds"]
    rerank_latency = statistics.mean(
        record["retrieval"]["latency_seconds"] for record in records
    )
    improved_vs_hybrid = [
        record["id"]
        for record in records
        if record["retrieval"]["document_hit_at_k"]
        and not record["hybrid_baseline_hit"]
    ]
    regressions_vs_hybrid = [
        record["id"]
        for record in records
        if not record["retrieval"]["document_hit_at_k"]
        and record["hybrid_baseline_hit"]
    ]
    summary = {
        "question_count": len(records),
        "document_hit_at_k_rate": rerank_hit_rate,
        "all_expected_documents_present_rate": statistics.mean(
            record["retrieval"]["all_expected_documents_present"] for record in records
        ),
        "mean_expected_document_recall_at_k": statistics.mean(
            record["retrieval"]["expected_document_recall_at_k"] for record in records
        ),
        "mean_retrieval_latency_seconds": rerank_latency,
        "retrieval_failure_count": sum(
            not record["retrieval"]["document_hit_at_k"] for record in records
        ),
        "failure_question_ids": [
            record["id"]
            for record in records
            if not record["retrieval"]["document_hit_at_k"]
        ],
        "dense_baseline_hit_rate": dense_hit_rate,
        "hybrid_baseline_hit_rate": hybrid_hit_rate,
        "dense_baseline_mean_retrieval_latency_seconds": dense_latency,
        "hybrid_baseline_mean_retrieval_latency_seconds": hybrid_latency,
        "latency_delta_seconds_vs_dense": rerank_latency - dense_latency,
        "latency_delta_seconds_vs_hybrid": rerank_latency - hybrid_latency,
        "hit_rate_delta_vs_dense": rerank_hit_rate - dense_hit_rate,
        "hit_rate_delta_vs_hybrid": rerank_hit_rate - hybrid_hit_rate,
        "improved_question_ids_vs_hybrid": improved_vs_hybrid,
        "regression_question_ids_vs_hybrid": regressions_vs_hybrid,
        "preserves_or_improves_hybrid_hit_rate": rerank_hit_rate >= hybrid_hit_rate,
        "has_regressions_vs_hybrid": bool(regressions_vs_hybrid),
        "focus_questions": {
            question_id: next(
                record for record in records if record["id"] == question_id
            )["retrieval"]["document_hit_at_k"]
            for question_id in ("q03", "q11", "q12")
        },
    }
    artifact = {
        "run": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "questions_path": str(args.questions.resolve()),
            "index_dir": str(args.index_dir.resolve()),
            "bm25_index": str(args.bm25_index.resolve()),
            "dense_baseline": str(args.dense_baseline.resolve()),
            "hybrid_baseline": str(args.hybrid_baseline.resolve()),
            "chunk_size": 800,
            "overlap": 100,
            "top_k": args.top_k,
            "candidate_top_n": args.candidate_top_n,
            "dense_top_n": args.dense_top_n,
            "bm25_top_n": args.bm25_top_n,
            "fusion": "reciprocal_rank_fusion",
            "rrf_k": args.rrf_k,
            "reranker": args.reranker_model,
            "reranker_batch_size": args.batch_size,
        },
        "summary": summary,
        "results": records,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output or args.output_dir / (
        f"rerank_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "latest_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\nSaved: {output.resolve()}\n{json.dumps(summary, indent=2)}")
    return artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--bm25-index", type=Path, default=DEFAULT_BM25_INDEX)
    parser.add_argument("--dense-baseline", type=Path, default=DEFAULT_DENSE_BASELINE)
    parser.add_argument("--hybrid-baseline", type=Path, default=DEFAULT_HYBRID_BASELINE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reranker-model", default=DEFAULT_RERANKER_MODEL)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--candidate-top-n", type=int, default=20)
    parser.add_argument("--dense-top-n", type=int, default=20)
    parser.add_argument("--bm25-top-n", type=int, default=20)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
