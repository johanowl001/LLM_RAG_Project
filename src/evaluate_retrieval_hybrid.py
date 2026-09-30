"""Evaluate hybrid retrieval on the fixed 12-question benchmark."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evaluate import load_json, retrieval_metrics, validate_questions
from retrieve_hybrid import DEFAULT_BM25_INDEX, DEFAULT_INDEX_DIR, HybridRetriever


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUESTIONS = PROJECT_ROOT / "evals" / "questions.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "evals" / "results" / "retrieval_hybrid"
DEFAULT_DENSE_BASELINE = (
    PROJECT_ROOT / "evals" / "results" / "retrieval_experiments" / "artifacts"
    / "chunk_800_overlap_100" / "top_k_8.json"
)


def compact(result: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in result.items() if key != "text"}


def run(args: argparse.Namespace) -> dict[str, Any]:
    questions = validate_questions(load_json(args.questions))
    retriever = HybridRetriever(args.index_dir, args.bm25_index)
    dense_artifact = load_json(args.dense_baseline)
    dense_by_id = {record["id"]: record for record in dense_artifact["results"]}
    records: list[dict[str, Any]] = []

    retriever.retrieve("retrieval timing warm-up", top_k=1,
                       dense_top_n=args.dense_top_n, bm25_top_n=args.bm25_top_n,
                       rrf_k=args.rrf_k)
    for item in questions:
        started = time.perf_counter()
        retrieved = retriever.retrieve(item["question"], top_k=args.top_k,
                                       dense_top_n=args.dense_top_n,
                                       bm25_top_n=args.bm25_top_n, rrf_k=args.rrf_k)
        latency = time.perf_counter() - started
        chunks = [compact(result) for result in retrieved]
        metrics = retrieval_metrics(item["expected_documents"], chunks)
        metrics["latency_seconds"] = latency
        dense_hit = dense_by_id[item["id"]]["retrieval"]["document_hit_at_k"]
        records.append({"id": item["id"], "question": item["question"],
                        "expected_documents": item["expected_documents"],
                        "retrieval": metrics, "retrieved_chunks": chunks,
                        "dense_baseline_hit": dense_hit,
                        "hit_changed_vs_dense": metrics["document_hit_at_k"] != dense_hit})
        print(f"{item['id']}: hybrid_hit={metrics['document_hit_at_k']} dense_hit={dense_hit}")

    changed = [record["id"] for record in records if record["hit_changed_vs_dense"]]
    newly_hit = [record["id"] for record in records if record["hit_changed_vs_dense"] and record["retrieval"]["document_hit_at_k"]]
    newly_missed = [record["id"] for record in records if record["hit_changed_vs_dense"] and not record["retrieval"]["document_hit_at_k"]]
    summary = {
        "question_count": len(records),
        "document_hit_at_k_rate": statistics.mean(r["retrieval"]["document_hit_at_k"] for r in records),
        "all_expected_documents_present_rate": statistics.mean(r["retrieval"]["all_expected_documents_present"] for r in records),
        "mean_expected_document_recall_at_k": statistics.mean(r["retrieval"]["expected_document_recall_at_k"] for r in records),
        "mean_retrieval_latency_seconds": statistics.mean(r["retrieval"]["latency_seconds"] for r in records),
        "retrieval_failure_count": sum(not r["retrieval"]["document_hit_at_k"] for r in records),
        "failure_question_ids": [r["id"] for r in records if not r["retrieval"]["document_hit_at_k"]],
        "dense_baseline_hit_rate": dense_artifact["summary"]["document_hit_at_k_rate"],
        "dense_baseline_failure_question_ids": [r["id"] for r in dense_artifact["results"] if not r["retrieval"]["document_hit_at_k"]],
        "hit_rate_delta_vs_dense": statistics.mean(r["retrieval"]["document_hit_at_k"] for r in records) - dense_artifact["summary"]["document_hit_at_k_rate"],
        "changed_question_ids": changed,
        "newly_hit_question_ids": newly_hit,
        "newly_missed_question_ids": newly_missed,
        "focus_questions": {qid: next(r for r in records if r["id"] == qid)["retrieval"]["document_hit_at_k"] for qid in ("q03", "q11", "q12")},
    }
    artifact = {"run": {"created_at": datetime.now(timezone.utc).isoformat(),
                         "questions_path": str(args.questions.resolve()),
                         "index_dir": str(args.index_dir.resolve()),
                         "bm25_index": str(args.bm25_index.resolve()),
                         "chunk_size": 800, "overlap": 100, "top_k": args.top_k,
                         "dense_top_n": args.dense_top_n, "bm25_top_n": args.bm25_top_n,
                         "fusion": "reciprocal_rank_fusion", "rrf_k": args.rrf_k,
                         "reranker": None},
                "summary": summary, "results": records}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output or args.output_dir / f"hybrid_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "latest_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved: {output.resolve()}\n{json.dumps(summary, indent=2)}")
    return artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--bm25-index", type=Path, default=DEFAULT_BM25_INDEX)
    parser.add_argument("--dense-baseline", type=Path, default=DEFAULT_DENSE_BASELINE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--dense-top-n", type=int, default=15)
    parser.add_argument("--bm25-top-n", type=int, default=15)
    parser.add_argument("--rrf-k", type=int, default=60)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
