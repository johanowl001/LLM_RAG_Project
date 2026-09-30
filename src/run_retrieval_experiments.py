"""Run a reproducible chunking/top-k retrieval experiment grid."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from chunk import DEFAULT_ENCODING, chunk_corpus
from embed import DEFAULT_MODEL, build_index
from evaluate import BaselineRetriever, load_json, retrieval_metrics, validate_questions


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_QUESTIONS = PROJECT_ROOT / "evals" / "questions.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "evals" / "results" / "retrieval_experiments"
DEFAULT_GRID = ((300, 50), (500, 75), (800, 100))
DEFAULT_TOP_K = (3, 5, 8)


def evaluate_index(
    retriever: BaselineRetriever,
    questions: list[dict[str, Any]],
    top_k: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Evaluate one already-loaded index at one top-k setting."""
    records: list[dict[str, Any]] = []
    for item in questions:
        retrieved, latency = retriever.retrieve(item["question"], top_k)
        metrics = retrieval_metrics(item["expected_documents"], retrieved)
        metrics["latency_seconds"] = latency
        records.append({
            "id": item["id"],
            "question": item["question"],
            "expected_documents": item["expected_documents"],
            "retrieval": metrics,
            "retrieved_chunks": retrieved,
        })

    multi = [record for record in records if len(record["expected_documents"]) > 1]
    summary = {
        "question_count": len(records),
        "multi_document_question_count": len(multi),
        "document_hit_at_k_rate": statistics.mean(
            record["retrieval"]["document_hit_at_k"] for record in records
        ),
        "all_expected_documents_present_rate": statistics.mean(
            record["retrieval"]["all_expected_documents_present"] for record in records
        ),
        "multi_document_all_expected_documents_present_rate": (
            statistics.mean(
                record["retrieval"]["all_expected_documents_present"] for record in multi
            )
            if multi else None
        ),
        "mean_expected_document_recall_at_k": statistics.mean(
            record["retrieval"]["expected_document_recall_at_k"] for record in records
        ),
        "mean_retrieval_latency_seconds": statistics.mean(
            record["retrieval"]["latency_seconds"] for record in records
        ),
    }
    return records, summary


def ranking_key(row: dict[str, Any]) -> tuple[float, float, float, float]:
    """Prefer broad hits, then strict multi-doc coverage, recall, and latency."""
    return (
        -row["document_hit_at_k_rate"],
        -row["multi_document_all_expected_documents_present_rate"],
        -row["mean_expected_document_recall_at_k"],
        row["mean_retrieval_latency_seconds"],
    )


def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    questions = validate_questions(load_json(args.questions))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for chunk_size, overlap in DEFAULT_GRID:
        config_id = f"chunk_{chunk_size}_overlap_{overlap}"
        config_dir = args.output_dir / "artifacts" / config_id
        chunks_dir = config_dir / "chunks"
        index_dir = config_dir / "vector_index"
        if args.reuse_existing and (index_dir / "chunks.faiss").exists():
            chunk_summary = load_json(chunks_dir / "summary.json")
            index_config = load_json(index_dir / "config.json")
            chunk_count = chunk_summary["total_chunks"]
        else:
            chunks, _ = chunk_corpus(
                args.processed_dir,
                chunks_dir,
                chunk_size=chunk_size,
                overlap=overlap,
                encoding_name=args.encoding,
            )
            index_config = build_index(
                chunks_dir / "chunks.jsonl",
                index_dir,
                model_name=args.model,
                batch_size=args.batch_size,
            )
            chunk_count = len(chunks)
        index_size_bytes = (index_dir / "chunks.faiss").stat().st_size
        retriever = BaselineRetriever(index_dir)
        # Exclude one-time model/backend initialization from every measured grid cell.
        retriever.retrieve("retrieval timing warm-up", 1)

        for top_k in DEFAULT_TOP_K:
            records, metrics = evaluate_index(retriever, questions, top_k)
            row = {
                "chunk_size": chunk_size,
                "overlap": overlap,
                "top_k": top_k,
                **metrics,
                "chunk_count": chunk_count,
                "index_size_bytes": index_size_bytes,
                "embedding_model": args.model,
                "encoding": args.encoding,
            }
            rows.append(row)
            artifact = {
                "run": {
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "questions_path": str(args.questions.resolve()),
                    "chunks_dir": str(chunks_dir.resolve()),
                    "index_dir": str(index_dir.resolve()),
                    "chunk_size": chunk_size,
                    "overlap": overlap,
                    "top_k": top_k,
                    "embedding": index_config,
                    "generator": "none",
                },
                "summary": row,
                "results": records,
            }
            detail_path = config_dir / f"top_k_{top_k}.json"
            detail_path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
            print(
                f"{config_id} top_k={top_k}: "
                f"hit={metrics['document_hit_at_k_rate']:.3f} "
                f"multi_all={metrics['multi_document_all_expected_documents_present_rate']:.3f} "
                f"latency={metrics['mean_retrieval_latency_seconds']:.4f}s"
            )

    ranked = sorted(rows, key=ranking_key)
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank

    json_path = args.output_dir / "comparison.json"
    csv_path = args.output_dir / "comparison.csv"
    json_path.write_text(
        json.dumps({"ranking_method": "hit rate, multi-doc completeness, recall, latency", "results": ranked}, indent=2) + "\n",
        encoding="utf-8",
    )
    with csv_path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=list(ranked[0]))
        writer.writeheader()
        writer.writerows(ranked)

    print("\nRanked summary")
    for row in ranked:
        print(
            f"{row['rank']:>2}. {row['chunk_size']}/{row['overlap']} k={row['top_k']} "
            f"hit={row['document_hit_at_k_rate']:.3f} "
            f"multi_all={row['multi_document_all_expected_documents_present_rate']:.3f} "
            f"recall={row['mean_expected_document_recall_at_k']:.3f} "
            f"latency={row['mean_retrieval_latency_seconds']:.4f}s "
            f"chunks={row['chunk_count']} index={row['index_size_bytes']}B"
        )
    print(f"\nSaved {csv_path.resolve()}")
    print(f"Saved {json_path.resolve()}")
    return ranked


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--encoding", default=DEFAULT_ENCODING)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Reuse matching experiment indexes already present in the output directory",
    )
    return parser.parse_args()


if __name__ == "__main__":
    try:
        run(parse_args())
    except KeyboardInterrupt:
        sys.exit(130)
