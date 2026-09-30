"""Evaluate full RAG while keeping retrieval and generation metrics separate.

The benchmark's expected_documents remain the sole source of deterministic
retrieval scores. Answer-quality scores use reference answers when supplied and
can optionally be augmented by an explicitly selected LLM judge.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from evaluate import BaselineRetriever, load_json, retrieval_metrics, validate_questions
from generate import build_prompt, source_label

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUESTIONS = PROJECT_ROOT / "evals" / "questions.json"
DEFAULT_INDEX_DIR = (
    PROJECT_ROOT
    / "evals"
    / "results"
    / "retrieval_experiments"
    / "artifacts"
    / "chunk_800_overlap_100"
    / "vector_index"
)
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "evals" / "results" / "full_rag"
DEFAULT_TOP_K = 8
DEFAULT_MAX_CHARS_PER_CHUNK = 1200
INSUFFICIENT = "the retrieved context does not provide enough evidence"


def compact_chunk(chunk: dict[str, Any], rank: int) -> dict[str, Any]:
    """Keep source metadata in the artifact but avoid duplicating prompt text."""
    return {
        "source_id": f"S{rank}",
        "source_label": source_label(chunk, rank),
        **{key: value for key, value in chunk.items() if key != "text"},
    }


def build_bounded_prompt(
    question: str, chunks: list[dict[str, Any]], max_chars_per_chunk: int
) -> str:
    """Keep all top-k sources while fitting a modest local-model context window."""
    prompt_chunks = []
    for chunk in chunks:
        excerpt = chunk["text"].strip()
        if len(excerpt) > max_chars_per_chunk:
            excerpt = excerpt[:max_chars_per_chunk].rsplit(" ", 1)[0] + " ..."
        prompt_chunks.append({**chunk, "text": excerpt})
    return build_prompt(question, prompt_chunks)


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text.lower())


def token_f1(answer: str, reference: str) -> float:
    answer_tokens, reference_tokens = tokenize(answer), tokenize(reference)
    if not answer_tokens or not reference_tokens:
        return float(answer_tokens == reference_tokens)
    overlap = sum((Counter(answer_tokens) & Counter(reference_tokens)).values())
    precision = overlap / len(answer_tokens)
    recall = overlap / len(reference_tokens)
    return 2 * precision * recall / (precision + recall) if overlap else 0.0


def token_recall(answer: str, reference: str) -> float:
    answer_tokens, reference_tokens = tokenize(answer), tokenize(reference)
    if not reference_tokens:
        return 1.0
    return sum((Counter(answer_tokens) & Counter(reference_tokens)).values()) / len(reference_tokens)


def deterministic_answer_scores(
    answer: str, item: dict[str, Any], source_count: int
) -> dict[str, Any]:
    """Return transparent heuristics; never use these as retrieval metrics."""
    cited_numbers = [int(value) for value in re.findall(r"\[S(\d+)\]", answer)]
    valid = sorted({number for number in cited_numbers if 1 <= number <= source_count})
    invalid = sorted({number for number in cited_numbers if number < 1 or number > source_count})
    factual_sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", answer)
        if re.search(r"\d|\$|%", sentence)
    ]
    supported_factual = [
        sentence for sentence in factual_sentences if re.search(r"\[S\d+\]", sentence)
    ]
    citation_coverage = (
        len(supported_factual) / len(factual_sentences)
        if factual_sentences
        else (1.0 if valid else 0.0)
    )
    citation_score = citation_coverage if not invalid else 0.0
    reference = item.get("reference_answer")
    required_points = item.get("required_points") or []
    if reference:
        correctness = {
            "status": "scored",
            "score_0_to_1": token_f1(answer, reference),
            "method": "reference_answer_token_f1",
        }
    else:
        correctness = {
            "status": "not_scored",
            "score_0_to_1": None,
            "method": "no_reference_answer",
        }
    if required_points:
        matches = [point for point in required_points if point.lower() in answer.lower()]
        completeness = {
            "status": "scored",
            "score_0_to_1": len(matches) / len(required_points),
            "method": "required_point_exact_match",
            "matched_points": matches,
            "required_points": required_points,
        }
    elif reference:
        completeness = {
            "status": "scored",
            "score_0_to_1": token_recall(answer, reference),
            "method": "reference_answer_token_recall",
        }
    else:
        completeness = {
            "status": "not_scored",
            "score_0_to_1": None,
            "method": "no_required_points",
        }
    return {
        "correctness": correctness,
        "groundedness": {
            "status": "heuristic",
            "score_0_to_1": citation_score,
            "method": "numeric_claim_valid_citation_coverage",
            "note": "Citation presence is auditable but does not prove entailment; use --judge for semantic scoring.",
        },
        "citation_source_support": {
            "status": "heuristic",
            "score_0_to_1": citation_score,
            "method": "valid_source_ids_and_numeric_claim_coverage",
            "valid_source_ids": [f"S{number}" for number in valid],
            "invalid_source_ids": [f"S{number}" for number in invalid],
            "factual_sentence_count": len(factual_sentences),
            "cited_factual_sentence_count": len(supported_factual),
        },
        "completeness": completeness,
        "refused_for_insufficient_context": INSUFFICIENT in answer.lower(),
    }


def parse_json_object(text: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("judge returned no JSON object")
    value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("judge result must be a JSON object")
    for component in ("correctness", "groundedness", "citation_source_support", "completeness"):
        score = value.get(component, {}).get("score")
        if not isinstance(score, (int, float)) or not 1 <= score <= 5:
            raise ValueError(f"judge {component} score must be between 1 and 5")
    return value


def judge_prompt(item: dict[str, Any], answer: str, chunks: list[dict[str, Any]]) -> str:
    context = "\n\n".join(
        f"[{source_label(chunk, rank)}]\n{chunk['text']}"
        for rank, chunk in enumerate(chunks, start=1)
    )
    reference = item.get("reference_answer") or "No complete reference answer is available."
    return f"""You are grading a RAG answer. Return JSON only.

Use four independent 1-5 scores:
- correctness: factual agreement with the reference and evidence (1 wrong, 5 fully correct)
- groundedness: claims are entailed by retrieved context (1 unsupported, 5 fully supported)
- citation_source_support: citations point to sources supporting their claims (1 absent/wrong, 5 precise)
- completeness: all parts of the question/answer notes are addressed (1 missing, 5 complete)

For each component return {{"score": integer, "reason": short string}}. Also return
"overall_score_0_to_1" as the mean of the four scores mapped linearly from 1..5 to 0..1.

Question: {item['question']}
Answer notes: {item.get('answer_notes', '')}
Reference answer: {reference}
Candidate answer: {answer}

Retrieved context:
{context}
"""


def backend_generator(
    backend: str, model: str | None, ollama_url: str
) -> tuple[Callable[[str], tuple[str, dict[str, Any]]], str]:
    if backend == "ollama":
        from generate_ollama import DEFAULT_MODEL, generate_answer

        selected = model or os.getenv("OLLAMA_MODEL", DEFAULT_MODEL)
        return lambda prompt: generate_answer(prompt, selected, ollama_url), selected
    from generate import DEFAULT_MODEL, generate_answer

    selected = model or os.getenv("OPENAI_MODEL", DEFAULT_MODEL)
    return lambda prompt: generate_answer(prompt, selected), selected


def score_value(record: dict[str, Any], component: str) -> float | None:
    judged = record["answer_evaluation"].get("llm_judge")
    if judged and judged.get("status") == "completed":
        return (judged["scores"][component]["score"] - 1) / 4
    return record["answer_evaluation"]["deterministic"][component].get("score_0_to_1")


def mean_available(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return statistics.mean(present) if present else None


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    retrieval = [record["retrieval_metrics"] for record in records]
    completed = [record for record in records if record["generation_metrics"]["status"] == "completed"]
    generation_failed = [record["id"] for record in records if record["generation_metrics"]["status"] != "completed"]
    answer_summary = {
        component: {
            "mean_score_0_to_1": mean_available([score_value(record, component) for record in completed]),
            "scored_count": sum(score_value(record, component) is not None for record in completed),
        }
        for component in ("correctness", "groundedness", "citation_source_support", "completeness")
    }
    successful_answers = {
        record["id"]
        for record in completed
        if score_value(record, "correctness") is not None
        and score_value(record, "correctness") >= 0.75
        and score_value(record, "groundedness") is not None
        and score_value(record, "groundedness") >= 0.75
    }
    return {
        "retrieval_metrics": {
            "question_count": len(records),
            "document_hit_at_k_rate": statistics.mean(item["document_hit_at_k"] for item in retrieval),
            "mean_expected_document_recall_at_k": statistics.mean(item["expected_document_recall_at_k"] for item in retrieval),
            "all_expected_documents_present_rate": statistics.mean(item["all_expected_documents_present"] for item in retrieval),
            "mean_latency_seconds": statistics.mean(item["latency_seconds"] for item in retrieval),
        },
        "generation_metrics": {
            "requested_count": len(records),
            "completed_count": len(completed),
            "failed_ids": generation_failed,
            "mean_latency_seconds": mean_available([record["generation_metrics"].get("latency_seconds") for record in completed]),
            "total_input_tokens": sum(record["generation_metrics"].get("input_tokens") or 0 for record in completed),
            "total_output_tokens": sum(record["generation_metrics"].get("output_tokens") or 0 for record in completed),
        },
        "answer_quality_metrics": answer_summary,
        "diagnostic_cases": {
            "retrieval_succeeded_generation_failed": [
                record["id"] for record in records
                if record["retrieval_metrics"]["document_hit_at_k"]
                and record["id"] not in successful_answers
            ][:5],
            "retrieval_failed_generation_succeeded": [
                record["id"] for record in records
                if not record["retrieval_metrics"]["document_hit_at_k"]
                and record["id"] in successful_answers
            ][:5],
            "note": "Answer success requires correctness and groundedness >= 0.75; unscored answers are conservatively not classified as successful.",
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    questions = validate_questions(load_json(args.questions))
    if args.limit:
        questions = questions[: args.limit]
    retriever = BaselineRetriever(args.index_dir)
    generate, selected_model = backend_generator(args.generator, args.model, args.ollama_url)
    judge = None
    judge_model = None
    if args.judge != "none":
        judge, judge_model = backend_generator(args.judge, args.judge_model, args.ollama_url)

    records = []
    for number, item in enumerate(questions, start=1):
        started = time.perf_counter()
        query = retriever.config.get("query_prefix", "") + item["question"]
        vector = retriever.model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        ).astype("float32")
        scores, positions = retriever.index.search(vector, min(args.top_k, retriever.index.ntotal))
        chunks = [
            {"score": float(score), **retriever.metadata[int(position)]}
            for score, position in zip(scores[0], positions[0]) if position >= 0
        ]
        retrieval = retrieval_metrics(item["expected_documents"], chunks)
        retrieval["latency_seconds"] = time.perf_counter() - started
        record: dict[str, Any] = {
            **item,
            "retrieved_sources": [compact_chunk(chunk, rank) for rank, chunk in enumerate(chunks, 1)],
            "retrieval_metrics": retrieval,
            "answer": None,
            "generation_metrics": {"status": "pending", "model": selected_model},
            "answer_evaluation": {"deterministic": None, "llm_judge": None},
        }
        try:
            answer, metrics = generate(
                build_bounded_prompt(item["question"], chunks, args.max_chars_per_chunk)
            )
            record["answer"] = answer
            record["generation_metrics"] = {"status": "completed", **metrics}
            record["answer_evaluation"]["deterministic"] = deterministic_answer_scores(
                answer, item, len(chunks)
            )
            if judge:
                try:
                    raw, judge_metrics = judge(judge_prompt(item, answer, chunks))
                    record["answer_evaluation"]["llm_judge"] = {
                        "status": "completed",
                        "model": judge_model,
                        "rubric": "four independent 1-5 component scores",
                        "scores": parse_json_object(raw),
                        "usage": judge_metrics,
                    }
                except Exception as error:
                    record["answer_evaluation"]["llm_judge"] = {
                        "status": "error", "model": judge_model,
                        "error": f"{type(error).__name__}: {error}",
                    }
                    if args.fail_fast:
                        raise
        except Exception as error:
            record["generation_metrics"] = {
                "status": "error", "model": selected_model,
                "error": f"{type(error).__name__}: {error}",
            }
            if args.fail_fast:
                raise
        records.append(record)
        print(
            f"[{number}/{len(questions)}] {item['id']}: "
            f"retrieval_hit={retrieval['document_hit_at_k']} "
            f"generation={record['generation_metrics']['status']}"
        )

    artifact = {
        "run": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "questions_path": str(args.questions.resolve()),
            "index_dir": str(args.index_dir.resolve()),
            "retrieval_configuration": {"chunk_size": 800, "overlap": 100, "top_k": args.top_k},
            "generation_context": {
                "source_count": args.top_k,
                "max_chars_per_chunk": args.max_chars_per_chunk,
                "reason": "preserve all top-k sources within the local model context window",
            },
            "embedding": retriever.config,
            "generator": args.generator,
            "generation_model": selected_model,
            "judge": args.judge,
            "judge_model": judge_model,
        },
        "summary": {},
        "results": records,
    }
    artifact["summary"] = aggregate(records)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = args.output or args.results_dir / f"{args.generator}_{selected_model.replace(':', '-')}_{timestamp}.json"
    output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    (args.results_dir / "latest_summary.json").write_text(
        json.dumps({"result_file": str(output.resolve()), **artifact["summary"]}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\nSaved: {output.resolve()}")
    print(json.dumps(artifact["summary"], indent=2))
    return artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--max-chars-per-chunk", type=int, default=DEFAULT_MAX_CHARS_PER_CHUNK)
    parser.add_argument("--generator", choices=("ollama", "openai"), default="ollama")
    parser.add_argument("--model")
    parser.add_argument("--judge", choices=("none", "ollama", "openai"), default="none")
    parser.add_argument("--judge-model")
    parser.add_argument("--ollama-url", default=os.getenv("OLLAMA_URL", "http://localhost:11434"))
    parser.add_argument("--limit", type=int, help="Run only the first N questions (smoke testing)")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()
    if args.top_k != DEFAULT_TOP_K:
        parser.error("this stage fixes retrieval at top_k=8; omit --top-k or pass 8")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.max_chars_per_chunk <= 0:
        parser.error("--max-chars-per-chunk must be positive")
    return args


if __name__ == "__main__":
    try:
        run(parse_args())
    except KeyboardInterrupt:
        sys.exit(130)
