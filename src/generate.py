"""Answer an Amazon question using retrieved evidence and a hosted LLM.

Retrieval deliberately happens before generation: FAISS selects relevant document
chunks, then the LLM receives only those chunks plus the question. The LLM is told
not to fill gaps with prior knowledge.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any

from retrieve import DEFAULT_INDEX_DIR, retrieve


DEFAULT_MODEL = "gpt-5-mini"


def source_label(result: dict[str, Any], rank: int) -> str:
    """Create a short, stable citation label for one retrieved chunk."""
    parts = [str(result.get("document_id") or result.get("source_filename") or "source")]
    period = " ".join(
        str(value) for value in (result.get("year"), result.get("quarter")) if value
    )
    if period:
        parts.append(period)
    if result.get("chunk_index") is not None:
        parts.append(f"chunk {result['chunk_index']}")
    return f"S{rank}: " + ", ".join(parts)


def build_prompt(question: str, results: list[dict[str, Any]]) -> str:
    """Build the complete grounded prompt sent to the generative model."""
    if not results:
        raise ValueError("At least one retrieved chunk is required")

    context_blocks = []
    for rank, result in enumerate(results, start=1):
        label = source_label(result, rank)
        context_blocks.append(f"[{label}]\n{result['text'].strip()}")

    context = "\n\n".join(context_blocks)
    return f"""You are answering a question about Amazon financial documents.

Use only the retrieved context below. Do not use prior knowledge, guess, or add
facts that are absent from the context. If the evidence is insufficient, say:
"The retrieved context does not provide enough evidence to answer this question."

Cite factual claims using the exact source IDs [S1], [S2], etc. If sources
disagree or refer to different periods, make that distinction explicit.

Question:
{question.strip()}

Retrieved context:
{context}

Answer:"""


def generate_answer(prompt: str, model: str = DEFAULT_MODEL) -> tuple[str, dict[str, Any]]:
    """Call OpenAI's Responses API and return text plus simple usage/latency data."""
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Export it in your shell; never put the key in source code."
        )
    try:
        from openai import OpenAI
    except ImportError as error:
        raise RuntimeError(
            "OpenAI SDK is missing. Run: python3 -m pip install -r requirements.txt"
        ) from error

    started = time.perf_counter()
    response = OpenAI().responses.create(model=model, input=prompt)
    latency_seconds = time.perf_counter() - started
    usage = getattr(response, "usage", None)
    metrics = {
        "model": model,
        "latency_seconds": latency_seconds,
        "input_tokens": getattr(usage, "input_tokens", None),
        "output_tokens": getattr(usage, "output_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }
    return response.output_text, metrics


def print_sources(results: list[dict[str, Any]], snippet_chars: int = 700) -> None:
    """Show exactly which evidence was supplied to the model."""
    print("\nRetrieved sources")
    print("-----------------")
    for rank, result in enumerate(results, start=1):
        snippet = " ".join(result["text"].split())[:snippet_chars]
        print(f"[{source_label(result, rank)}] score={result['score']:.4f}")
        suffix = "..." if len(result["text"]) > snippet_chars else ""
        print(f"  {snippet}{suffix}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", help="Question to answer")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", DEFAULT_MODEL))
    parser.add_argument(
        "--show-prompt",
        action="store_true",
        help="Print retrieval results and the prompt without calling the LLM",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.top_k <= 0:
        raise SystemExit("--top-k must be positive")

    # Step 1: retrieve evidence. Step 2: generate only from that evidence.
    results = retrieve(args.question, args.index_dir, args.top_k)
    prompt = build_prompt(args.question, results)
    print_sources(results)

    if args.show_prompt:
        print("Prompt (LLM call skipped)\n-------------------------")
        print(prompt)
        return

    answer, metrics = generate_answer(prompt, args.model)
    print("Answer\n------")
    print(answer)
    print("\nRun metrics\n-----------")
    print(f"Model: {metrics['model']}")
    print(f"Latency: {metrics['latency_seconds']:.2f}s")
    if metrics["total_tokens"] is not None:
        print(
            f"Tokens: {metrics['input_tokens']} input + "
            f"{metrics['output_tokens']} output = {metrics['total_tokens']} total"
        )
    print("Cost: see the OpenAI usage dashboard (pricing varies by model).")


if __name__ == "__main__":
    main()
