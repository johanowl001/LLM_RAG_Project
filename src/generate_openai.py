"""Answer an Amazon question with the winning retriever and OpenAI Responses API."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any

from evaluate_full_rag import (
    DEFAULT_INDEX_DIR,
    DEFAULT_MAX_CHARS_PER_CHUNK,
    DEFAULT_TOP_K,
    build_bounded_prompt,
)
from generate import print_sources
from retrieve import retrieve

DEFAULT_MODEL = "gpt-5-mini"


def generate_answer(prompt: str, model: str = DEFAULT_MODEL) -> tuple[str, dict[str, Any]]:
    """Return response text and usage data without changing the requested model."""
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Export it in your shell before running "
            "the OpenAI generator; never put the key in source code."
        )
    try:
        from openai import (
            APIStatusError,
            AuthenticationError,
            BadRequestError,
            NotFoundError,
            OpenAI,
            PermissionDeniedError,
        )
    except ImportError as error:
        raise RuntimeError(
            "The official OpenAI Python SDK is missing. Run: "
            "python3 -m pip install -r requirements.txt"
        ) from error

    started = time.perf_counter()
    try:
        response = OpenAI().responses.create(model=model, input=prompt)
    except AuthenticationError as error:
        raise RuntimeError(
            "OpenAI rejected OPENAI_API_KEY. Check that the key is valid and belongs "
            "to the intended project."
        ) from error
    except (NotFoundError, PermissionDeniedError) as error:
        raise RuntimeError(
            f"The OpenAI API account cannot access model '{model}'. Choose a model "
            "available to this API project with --model or OPENAI_MODEL. No fallback "
            "model was substituted."
        ) from error
    except BadRequestError as error:
        message = str(error)
        if "model" in message.lower():
            raise RuntimeError(
                f"OpenAI could not use requested model '{model}': {message}. "
                "No fallback model was substituted."
            ) from error
        raise RuntimeError(f"OpenAI rejected the request: {message}") from error
    except APIStatusError as error:
        raise RuntimeError(
            f"OpenAI API request failed with status {error.status_code}: {error}"
        ) from error

    answer = response.output_text.strip()
    if not answer:
        raise RuntimeError("OpenAI returned no answer text")
    usage = getattr(response, "usage", None)
    metrics = {
        "model": model,
        "latency_seconds": time.perf_counter() - started,
        "input_tokens": getattr(usage, "input_tokens", None),
        "output_tokens": getattr(usage, "output_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }
    return answer, metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", help="Question to answer")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", DEFAULT_MODEL))
    parser.add_argument(
        "--max-chars-per-chunk", type=int, default=DEFAULT_MAX_CHARS_PER_CHUNK
    )
    parser.add_argument("--show-prompt", action="store_true")
    args = parser.parse_args()
    if args.top_k != DEFAULT_TOP_K:
        parser.error("the comparison fixes retrieval at top_k=8")
    if args.max_chars_per_chunk <= 0:
        parser.error("--max-chars-per-chunk must be positive")
    return args


def main() -> None:
    args = parse_args()
    results = retrieve(args.question, args.index_dir, args.top_k)
    prompt = build_bounded_prompt(args.question, results, args.max_chars_per_chunk)
    print_sources(results)
    if args.show_prompt:
        print("Prompt (OpenAI call skipped)\n----------------------------")
        print(prompt)
        return
    answer, metrics = generate_answer(prompt, args.model)
    print("Answer\n------")
    print(answer)
    print("\nRun metrics\n-----------")
    print(f"Model: {metrics['model']}")
    print(f"Latency: {metrics['latency_seconds']:.2f}s")
    if metrics["input_tokens"] is not None:
        total = metrics["total_tokens"]
        suffix = f" = {total} total" if total is not None else ""
        print(
            f"Tokens: {metrics['input_tokens']} input + "
            f"{metrics['output_tokens']} output{suffix}"
        )


if __name__ == "__main__":
    main()
