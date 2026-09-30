"""Answer an Amazon question from retrieved evidence using local Ollama."""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from generate import build_prompt, print_sources
from retrieve import DEFAULT_INDEX_DIR, retrieve

DEFAULT_MODEL = "qwen2.5:3b"
DEFAULT_OLLAMA_URL = "http://localhost:11434"


def generate_answer(
    prompt: str,
    model: str = DEFAULT_MODEL,
    ollama_url: str = DEFAULT_OLLAMA_URL,
) -> tuple[str, dict[str, Any]]:
    """Call Ollama's local HTTP API and return the answer plus run metadata."""
    endpoint = f"{ollama_url.rstrip('/')}/api/generate"
    request_data = json.dumps(
        {"model": model, "prompt": prompt, "stream": False}
    ).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=request_data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        if error.code == 404:
            raise RuntimeError(
                f"Ollama could not find model '{model}'. Run: ollama pull {model}"
            ) from error
        raise RuntimeError(f"Ollama returned HTTP {error.code}: {detail}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(
            "Could not connect to Ollama. Install Ollama, start it with "
            "'ollama serve', then pull a model with "
            f"'ollama pull {model}'."
        ) from error

    answer = str(payload.get("response", "")).strip()
    if not answer:
        raise RuntimeError("Ollama returned an empty response")

    metrics = {
        "model": payload.get("model", model),
        "latency_seconds": time.perf_counter() - started,
        "input_tokens": payload.get("prompt_eval_count"),
        "output_tokens": payload.get("eval_count"),
    }
    return answer, metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", help="Question to answer")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--model", default=os.getenv("OLLAMA_MODEL", DEFAULT_MODEL))
    parser.add_argument(
        "--ollama-url",
        default=os.getenv("OLLAMA_URL", DEFAULT_OLLAMA_URL),
        help="Ollama server URL (default: %(default)s)",
    )
    parser.add_argument(
        "--show-prompt",
        action="store_true",
        help="Print retrieval results and the prompt without calling Ollama",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.top_k <= 0:
        raise SystemExit("--top-k must be positive")

    results = retrieve(args.question, args.index_dir, args.top_k)
    prompt = build_prompt(args.question, results)
    print_sources(results)

    if args.show_prompt:
        print("Prompt (Ollama call skipped)\n----------------------------")
        print(prompt)
        return

    answer, metrics = generate_answer(prompt, args.model, args.ollama_url)
    print("Answer\n------")
    print(answer)
    print("\nRun metrics\n-----------")
    print(f"Model: {metrics['model']}")
    print(f"Latency: {metrics['latency_seconds']:.2f}s")
    if metrics["input_tokens"] is not None:
        print(
            f"Tokens: {metrics['input_tokens']} input + "
            f"{metrics['output_tokens']} output"
        )


if __name__ == "__main__":
    main()
