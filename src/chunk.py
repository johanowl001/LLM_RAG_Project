"""Split processed documents into overlapping, fixed-token chunks."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import tiktoken


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_OUTPUT_DIR = DEFAULT_PROCESSED_DIR / "chunks"
DEFAULT_CHUNK_SIZE = 500
DEFAULT_OVERLAP = 75
DEFAULT_ENCODING = "cl100k_base"


def load_documents(processed_dir: Path) -> list[tuple[dict[str, Any], str]]:
    """Load successful documents listed in the ingestion manifest."""
    manifest_path = processed_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    documents: list[tuple[dict[str, Any], str]] = []

    for metadata in manifest:
        if metadata.get("status") == "error" or not metadata.get("text_file"):
            continue
        text_path = processed_dir / metadata["text_file"]
        documents.append((metadata, text_path.read_text(encoding="utf-8")))

    return documents


def chunk_document(
    metadata: dict[str, Any],
    text: str,
    *,
    chunk_size: int,
    overlap: int,
    encoding: tiktoken.Encoding,
) -> list[dict[str, Any]]:
    """Return token-exact chunks and inherited document metadata."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    document_id = Path(metadata["text_file"]).stem
    token_ids = encoding.encode(text)
    step = chunk_size - overlap
    chunks: list[dict[str, Any]] = []

    for chunk_index, start in enumerate(range(0, len(token_ids), step)):
        chunk_tokens = token_ids[start : start + chunk_size]
        if not chunk_tokens:
            break

        chunks.append(
            {
                "chunk_id": f"{document_id}_{chunk_index:04d}",
                "document_id": document_id,
                "source_filename": metadata.get("source_file"),
                "text_file": metadata["text_file"],
                "chunk_index": chunk_index,
                "token_count": len(chunk_tokens),
                "token_start": start,
                "token_end": start + len(chunk_tokens),
                "company": metadata.get("company"),
                "ticker": metadata.get("ticker"),
                "year": metadata.get("year"),
                "quarter": metadata.get("quarter"),
                "document_type": metadata.get("document_type"),
                # A token boundary can fall inside a multi-byte Unicode character.
                # Ignore only that incomplete boundary fragment instead of emitting �.
                "text": encoding.decode(chunk_tokens, errors="ignore"),
            }
        )

        if start + chunk_size >= len(token_ids):
            break

    return chunks


def chunk_corpus(
    processed_dir: Path,
    output_dir: Path,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    encoding_name: str = DEFAULT_ENCODING,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Chunk the processed corpus and write JSONL plus a run summary."""
    encoding = tiktoken.get_encoding(encoding_name)
    all_chunks: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()

    for metadata, text in load_documents(processed_dir):
        chunks = chunk_document(
            metadata,
            text,
            chunk_size=chunk_size,
            overlap=overlap,
            encoding=encoding,
        )
        all_chunks.extend(chunks)
        counts[Path(metadata["text_file"]).stem] = len(chunks)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "chunks.jsonl"
    with output_path.open("w", encoding="utf-8") as output_file:
        for chunk in all_chunks:
            output_file.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    summary = {
        "encoding": encoding_name,
        "chunk_size": chunk_size,
        "overlap": overlap,
        "document_count": len(counts),
        "total_chunks": len(all_chunks),
        "chunks_per_document": dict(counts),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return all_chunks, dict(counts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP)
    parser.add_argument("--encoding", default=DEFAULT_ENCODING)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    chunks, counts = chunk_corpus(
        args.processed_dir,
        args.output_dir,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        encoding_name=args.encoding,
    )
    for document_id, count in counts.items():
        print(f"{document_id}: {count} chunks")
    print(f"Finished: {len(chunks)} chunks from {len(counts)} documents")


if __name__ == "__main__":
    main()
