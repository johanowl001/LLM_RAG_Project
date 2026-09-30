"""Build a persistent FAISS index from the saved Amazon document chunks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "chunks" / "chunks.jsonl"
DEFAULT_INDEX_DIR = PROJECT_ROOT / "data" / "processed" / "vector_index"
DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def load_chunks(path: Path) -> list[dict[str, Any]]:
    """Load non-empty chunk records from JSON Lines."""
    chunks: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            chunk = json.loads(line)
            if not chunk.get("text", "").strip():
                raise ValueError(f"Chunk on line {line_number} has no text")
            chunks.append(chunk)
    if not chunks:
        raise ValueError(f"No chunks found in {path}")
    return chunks


def build_index(
    chunks_path: Path,
    index_dir: Path,
    *,
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 32,
) -> dict[str, Any]:
    """Embed all chunks and persist a cosine-similarity FAISS index.

    An embedding model converts text into a fixed-length numeric vector used for
    similarity search. It does not write an answer. A generative LLM instead
    predicts output tokens to produce new text; that later stage can consume the
    evidence returned by this index.
    """
    chunks = load_chunks(chunks_path)
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        [chunk["text"] for chunk in chunks],
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    index_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_dir / "chunks.faiss"))
    with (index_dir / "metadata.jsonl").open("w", encoding="utf-8") as target:
        for chunk in chunks:
            target.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    config = {
        "model_name": model_name,
        "metric": "cosine_similarity",
        "normalized_embeddings": True,
        "chunk_count": len(chunks),
        "embedding_dimension": int(embeddings.shape[1]),
        "query_prefix": DEFAULT_QUERY_PREFIX,
        "source_chunks": str(chunks_path.resolve()),
    }
    (index_dir / "config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS_PATH)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = build_index(
        args.chunks,
        args.index_dir,
        model_name=args.model,
        batch_size=args.batch_size,
    )
    print(
        f"Indexed {config['chunk_count']} chunks as "
        f"{config['embedding_dimension']}-dimensional vectors using "
        f"{config['model_name']}"
    )
    print(f"Saved index and metadata to {args.index_dir.resolve()}")


if __name__ == "__main__":
    main()
