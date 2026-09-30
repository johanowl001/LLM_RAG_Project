"""Extract text and basic metadata from the project's raw PDF documents."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from pypdf import PdfReader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def infer_metadata(path: Path) -> dict[str, Any]:
    """Infer a small, consistent metadata set from an Amazon PDF filename."""
    name = path.stem
    lowered = name.lower()
    year: int | None = None
    quarter: str | None = None

    full_year = re.search(r"(?:19|20)\d{2}", name)
    if full_year:
        year = int(full_year.group())
    else:
        short_year = re.search(r"(?:^|[_-])(\d{2})(?:[_-]|$)", name)
        if short_year:
            year = 2000 + int(short_year.group(1))

    quarter_match = re.search(r"q([1-4])", lowered)
    if quarter_match:
        quarter = f"Q{quarter_match.group(1)}"

    # Slide filenames compact quarter and two-digit year, e.g. Q120 = Q1 2020.
    compact_period = re.search(r"q([1-4])(\d{2})(?:\D|$)", lowered)
    if compact_period and year is None:
        quarter = f"Q{compact_period.group(1)}"
        year = 2000 + int(compact_period.group(2))

    if "10q" in lowered:
        document_type = "10-Q"
    elif "annual-report" in lowered:
        document_type = "annual_report"
    elif "shareholder-letter" in lowered:
        document_type = "shareholder_letter"
    elif "earnings-release" in lowered:
        document_type = "earnings_release"
    elif "webslides" in lowered:
        document_type = "earnings_slides"
    elif "business_and_financial_update" in lowered:
        # This downloaded file is Amazon's Q4/full-year 2021 earnings release.
        document_type = "earnings_release"
        year = 2021
        quarter = "Q4"
    else:
        document_type = "unknown"

    return {
        "company": "Amazon.com, Inc.",
        "ticker": "AMZN",
        "year": year,
        "quarter": quarter,
        "document_type": document_type,
    }


def extract_pdf(pdf_path: Path) -> tuple[str, dict[str, Any]]:
    """Return page-delimited text and extraction details for one PDF."""
    reader = PdfReader(pdf_path)
    page_texts: list[str] = []
    empty_pages: list[int] = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            empty_pages.append(page_number)
        page_texts.append(f"--- Page {page_number} ---\n{text}")

    text = "\n\n".join(page_texts).strip() + "\n"
    details = {
        "page_count": len(reader.pages),
        "empty_text_pages": empty_pages,
        "character_count": len(text),
    }
    return text, details


def ingest_file(pdf_path: Path, processed_dir: Path) -> dict[str, Any]:
    """Extract one PDF into a text file and JSON metadata sidecar."""
    text, extraction = extract_pdf(pdf_path)
    metadata = {
        **infer_metadata(pdf_path),
        "source_file": pdf_path.name,
        "text_file": f"{pdf_path.stem}.txt",
        **extraction,
    }

    (processed_dir / metadata["text_file"]).write_text(text, encoding="utf-8")
    (processed_dir / f"{pdf_path.stem}.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


def ingest_directory(raw_dir: Path, processed_dir: Path) -> list[dict[str, Any]]:
    """Ingest every PDF in a directory and write a corpus manifest."""
    pdf_paths = sorted(raw_dir.glob("*.pdf"), key=lambda path: path.name.lower())
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found in {raw_dir}")

    processed_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []

    for index, pdf_path in enumerate(pdf_paths, start=1):
        print(f"[{index}/{len(pdf_paths)}] {pdf_path.name}")
        try:
            manifest.append(ingest_file(pdf_path, processed_dir))
        except Exception as exc:  # Continue so one damaged PDF does not lose the run.
            manifest.append(
                {
                    **infer_metadata(pdf_path),
                    "source_file": pdf_path.name,
                    "status": "error",
                    "error": str(exc),
                }
            )
            print(f"  ERROR: {exc}")

    (processed_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = ingest_directory(args.raw_dir, args.processed_dir)
    succeeded = sum(item.get("status") != "error" for item in manifest)
    failed = len(manifest) - succeeded
    print(f"Finished: {succeeded} succeeded, {failed} failed")


if __name__ == "__main__":
    main()
