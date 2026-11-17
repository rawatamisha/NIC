"""
Scans data/raw_pdfs/ for PDFs, runs each through OCR (src/ingestion/ocr_client.py),
and writes structured output to data/processed/.

- Skips documents that already have processed output, unless --force is passed
  (cost awareness — avoid re-billing OCR pages unnecessarily).
- A failure on one document is logged and does NOT stop the batch.
- Supports --limit for cheap testing on a small sample before a full run

Usage:
    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --limit 2
    python scripts/run_pipeline.py --force
    python scripts/run_pipeline.py --language en-IN
"""

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import PROCESSED_DIR, RAW_PDF_DIR 
from src.ingestion.ocr_client import ocr_document  
from src.ingestion.language_detect import detect_language
from src.processing.chunker import chunk_document 
from src.processing.summarizer import summarize_document 
from src.retrieval.vector_store import DocumentVectorStore 
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PipelineSummary:
    """Tracks batch-level results so the run ends with a clear pass/fail report."""
    processed: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    summarized: list[str] = field(default_factory=list)
    indexed: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list) 

    def report(self) -> str:
        lines = [
            "",
            "=" * 60,
            "PIPELINE RUN SUMMARY",
            "=" * 60,
            f"Processed: {len(self.processed)}",
            f"Skipped (already done): {len(self.skipped)}",
            f"Summarized: {len(self.summarized)}",
            f"Indexed: {len(self.indexed)}",
            f"Failed: {len(self.failed)}",
        ]
        if self.failed:
            lines.append("\nFailed documents:")
            lines.extend(f"  - {name}: {err}" for name, err in self.failed)
        lines.append("=" * 60)
        return "\n".join(lines)


def discover_pdfs(root: Path) -> list[Path]:
    """
    Recursively finds all PDFs under `root`, since ministry documents are
    typically organized in subfolders by department/date (see ARCHITECTURE.md).
    """
    if not root.exists():
        logger.warning("Input directory %s does not exist yet — creating it.", root)
        root.mkdir(parents=True, exist_ok=True)
        return []

    pdfs = sorted(root.rglob("*.pdf"))
    logger.info("Discovered %d PDF(s) under %s", len(pdfs), root)
    return pdfs


def already_processed(pdf_path: Path) -> bool:
    """
    Checks whether this PDF already has OCR output, so re-runs don't re-bill
    Sarvam API pages unnecessarily (RULES.md §6 — cost awareness).
    """
    expected_output = PROCESSED_DIR / f"{pdf_path.stem}.md"
    return expected_output.exists()


def processed_output_path(pdf_path: Path) -> Path:
    """Returns the OCR markdown path for a source PDF."""
    return PROCESSED_DIR / f"{pdf_path.stem}.md"


def summary_output_path(pdf_path: Path) -> Path:
    """Returns the summary markdown path saved next to the source PDF."""
    return pdf_path.with_name(f"{pdf_path.stem}_summary.md")


def should_run_summary(force: bool, summary_path: Path) -> bool:
    """True when a summary should be regenerated from already-processed OCR text."""
    return force or not summary_path.exists()


def load_processed_text(pdf_path: Path) -> str:
    """Loads existing OCR output for a PDF, or raises if it is missing."""
    path = processed_output_path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"Processed OCR output not found for {pdf_path.name}")
    return path.read_text(encoding="utf-8")


def run(input_dir: Path, language: str, limit: int | None, force: bool, summaries_only: bool = False) -> PipelineSummary:
    """
    Runs the Phase 1 ingestion pipeline over every PDF in `input_dir`.

    Args:
        input_dir: directory to scan recursively for PDFs.
        language: language hint passed to the Sarvam OCR API for every document
                  (a per-document auto-detect can replace this in a later phase).
        limit: if set, only process the first N discovered PDFs — useful for
               cheap testing before spending credits on a full batch.
        force: if True, re-process documents even if output already exists.

    Returns:
        PipelineSummary with counts and any per-document errors.
    """
    summary = PipelineSummary()

    pdfs = discover_pdfs(input_dir)
    if limit is not None:
        pdfs = pdfs[:limit]
        logger.info("Limiting run to first %d document(s) (--limit)", limit)

    if not pdfs:
        logger.info("No PDFs found in %s — nothing to do.", input_dir)
        return summary

    with DocumentVectorStore() as vector_store:
        for i, pdf_path in enumerate(pdfs, start=1):
            logger.info("[%d/%d] %s", i, len(pdfs), pdf_path.name)

            start_time = time.monotonic()
            try:
                processed_path = processed_output_path(pdf_path)
                if summaries_only:
                    logger.info("  -> summaries-only mode: reusing existing OCR output")
                    processed_text = load_processed_text(pdf_path)
                    summary.skipped.append(pdf_path.name)
                elif force or not processed_path.exists():
                    processed_text = ocr_document(pdf_path, language=language)
                    summary.processed.append(pdf_path.name)
                else:
                    logger.info("  -> OCR already done, reusing existing processed text")
                    processed_text = load_processed_text(pdf_path)
                    summary.skipped.append(pdf_path.name)

                detected_language = detect_language(processed_text)

                if not summaries_only:
                    chunks = chunk_document(
                        text=processed_text,
                        source_file=pdf_path.name,
                        language=detected_language,
                    )
                    if chunks:
                        vector_store.add_chunks(chunks)
                        summary.indexed.append(pdf_path.name)

                summary_path = summary_output_path(pdf_path)
                if should_run_summary(force=force, summary_path=summary_path):
                    summarize_document(
                        source_pdf_path=pdf_path,
                        processed_text=processed_text,
                        language=detected_language,
                    )
                    summary.summarized.append(pdf_path.name)
                else:
                    logger.info("  -> summary already exists, skipping LLM call")

                elapsed = time.monotonic() - start_time
                logger.info("  -> phase 2 done in %.1fs", elapsed)
            except Exception as exc:
                # Per RULES.md §4: never let one document's failure stop the batch.
                logger.error("  -> FAILED: %s", exc)
                summary.failed.append((pdf_path.name, str(exc)))
                continue

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 1: OCR ingestion pipeline for government PDFs."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=RAW_PDF_DIR,
        help=f"Directory to scan for PDFs (default: {RAW_PDF_DIR})",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en-IN",
        help="Language hint passed to the Sarvam OCR API (default: en-IN)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N documents — use this for a cheap test run first.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-process documents even if output already exists.",
    )
    parser.add_argument(
        "--summaries-only",
        action="store_true",
        help="Skip OCR and only re-run summaries from existing processed text.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logger.info("Starting Phase 1 ingestion pipeline")
    logger.info(
        "Input dir: %s | Language: %s | Limit: %s | Force: %s | Summaries-only: %s",
        args.input_dir,
        args.language,
        args.limit,
        args.force,
        args.summaries_only,
    )

    summary = run(
        input_dir=args.input_dir,
        language=args.language,
        limit=args.limit,
        force=args.force,
        summaries_only=args.summaries_only,
    )

    print(summary.report())

    sys.exit(1 if summary.failed else 0)

if __name__ == "__main__":
    main()
