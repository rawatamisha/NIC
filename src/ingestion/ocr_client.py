"""
Wrapper around the Sarvam Document Intelligence (OCR + layout) API.

Handles: job creation, upload, polling, download, retries on transient failure,
and reassembly of multi-batch documents (long PDFs split by pdf_splitter.py)
back into a single structured Markdown output with correct page numbering.

See ARCHITECTURE.md §2.1, RULES.md §4 (error handling).
"""

import os
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from sarvamai import SarvamAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from config import (
    PROCESSED_DIR,
    SARVAM_API_TIMEOUT_SECONDS,
    SARVAM_MAX_RETRIES,
)
from src.ingestion.pdf_splitter import split_pdf, PdfBatch
from src.utils.logger import get_logger

logger = get_logger(__name__)
load_dotenv()

_API_KEY = os.getenv("SARVAM_API_KEY")
if not _API_KEY:
    raise EnvironmentError(
        "SARVAM_API_KEY not set. Please create a .env file with your Sarvam API key or set the environment variable."
    )

_client = SarvamAI(api_subscription_key=_API_KEY)


@dataclass
class OcrResult:
    """Structured OCR output for one (sub-)PDF, tagged with its original page range."""
    text_markdown: str
    start_page: int
    end_page: int


class OcrTransientError(Exception):
    """Raised for retryable failures (timeouts, rate limits, transient API errors)."""


def _load_ocr_output(download_path: Path, output_format: str) -> str:
    """
    Reads the structured OCR payload returned by Sarvam.

    The SDK may materialize the download as either a directory of files or a
    single archive/file path. This helper handles both forms and returns the
    first markdown-like document found.
    """
    if download_path.is_dir():
        candidates = sorted(download_path.rglob(f"*.{output_format}"))
        if not candidates and output_format == "md":
            candidates = sorted(download_path.rglob("*.markdown"))
        if not candidates:
            raise RuntimeError(f"No {output_format} output found in {download_path}")
        return candidates[0].read_text(encoding="utf-8")

    if download_path.is_file():
        if zipfile.is_zipfile(download_path):
            extract_dir = download_path.parent / f"{download_path.stem}__extracted"
            if extract_dir.exists():
                shutil.rmtree(extract_dir)
            extract_dir.mkdir(parents=True, exist_ok=True)

            with zipfile.ZipFile(download_path) as archive:
                archive.extractall(extract_dir)

            candidates = sorted(extract_dir.rglob(f"*.{output_format}"))
            if not candidates and output_format == "md":
                candidates = sorted(extract_dir.rglob("*.markdown"))
            if not candidates:
                raise RuntimeError(
                    f"No {output_format} output found in extracted archive {download_path}"
                )
            return candidates[0].read_text(encoding="utf-8")

        if download_path.suffix.lstrip(".") == output_format:
            return download_path.read_text(encoding="utf-8")

    raise RuntimeError(f"Unsupported OCR output format at {download_path}")


@retry(
    reraise=True,
    stop=stop_after_attempt(SARVAM_MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=2, max=20),
    retry=retry_if_exception_type(OcrTransientError),
)
def _run_ocr_job(pdf_path: Path, language: str, output_format: str = "md") -> str:
    """
    Runs a single Sarvam Document Intelligence job on one (sub-)PDF.
    Retries automatically on transient failures — see RULES.md §4.

    Args:
        pdf_path: path to a PDF that is within the 10-page/job limit.
        language: BCP-47-style language code, e.g. "hi-IN", "en-IN".
        output_format: "md" or "html" — Markdown is used elsewhere in the pipeline.

    Returns:
        Extracted structured text as a string.

    Raises:
        OcrTransientError: on timeout / rate limit / transient failure (triggers retry).
        RuntimeError: on a non-retryable failure after logging the cause.
    """
    try:
        job = _client.document_intelligence.create_job(
            language=language, output_format=output_format
        )
        job.upload_file(str(pdf_path))
        job.start()
        job.wait_until_complete(timeout=SARVAM_API_TIMEOUT_SECONDS)

        output_path = pdf_path.parent / f"{pdf_path.stem}__ocr_out"
        job.download_output(str(output_path))

        return _load_ocr_output(output_path, output_format)

    except TimeoutError as exc:
        logger.warning("OCR job timed out for %s, will retry: %s", pdf_path.name, exc)
        raise OcrTransientError(str(exc)) from exc
    except Exception as exc:
        # Treat anything with "rate limit" / "429" / "503" in the message as transient.
        msg = str(exc).lower()
        if "rate limit" in msg or "429" in msg or "503" in msg:
            logger.warning("Transient OCR error for %s, will retry: %s", pdf_path.name, exc)
            raise OcrTransientError(str(exc)) from exc
        logger.error("Non-retryable OCR error for %s: %s", pdf_path.name, exc)
        raise


def ocr_document(source_path: Path, language: str = "hi-IN") -> str:
    """
    Runs full OCR + layout parsing on a PDF of any length, transparently
    splitting it into Sarvam's 10-page/job limit and reassembling the result.

    A failure on one batch is logged and skipped (marked in the output) rather
    than crashing the whole document — see RULES.md §4. The final combined
    Markdown includes page-boundary markers so downstream chunking (chunker.py)
    can recover accurate page numbers for citations.

    Args:
        source_path: path to the source PDF (any number of pages).
        language: language hint passed to the Sarvam API.

    Returns:
        Combined structured Markdown text for the whole document, with page
        markers of the form `<!-- page: N -->` inserted between batches.
    """
    split_dir = PROCESSED_DIR / "_split_tmp" / source_path.stem
    batches: list[PdfBatch] = split_pdf(source_path, split_dir)

    combined_parts: list[str] = []

    for batch in batches:
        logger.info(
            "OCR-ing %s pages %d-%d",
            source_path.name, batch.start_page, batch.end_page,
        )
        try:
            text = _run_ocr_job(batch.file_path, language=language)
            combined_parts.append(f"<!-- page: {batch.start_page} -->\n{text}")
        except Exception as exc:
            logger.error(
                "Skipping pages %d-%d of %s after repeated failures: %s",
                batch.start_page, batch.end_page, source_path.name, exc,
            )
            combined_parts.append(
                f"<!-- page: {batch.start_page} -->\n"
                f"[OCR FAILED for pages {batch.start_page}-{batch.end_page} — needs manual review]"
            )

    combined_text = "\n\n".join(combined_parts)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DIR / f"{source_path.stem}.md"
    output_path.write_text(combined_text, encoding="utf-8")
    logger.info("Saved combined OCR output to %s", output_path)

    return combined_text
