"""
Minimal tests for the ingestion layer. These should not
require a live Sarvam API call — mock the network boundary instead.
"""

from pathlib import Path
import zipfile

import fitz  # PyMuPDF
import pytest

from src.ingestion.ocr_client import _load_ocr_output
from src.ingestion.pdf_splitter import split_pdf


def _make_dummy_pdf(path: Path, num_pages: int) -> None:
    doc = fitz.open()
    for i in range(num_pages):
        page = doc.new_page()
        page.insert_text((72, 72), f"Test page {i + 1}")
    doc.save(path)
    doc.close()


def test_split_pdf_single_batch_when_under_limit(tmp_path: Path):
    pdf_path = tmp_path / "short_doc.pdf"
    _make_dummy_pdf(pdf_path, num_pages=5)

    batches = split_pdf(pdf_path, tmp_path / "out")

    assert len(batches) == 1
    assert batches[0].start_page == 1
    assert batches[0].end_page == 5


def test_split_pdf_multiple_batches_when_over_limit(tmp_path: Path):
    pdf_path = tmp_path / "long_doc.pdf"
    _make_dummy_pdf(pdf_path, num_pages=23)  # over the 10-page/job limit

    batches = split_pdf(pdf_path, tmp_path / "out")

    assert len(batches) == 3  # 10 + 10 + 3
    assert batches[0].start_page == 1 and batches[0].end_page == 10
    assert batches[1].start_page == 11 and batches[1].end_page == 20
    assert batches[2].start_page == 21 and batches[2].end_page == 23


def test_split_pdf_raises_on_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        split_pdf(tmp_path / "does_not_exist.pdf", tmp_path / "out")


def test_load_ocr_output_reads_markdown_from_zip_archive(tmp_path: Path):
    archive_path = tmp_path / "sample__ocr_out"
    with zipfile.ZipFile(archive_path, mode="w") as archive:
        archive.writestr("document.md", "# OCR text\n\npage 1")

    text = _load_ocr_output(archive_path, "md")

    assert "OCR text" in text


def test_load_ocr_output_reads_markdown_from_directory(tmp_path: Path):
    output_dir = tmp_path / "sample__ocr_out"
    output_dir.mkdir()
    (output_dir / "document.md").write_text("# OCR text\n\npage 1", encoding="utf-8")

    text = _load_ocr_output(output_dir, "md")

    assert "OCR text" in text
