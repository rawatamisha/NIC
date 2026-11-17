"""
Structured document summarization using Sarvam chat completions.

The summary is validated against the source text before saving so the pipeline
does not silently persist fabricated facts.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError
from sarvamai import SarvamAI
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from config import SARVAM_API_TIMEOUT_SECONDS, SARVAM_MAX_RETRIES, SARVAM_SUMMARY_MODEL
from src.ingestion.language_detect import detect_language
from src.utils.logger import get_logger

logger = get_logger(__name__)
load_dotenv()

_API_KEY = os.getenv("SARVAM_API_KEY")
if not _API_KEY:
    raise EnvironmentError("SARVAM_API_KEY not set. Copy .env.example to .env and add your key.")

_client = SarvamAI(api_subscription_key=_API_KEY)


class SummaryTransientError(Exception):
    """Retryable summarization failure."""


class DocumentSummary(BaseModel):
    """Structured summary output saved to disk and consumed by the UI."""

    source_file: str
    language: str
    purpose: str = "not found in document"
    key_decisions: list[str] = Field(default_factory=list)
    dates: list[str] = Field(default_factory=list)
    amounts: list[str] = Field(default_factory=list)
    responsible_departments: list[str] = Field(default_factory=list)
    action_items: list[str] = Field(default_factory=list)
    not_found_fields: list[str] = Field(default_factory=list)

    def to_markdown(self) -> str:
        """Serializes the summary into a Markdown file with embedded JSON."""
        body = json.dumps(self.model_dump(), indent=2, ensure_ascii=False)
        return f"# Summary: {self.source_file}\n\n```json\n{body}\n```\n"


def _normalize_for_search(value: str) -> str:
    return re.sub(r"[^\w\u0900-\u097F]+", "", value.casefold())


def _contains_fact(source_text: str, candidate: str) -> bool:
    normalized_candidate = _normalize_for_search(candidate)
    if not normalized_candidate:
        return False
    normalized_source = _normalize_for_search(source_text)
    return normalized_candidate in normalized_source


def _extract_json_payload(text: str) -> str:
    fenced = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1)

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]

    raise ValueError("Summary response did not contain JSON")


def _response_to_text(response: Any) -> str:
    if isinstance(response, str):
        return response
    if isinstance(response, dict):
        if "choices" in response:
            choice = response["choices"][0]
            if isinstance(choice, dict):
                message = choice.get("message", {})
                if isinstance(message, dict):
                    return str(message.get("content", ""))
            return str(choice)
        if "content" in response:
            return str(response["content"])
    choices = getattr(response, "choices", None)
    if choices:
        first_choice = choices[0]
        message = getattr(first_choice, "message", None)
        if message is not None and hasattr(message, "content"):
            return str(message.content)
        if hasattr(first_choice, "text"):
            return str(first_choice.text)
    if hasattr(response, "content"):
        return str(response.content)
    raise ValueError(f"Unsupported summarization response type: {type(response)!r}")


def _build_summary_prompt(source_text: str, language: str, source_file: str) -> list[dict[str, str]]:
    language_instruction = "Hindi" if language == "hi" else "English"
    system_message = (
        "You extract only facts that appear in the source document. "
        "Never invent facts, numbers, dates, or names. "
        "Output only valid JSON matching the requested schema."
    )
    user_message = f"""
Summarize the following government document in {language_instruction}.

Return a JSON object with exactly these keys:
- source_file
- language
- purpose
- key_decisions
- dates
- amounts
- responsible_departments
- action_items
- not_found_fields

Rules:
- Use only facts present in the document text.
- If a field is not present, set `purpose` to `not found in document` or leave the list empty and add the field name to `not_found_fields`.
- Do not guess, infer, or add any outside knowledge.
- Keep the language of the field values aligned with the document language.

source_file: {source_file}
language: {language}

Document text:
{source_text}
""".strip()
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]


def _validate_summary(summary: DocumentSummary, source_text: str) -> DocumentSummary:
    not_found = set(summary.not_found_fields)

    if not summary.purpose or summary.purpose.strip() == "":
        summary.purpose = "not found in document"
        not_found.add("purpose")
    elif not _contains_fact(source_text, summary.purpose):
        summary.purpose = "not found in document"
        not_found.add("purpose")

    for field_name in ("dates", "amounts"):
        values = list(getattr(summary, field_name))
        verified = [value for value in values if _contains_fact(source_text, value)]
        if len(verified) != len(values):
            not_found.add(field_name)
        setattr(summary, field_name, verified)

    for field_name in ("key_decisions", "responsible_departments", "action_items"):
        values = [value for value in getattr(summary, field_name) if value and value.strip()]
        if not values:
            not_found.add(field_name)
        setattr(summary, field_name, values)

    summary.not_found_fields = sorted(not_found)
    return summary


@retry(
    reraise=True,
    stop=stop_after_attempt(SARVAM_MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=2, max=20),
    retry=retry_if_exception_type(SummaryTransientError),
)
def _run_summary_job(source_text: str, source_file: str, language: str) -> DocumentSummary:
    """Calls Sarvam chat completions and parses the structured summary."""
    try:
        response = _client.chat.completions(
            messages=_build_summary_prompt(source_text[:3000], language, source_file),
            model=SARVAM_SUMMARY_MODEL,
            temperature=0.0,
            max_tokens=4000,
            request_options={"timeout_in_seconds": SARVAM_API_TIMEOUT_SECONDS},
        )
        response_text = _response_to_text(response)

        print("SARVAM FINISH REASON:",
            getattr(response.choices[0], "finish_reason", None))

        print("SARVAM RESPONSE PREVIEW:",
            repr(response_text[:500]))

        payload = json.loads(_extract_json_payload(response_text))
        return DocumentSummary.model_validate(payload)
    except TimeoutError as exc:
        logger.warning("Summary call timed out for %s, will retry: %s", source_file, exc)
        raise SummaryTransientError(str(exc)) from exc
    except (ValueError, json.JSONDecodeError, ValidationError) as exc:
        logger.warning("Malformed summary response for %s, will retry: %s", source_file, exc)
        raise SummaryTransientError(str(exc)) from exc
    except Exception as exc:
        msg = str(exc).lower()
        if "rate limit" in msg or "429" in msg or "503" in msg:
            logger.warning("Transient summary error for %s, will retry: %s", source_file, exc)
            raise SummaryTransientError(str(exc)) from exc
        logger.error("Non-retryable summary error for %s: %s", source_file, exc)
        raise


def summarize_document(
    source_pdf_path: Path,
    processed_text: str,
    language: str | None = None,
) -> DocumentSummary:
    """
    Generates, validates, and saves the structured summary for a source PDF.
    """
    detected_language = language or detect_language(processed_text)
    try:
        summary = _run_summary_job(
            source_text=processed_text,
            source_file=source_pdf_path.name,
            language=detected_language,
        )
    except SummaryTransientError as exc:
        logger.error(
            "Falling back to a safe empty summary for %s after repeated Sarvam failures: %s",
            source_pdf_path.name,
            exc,
        )
        summary = DocumentSummary(
            source_file=source_pdf_path.name,
            language=detected_language,
            purpose="not found in document",
            not_found_fields=[
                "purpose",
                "key_decisions",
                "dates",
                "amounts",
                "responsible_departments",
                "action_items",
            ],
        )

    summary = _validate_summary(summary, processed_text)

    output_path = source_pdf_path.with_name(f"{source_pdf_path.stem}_summary.md")
    output_path.write_text(summary.to_markdown(), encoding="utf-8")
    logger.info("Saved summary to %s", output_path)
    return summary
