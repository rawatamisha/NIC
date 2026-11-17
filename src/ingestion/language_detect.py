"""
Lightweight language detection for processed OCR text.
"""

from __future__ import annotations

import re

from langdetect import DetectorFactory, LangDetectException, detect

from src.utils.logger import get_logger

logger = get_logger(__name__)

DetectorFactory.seed = 0

_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")


def detect_language(text: str) -> str:
    """
    Detects the dominant language for a document.

    Returns:
        "hi" for Hindi, "en" for English, or "other" for everything else
        and for short/ambiguous text.
    """
    sample = text.strip()
    if not sample:
        return "other"

    devanagari_count = len(_DEVANAGARI_RE.findall(sample))
    ratio = devanagari_count / max(len(re.sub(r"\s+", "", text)), 1)
    if ratio > 0.2:
        return "hi"

    try:
        detected = detect(sample)
    except LangDetectException:
        logger.debug("Language detection failed for short or noisy text")
        return "other"

    if detected.startswith("hi"):
        return "hi"
    if detected.startswith("en"):
        return "en"
    return "other"
