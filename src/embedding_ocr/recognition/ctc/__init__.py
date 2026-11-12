"""CTC line recognition: character vocabulary, text normalization, model."""

from __future__ import annotations

from embedding_ocr.recognition.ctc.charset import CharSet, UnknownCharacterError
from embedding_ocr.recognition.ctc.normalize import (
    detect_script,
    normalize_for_cer,
    normalize_text,
)

__all__ = [
    "CharSet",
    "UnknownCharacterError",
    "detect_script",
    "normalize_for_cer",
    "normalize_text",
]
