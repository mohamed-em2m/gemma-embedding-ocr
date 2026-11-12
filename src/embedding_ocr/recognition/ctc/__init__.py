"""CTC line recognition: character vocabulary, text normalization, model."""

from __future__ import annotations

from embedding_ocr.recognition.ctc.charset import CharSet, UnknownCharacterError
from embedding_ocr.recognition.ctc.decode import (
    Decoded,
    greedy_decode,
    log_softmax,
    prefix_beam_search,
)
from embedding_ocr.recognition.ctc.normalize import (
    detect_script,
    normalize_for_cer,
    normalize_text,
)

__all__ = [
    "CharSet",
    "Decoded",
    "UnknownCharacterError",
    "detect_script",
    "greedy_decode",
    "log_softmax",
    "normalize_for_cer",
    "normalize_text",
    "prefix_beam_search",
]
