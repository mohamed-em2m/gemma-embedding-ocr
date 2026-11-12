"""Script-aware text normalization for line-level CTC recognition.

Two normalizations coexist and must not be confused:

``normalize_text``
    Applied once while building the training manifest. It folds Unicode
    presentation forms, removes invisible characters, strips Arabic
    diacritics and tatweel, folds the alef variants and collapses
    whitespace. The CTC model is trained to emit exactly this form.

``normalize_for_cer``
    Applied at evaluation time to references and hypotheses alike. It keeps
    Latin letter case so case errors are not forgiven, and applies the same
    Arabic folding so the score matches what the model was trained on.

The Arabic policy is a deliberate trade-off: folding ``أ إ آ ٱ`` to ``ا`` and
dropping harakat removes distinctions that most handwriting corpora label
inconsistently. It lowers the achievable CER and must be recorded in
``docs/decisions.md`` whenever it changes.
"""

from __future__ import annotations

import re
import unicodedata

from embedding_ocr.data.scripts import ARABIC, LATIN, char_in_script

_WHITESPACE_RE = re.compile(r"\s+")
# Arabic harakat, Quranic annotation marks and tatweel (U+0640).
_ARABIC_MARKS_RE = re.compile("[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed\u0640]")
# Zero-width and bidi control characters never belong to a label.
_INVISIBLE_RE = re.compile("[\u200b-\u200f\u202a-\u202e\u2060-\u2064\ufeff]")
_ALEF_FOLD = str.maketrans(
    {
        "\u0623": "\u0627",
        "\u0625": "\u0627",
        "\u0622": "\u0627",
        "\u0671": "\u0627",
        "\u0672": "\u0627",
        "\u0673": "\u0627",
        "\u0675": "\u0627",
    }
)


def normalize_text(
    text: str,
    *,
    strip_diacritics: bool = True,
    fold_alefs: bool = True,
) -> str:
    """Normalize a label into the form the CTC model is trained to emit.

    Args:
        text: Raw label, possibly containing presentation forms or diacritics.
        strip_diacritics: Drop Arabic harakat, Quranic marks and tatweel.
        fold_alefs: Fold ``أ إ آ ٱ`` variants onto bare alef ``ا``.

    Returns:
        A single-line string with collapsed whitespace and no control
        characters. The empty string means the label is unusable.
    """
    normalized = unicodedata.normalize("NFKC", text)
    normalized = _INVISIBLE_RE.sub("", normalized)
    normalized = (
        normalized.replace("\u00a0", " ").replace("\u2007", " ").replace("\u202f", " ")
    )
    normalized = "".join(
        char
        for char in normalized
        if char.isspace() or unicodedata.category(char) not in {"Cc", "Cf"}
    )
    if fold_alefs:
        normalized = normalized.translate(_ALEF_FOLD)
    if strip_diacritics:
        normalized = _ARABIC_MARKS_RE.sub("", normalized)
    return _WHITESPACE_RE.sub(" ", normalized).strip()


def detect_script(text: str) -> str:
    """Return ``latin``, ``arabic``, ``mixed`` or ``unknown`` for ``text``.

    Punctuation, digits and whitespace are ignored; only letters decide the
    result. ``mixed`` means the line contains both Latin and Arabic letters,
    which is common in Arabic technical text and in KHATT transcripts.
    """
    arabic = sum(1 for char in text if char_in_script(char, ARABIC))
    latin = sum(1 for char in text if char_in_script(char, LATIN))
    if arabic and latin:
        return "mixed"
    if arabic:
        return ARABIC
    if latin:
        return LATIN
    return "unknown"


def normalize_for_cer(
    text: str,
    *,
    case_sensitive: bool = True,
    strip_diacritics: bool = True,
) -> str:
    """Normalize text for error-rate scoring.

    Args:
        text: Reference or hypothesis string.
        case_sensitive: Keep Latin letter case. Set to ``False`` for a
            case-insensitive secondary score.
        strip_diacritics: Apply the same Arabic folding used in training.

    Returns:
        The normalized string. Whitespace is collapsed but never removed
        inside the text, because word boundaries carry CER signal.
    """
    normalized = normalize_text(text, strip_diacritics=strip_diacritics)
    if not case_sensitive:
        normalized = normalized.casefold()
    return normalized
