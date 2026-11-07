"""Script detection and reading-order helpers.

The OCR pipeline is script-agnostic at the interface level, but several
heuristics (reading order, case resolution, character segmentation) are
script-specific. This module centralizes the lightweight, dependency-free
logic used to normalize script names and detect RTL scripts.
"""

from __future__ import annotations

from collections.abc import Sequence

# Canonical script names used by the project.
LATIN = "latin"
ARABIC = "arabic"
HEBREW = "hebrew"
CYRILLIC = "cyrillic"
GREEK = "greek"
CJK = "cjk"
DEVANAGARI = "devanagari"
THAI = "thai"

RTL_SCRIPTS = frozenset(
    {
        ARABIC,
        HEBREW,
        "persian",
        "urdu",
        "pashto",
        "sindhi",
        "syriac",
        "thaana",
        "nko",
        "adlam",
    }
)

_SCRIPT_ALIASES: dict[str, str] = {
    "ar": ARABIC,
    "ara": ARABIC,
    "fas": ARABIC,
    "fa": ARABIC,
    "per": ARABIC,
    "urd": ARABIC,
    "ur": ARABIC,
    "pus": ARABIC,
    "ps": ARABIC,
    "snd": ARABIC,
    "sd": ARABIC,
    "he": HEBREW,
    "heb": HEBREW,
    "iw": HEBREW,
    "ru": CYRILLIC,
    "rus": CYRILLIC,
    "uk": CYRILLIC,
    "ukr": CYRILLIC,
    "bg": CYRILLIC,
    "el": GREEK,
    "ell": GREEK,
    "gre": GREEK,
    "zh": CJK,
    "zho": CJK,
    "ja": CJK,
    "jpn": CJK,
    "ko": CJK,
    "kor": CJK,
    "hi": DEVANAGARI,
    "hin": DEVANAGARI,
    "mr": DEVANAGARI,
    "ne": DEVANAGARI,
    "th": THAI,
    "tha": THAI,
}

# Ordered by specificity. Overlapping ranges are resolved by the first match.
_SCRIPT_RANGES: tuple[tuple[str, tuple[tuple[int, int], ...]], ...] = (
    (
        ARABIC,
        (
            (0x0600, 0x06FF),
            (0x0750, 0x077F),
            (0x08A0, 0x08FF),
            (0xFB50, 0xFDFF),
            (0xFE70, 0xFEFF),
        ),
    ),
    (HEBREW, ((0x0590, 0x05FF), (0xFB1D, 0xFB4F))),
    (CYRILLIC, ((0x0400, 0x04FF), (0x0500, 0x052F))),
    (GREEK, ((0x0370, 0x03FF), (0x1F00, 0x1FFF))),
    (DEVANAGARI, ((0x0900, 0x097F),)),
    (THAI, ((0x0E00, 0x0E7F),)),
    (
        CJK,
        (
            (0x3040, 0x30FF),
            (0x3400, 0x4DBF),
            (0x4E00, 0x9FFF),
            (0xAC00, 0xD7AF),
            (0xF900, 0xFAFF),
        ),
    ),
    (LATIN, ((0x0041, 0x005A), (0x0061, 0x007A), (0x00C0, 0x024F), (0x1E00, 0x1EFF))),
)


def normalize_script_name(script: str) -> str:
    """Return the canonical script name for a language/script alias.

    Examples:
        ``"ar"`` -> ``"arabic"``, ``"fa"`` -> ``"arabic"``, ``"Latin"`` -> ``"latin"``.
    """
    key = script.strip().lower().replace("_", "-")
    if key in _SCRIPT_ALIASES:
        return _SCRIPT_ALIASES[key]
    return key


def is_rtl_script(script: str) -> bool:
    """Return True when text in ``script`` is normally written right-to-left."""
    return normalize_script_name(script) in RTL_SCRIPTS


def _script_for_char(char: str) -> str | None:
    code = ord(char)
    for script, ranges in _SCRIPT_RANGES:
        for start, end in ranges:
            if start <= code <= end:
                return script
    return None


def char_in_script(char: str, script: str) -> bool:
    """Return True when ``char`` belongs to the Unicode block of ``script``."""
    return _script_for_char(char) == normalize_script_name(script)


def is_text_in_script(text: str, script: str) -> bool:
    """Return True when every non-space character of ``text`` fits ``script``.

    Latin also accepts the punctuation used by the existing vocabulary filter.
    Unknown/auto scripts accept any Unicode letter, mark, number or common
    punctuation, which keeps multilingual tokens usable without a hardcoded
    script list.
    """
    normalized = normalize_script_name(script)
    if normalized == LATIN:
        return all(
            char.isascii() and (char.isalnum() or char in "-_'.,:;!?") for char in text
        )
    if normalized == "auto":
        import unicodedata

        return all(
            char.isspace()
            or unicodedata.category(char)[0] in {"L", "M", "N"}
            or char in "-_'.,:;!?،؛؟"
            for char in text
        )
    return all(char.isspace() or char_in_script(char, normalized) for char in text)


def script_for_text(text: str) -> str:
    """Detect the dominant script in ``text`` using Unicode ranges.

    Digits, punctuation, whitespace and combining marks are ignored unless
    they are part of a recognized script block. Returns ``"unknown"`` when no
    script can be inferred.
    """
    counts: dict[str, int] = {}
    for char in text:
        script = _script_for_char(char)
        if script is not None:
            counts[script] = counts.get(script, 0) + 1

    if not counts:
        return "unknown"
    return max(counts.items(), key=lambda item: item[1])[0]


def is_rtl_text(text: str) -> bool:
    """Return True when the dominant script of ``text`` is RTL."""
    return is_rtl_script(script_for_text(text))


def reading_order(xs: Sequence[float], rtl: bool = False) -> list[int]:
    """Return indices of ``xs`` in logical reading order.

    For LTR scripts this is ascending x; for RTL scripts it is descending x.
    """
    return sorted(range(len(xs)), key=lambda i: xs[i], reverse=rtl)
