"""Short-text sampling, validation, and hard-negative construction."""

from __future__ import annotations

import hashlib
import random
import re
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from rapidfuzz import fuzz, process

MAX_WORDS = 20
MAX_CHARS = 120
_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'&./+-]*")
_WHITESPACE_RE = re.compile(r"\s+")
LOOKALIKE_SWAPS: tuple[tuple[str, str], ...] = (
    ("O", "0"),
    ("0", "O"),
    ("l", "1"),
    ("1", "l"),
    ("I", "1"),
    ("rn", "m"),
    ("m", "rn"),
    ("cl", "d"),
    ("d", "cl"),
    ("S", "5"),
    ("5", "S"),
    ("B", "8"),
    ("8", "B"),
    ("Z", "2"),
    ("2", "Z"),
)


@dataclass(frozen=True)
class TextSample:
    """A normalized training text and its stable identity hash."""

    text: str
    text_id: str
    source: str = "unknown"

    @classmethod
    def from_text(cls, text: str, *, source: str = "unknown") -> TextSample | None:
        normalized = normalize_text(text)
        if not is_valid_text(normalized):
            return None
        return cls(text=normalized, text_id=text_id(normalized), source=source)


def normalize_text(text: str) -> str:
    """NFKC-normalize, collapse whitespace, and strip line breaks."""
    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.replace("\u00a0", " ")
    normalized = _WHITESPACE_RE.sub(" ", normalized).strip()
    return normalized


def is_valid_text(text: str) -> bool:
    """Return whether a text is a usable 1-20 word OCR label."""
    if not text or len(text) > MAX_CHARS:
        return False
    words = text.split()
    if not 1 <= len(words) <= MAX_WORDS:
        return False
    return bool(_WORD_RE.search(text))


def text_id(text: str) -> str:
    """Stable hash used to avoid duplicate texts in a contrastive batch."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def dedupe_texts(
    texts: Iterable[str],
    *,
    limit: int | None = None,
    source: str = "unknown",
) -> list[TextSample]:
    """Normalize, validate, and deduplicate texts while preserving order."""
    seen: set[str] = set()
    samples: list[TextSample] = []
    for raw in texts:
        sample = TextSample.from_text(raw, source=source)
        if sample is None or sample.text_id in seen:
            continue
        seen.add(sample.text_id)
        samples.append(sample)
        if limit is not None and len(samples) >= limit:
            break
    return samples


def words_from_document(text: str, *, rng: random.Random) -> str | None:
    """Extract a random 1-20 word span from a longer document."""
    words = _WORD_RE.findall(text)
    if not words:
        return None
    count = rng.randint(1, min(MAX_WORDS, len(words)))
    start = rng.randint(0, len(words) - count)
    candidate = " ".join(words[start : start + count])
    return candidate if is_valid_text(candidate) else None


def lookalike_negative(text: str, *, rng: random.Random) -> str:
    """Apply one visually confusable substitution, if possible."""
    candidates: list[str] = []
    for index in range(len(text)):
        for source, target in LOOKALIKE_SWAPS:
            if text.startswith(source, index):
                candidate = text[:index] + target + text[index + len(source) :]
                if candidate != text and is_valid_text(candidate):
                    candidates.append(candidate)
    return rng.choice(candidates) if candidates else text


def edit_negative(text: str, *, rng: random.Random) -> str:
    """Create a one-edit negative by insertion, deletion, or substitution."""
    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    if not text:
        return "x"
    operation = rng.choice(("substitute", "insert", "delete"))
    index = rng.randrange(len(text))
    if operation == "substitute":
        replacement = rng.choice(alphabet)
        return text[:index] + replacement + text[index + 1 :]
    if operation == "insert":
        return text[:index] + rng.choice(alphabet) + text[index:]
    if len(text) == 1:
        return text + rng.choice(alphabet)
    return text[:index] + text[index + 1 :]


def hard_negative(
    text: str,
    lexicon: Sequence[str],
    *,
    rng: random.Random,
    min_ratio: float = 75.0,
) -> str:
    """Choose a lexically close but different text as a hard negative."""
    candidates = [candidate for candidate in lexicon if candidate != text]
    if candidates:
        match = process.extractOne(
            text,
            candidates,
            scorer=fuzz.ratio,
            score_cutoff=min_ratio,
        )
        if match is not None:
            return str(match[0])
    swapped = lookalike_negative(text, rng=rng)
    if swapped != text:
        return swapped
    for _ in range(8):
        candidate = edit_negative(text, rng=rng)
        if candidate != text and is_valid_text(candidate):
            return candidate
    return text + rng.choice("0123456789")


def make_synthetic_texts(
    *,
    count: int,
    seed: int = 42,
    lexicon: Sequence[str] | None = None,
) -> list[TextSample]:
    """Generate synthetic short strings with names, numbers, and mixed case."""
    rng = random.Random(seed)
    words = list(lexicon or DEFAULT_SYNTHETIC_WORDS)
    first_names = [
        "Alice",
        "Bob",
        "Chloe",
        "Dmitri",
        "Elena",
        "Farah",
        "Gabriel",
        "Hana",
        "Ivan",
        "Julia",
        "Kofi",
        "Lina",
        "Mateo",
        "Nadia",
        "Omar",
        "Priya",
        "Quentin",
        "Rosa",
        "Samir",
        "Tara",
    ]
    suffixes = ["Street", "Road", "Avenue", "Lane", "Boulevard"]
    samples: list[TextSample] = []
    seen: set[str] = set()
    attempts = 0
    while len(samples) < count and attempts < count * 20:
        attempts += 1
        kind = rng.random()
        if kind < 0.45:
            n = rng.randint(1, 8)
            text = " ".join(rng.choice(words) for _ in range(n))
        elif kind < 0.65:
            text = f"{rng.choice(first_names)} {rng.choice(words)}"
        elif kind < 0.78:
            text = f"{rng.choice(words)}_{rng.randint(0, 9999)}"
        elif kind < 0.9:
            text = f"{rng.randint(1, 9999)} {rng.choice(suffixes)}"
        else:
            text = rng.choice(
                (
                    f"{rng.choice(first_names)}.{rng.choice(words)}@example.com",
                    f"ID-{rng.randint(100, 999)}-{rng.randint(100, 999)}",
                    f"${rng.randint(1, 9999)}.{rng.randint(0, 99):02d}",
                    f"{rng.randint(1, 99)}% {rng.choice(words)}",
                )
            )
        if rng.random() < 0.3:
            text = text.upper()
        elif rng.random() < 0.3:
            text = text.title()
        sample = TextSample.from_text(text, source="synthetic")
        if sample is not None and sample.text_id not in seen:
            seen.add(sample.text_id)
            samples.append(sample)
    return samples


DEFAULT_SYNTHETIC_WORDS = (
    "hello",
    "world",
    "text",
    "image",
    "font",
    "letter",
    "detect",
    "training",
    "model",
    "dataset",
    "sample",
    "ocr",
    "vision",
    "computer",
    "python",
    "paper",
    "line",
    "word",
    "character",
    "document",
    "invoice",
    "total",
    "account",
    "service",
    "client",
    "network",
    "learning",
    "analysis",
    "report",
    "system",
    "quality",
    "result",
    "value",
    "market",
    "project",
    "design",
    "number",
    "price",
    "order",
    "address",
)
