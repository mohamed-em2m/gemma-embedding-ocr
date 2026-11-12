"""Character vocabulary for the CTC line recognizer.

The vocabulary is a flat, ordered tuple of single characters. Index ``0`` is
reserved for the CTC blank, so a model with ``len(chars) == n`` has ``n + 1``
output logits. Characters are sorted by code point so the same training set
always produces the same vocabulary, and the vocabulary is persisted next to
the checkpoint so training and inference can never disagree.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

BLANK_INDEX = 0
CHARSET_VERSION = 1
CHARSET_FILENAME = "charset.json"


class UnknownCharacterError(ValueError):
    """Raised when a text contains characters outside the vocabulary."""


@lru_cache(maxsize=8)
def _lookup(chars: tuple[str, ...]) -> dict[str, int]:
    return {char: index + 1 for index, char in enumerate(chars)}


@dataclass(frozen=True)
class CharSet:
    """An immutable character vocabulary with a reserved CTC blank slot.

    Attributes:
        chars: Single-character symbols in index order, excluding the blank.
            Index ``i`` in this tuple maps to logit ``i + 1``.
    """

    chars: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.chars:
            message = "charset must contain at least one character"
            raise ValueError(message)
        if len(set(self.chars)) != len(self.chars):
            message = "charset contains duplicate characters"
            raise ValueError(message)
        if any(len(char) != 1 for char in self.chars):
            message = "charset entries must be single characters"
            raise ValueError(message)

    @property
    def size(self) -> int:
        """Number of output logits, including the blank."""
        return len(self.chars) + 1

    def __len__(self) -> int:
        return self.size

    def __contains__(self, char: object) -> bool:
        return isinstance(char, str) and char in _lookup(self.chars)

    def index(self, char: str) -> int:
        """Return the logit index of ``char`` (1-based; blank is 0)."""
        try:
            return _lookup(self.chars)[char]
        except KeyError:
            message = f"character {char!r} (U+{ord(char):04X}) is not in the charset"
            raise UnknownCharacterError(message) from None

    def unknown(self, text: str) -> set[str]:
        """Return the characters of ``text`` that are missing from the charset."""
        lookup = _lookup(self.chars)
        return {char for char in text if char not in lookup}

    def encode(self, text: str, *, strict: bool = True) -> list[int]:
        """Encode ``text`` into logit indices.

        Args:
            text: Normalized label or hypothesis.
            strict: Raise :class:`UnknownCharacterError` on unknown characters
                instead of silently dropping them.

        Returns:
            Indices in ``[1, size)``. The blank is never emitted.
        """
        lookup = _lookup(self.chars)
        encoded: list[int] = []
        unknown: set[str] = set()
        for char in text:
            index = lookup.get(char)
            if index is None:
                unknown.add(char)
                continue
            encoded.append(index)
        if unknown and strict:
            rendered = ", ".join(
                f"{char!r} (U+{ord(char):04X})" for char in sorted(unknown)
            )
            message = f"text contains characters not in the charset: {rendered}"
            raise UnknownCharacterError(message)
        return encoded

    def decode(self, indices: Sequence[int]) -> str:
        """Decode logit indices back to text, skipping the blank and padding."""
        pieces: list[str] = []
        for index in indices:
            if 1 <= index <= len(self.chars):
                pieces.append(self.chars[index - 1])
        return "".join(pieces)

    @classmethod
    def from_texts(cls, texts: Iterable[str], *, extra: Iterable[str] = ()) -> CharSet:
        """Build a deterministic vocabulary from training texts.

        Args:
            texts: Normalized training labels. Only these decide the charset;
                validation and test text must never contribute.
            extra: Additional characters to force into the vocabulary.
        """
        symbols: set[str] = {char for text in texts for char in text}
        symbols.update(extra)
        if not symbols:
            message = "cannot build a charset from empty texts"
            raise ValueError(message)
        return cls(tuple(sorted(symbols)))

    def save(self, path: Path) -> None:
        """Write the vocabulary as JSON, sorted, UTF-8."""
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": CHARSET_VERSION, "chars": list(self.chars)}
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: Path) -> CharSet:
        """Load a vocabulary written by :meth:`save`."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        chars = payload.get("chars")
        if not isinstance(chars, list) or not all(isinstance(c, str) for c in chars):
            message = f"invalid charset file: {path}"
            raise ValueError(message)
        return cls(tuple(chars))
