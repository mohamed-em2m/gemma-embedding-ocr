"""Shared pipeline dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TextRegion:
    """A detected text region in image coordinates."""

    bbox: tuple[int, int, int, int]  # x, y, w, h
    confidence: float = 1.0


@dataclass(frozen=True)
class DetectedUnit:
    """A detected character or grapheme unit with its class text."""

    bbox: tuple[int, int, int, int]  # x, y, w, h
    text: str
    confidence: float = 1.0
    script: str | None = None


@dataclass(frozen=True)
class RecognizedRegion:
    """A text region and its recognition output."""

    region: TextRegion
    text: str
    confidence: float


@dataclass(frozen=True)
class OCRResult:
    """End-to-end OCR result."""

    text: str
    confidence: float
    regions: list[RecognizedRegion] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)
