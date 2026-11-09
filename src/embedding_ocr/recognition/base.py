"""Recognition base interfaces: RecognitionEngine and RecognitionResult."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class RecognitionResult:
    """Result of a recognition pass on a text region.

    Attributes:
        text: The recognized text string.
        confidence: Confidence score in [0, 1].
        metadata: Optional debug data, including recognizer-local ``word_boxes``.
    """

    text: str
    confidence: float
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not (0.0 <= self.confidence <= 1.0):
            msg = f"confidence must be in [0, 1]; got {self.confidence}"
            raise ValueError(msg)


class RecognitionEngine(Protocol):
    """Protocol for all OCR recognition engines.

    Only the recognition stage is replaced; preprocessing, detection,
    cropping and reconstruction remain unchanged across implementations.
    """

    def recognize(self, image: np.ndarray) -> RecognitionResult:
        """Recognize text in a single text-region crop.

        Args:
            image: Text-region image as uint8 numpy array (H, W, C).

        Returns:
            RecognitionResult with the recognized text and confidence.
        """
        ...
