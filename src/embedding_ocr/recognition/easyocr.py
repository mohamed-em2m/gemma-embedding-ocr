"""EasyOCR recognition adapter implementing RecognitionEngine."""

from __future__ import annotations

import easyocr
import numpy as np

from embedding_ocr.recognition.base import RecognitionEngine, RecognitionResult


class EasyOCRRecognizer(RecognitionEngine):
    """Adapter for EasyOCR."""

    def __init__(
        self,
        lang_list: list[str] | None = None,
        gpu: bool = True,
        recognizer_only: bool = True,
    ) -> None:
        """Initialize EasyOCR reader.

        Args:
            lang_list: Languages to support. Defaults to ['en'].
            gpu: Whether to enable GPU execution.
            recognizer_only: If True, uses recognize() directly on crop without full detection.
        """
        if lang_list is None:
            lang_list = ["en"]
        self.reader = easyocr.Reader(lang_list, gpu=gpu)
        self.recognizer_only = recognizer_only

    def recognize(self, image: np.ndarray) -> RecognitionResult:
        """Recognize text in a single text-region crop.

        Args:
            image: Text-region image as uint8 numpy array (H, W, C) or (H, W).

        Returns:
            RecognitionResult with recognized text and confidence in [0, 1].
        """
        # Ensure RGB format
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)

        try:
            # reader.readtext accepts cropped image array directly
            results = self.reader.readtext(image)
            if not results:
                return RecognitionResult(text="", confidence=0.0)

            # results is a list of (bbox, text, prob)
            texts = [r[1] for r in results]
            confs = [float(r[2]) for r in results]

            full_text = " ".join(texts).strip()
            avg_conf = sum(confs) / len(confs) if confs else 0.0
            avg_conf = max(0.0, min(1.0, avg_conf))

            return RecognitionResult(text=full_text, confidence=avg_conf)
        except Exception:
            return RecognitionResult(text="", confidence=0.0)
