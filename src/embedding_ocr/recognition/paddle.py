"""PaddleOCR recognition adapter implementing RecognitionEngine.

Uses the recognition-only model (e.g. PP-OCRv4 / PP-OCRv6) to match the
RecognitionEngine interface (single crop -> text + confidence) and bypass
the pipeline detection oneDNN PIR issue.
"""

from __future__ import annotations

import numpy as np
from paddlex import create_model

from embedding_ocr.recognition.base import RecognitionEngine, RecognitionResult


class PaddleRecognizer(RecognitionEngine):
    """Adapter for PaddleOCR recognition."""

    def __init__(
        self,
        model_name: str = "en_PP-OCRv4_mobile_rec",
        use_gpu: bool = False,
        device: str | None = None,
    ) -> None:
        """Initialize PaddleOCR text recognition model.

        Args:
            model_name: Recognition model name (default "en_PP-OCRv4_mobile_rec").
            use_gpu: Unused legacy kwarg kept for call-site compatibility.
            device: Unused legacy kwarg kept for call-site compatibility.
        """
        self.model = create_model(model_name=model_name)

    def recognize(self, image: np.ndarray) -> RecognitionResult:
        """Recognize text in a single text-region crop.

        Args:
            image: Text-region image as uint8 numpy array (H, W, C) or (H, W).

        Returns:
            RecognitionResult with recognized text and confidence in [0, 1].
        """
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)

        try:
            preds = list(self.model.predict(image))
            if not preds:
                return RecognitionResult(text="", confidence=0.0)

            res = preds[0]
            rec_text = str(res.get("rec_text", "")).strip()
            rec_score = float(res.get("rec_score", 0.0))
            rec_score = max(0.0, min(1.0, rec_score))

            return RecognitionResult(text=rec_text, confidence=rec_score)
        except Exception:
            return RecognitionResult(text="", confidence=0.0)
