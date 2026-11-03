"""Tesseract OCR recognition adapter implementing RecognitionEngine."""

from __future__ import annotations

import numpy as np
import pytesseract
from PIL import Image

from embedding_ocr.recognition.base import RecognitionEngine, RecognitionResult


class TesseractRecognizer(RecognitionEngine):
    """Adapter for Tesseract OCR using pytesseract."""

    def __init__(self, psm: int = 10, lang: str = "eng") -> None:
        """Initialize Tesseract recognizer.

        Args:
            psm: Page segmentation mode. Default 10: Treat image as a single character.
                 Use 7 or 8 for single text line / word.
            lang: Language code (default "eng").
        """
        self.psm = psm
        self.lang = lang
        self.config = f"--psm {psm}"

    def recognize(self, image: np.ndarray) -> RecognitionResult:
        """Recognize text in a single text-region crop.

        Args:
            image: Text-region image as uint8 numpy array (H, W, C) or (H, W).

        Returns:
            RecognitionResult with recognized text and confidence in [0, 1].
        """
        pil_img = Image.fromarray(image)
        try:
            data = pytesseract.image_to_data(
                pil_img,
                lang=self.lang,
                config=self.config,
                output_type=pytesseract.Output.DICT,
            )
            # data contains 'text' and 'conf'
            confidences = [
                float(c)
                for c, t in zip(
                    data.get("conf", []), data.get("text", []), strict=False
                )
                if str(c).strip() not in ("-1", "") and str(t).strip()
            ]
            raw_text = pytesseract.image_to_string(
                pil_img, lang=self.lang, config=self.config
            )
            cleaned_text = raw_text.strip().replace("\n", "").replace("\r", "")

            avg_conf = (
                (sum(confidences) / len(confidences) / 100.0)
                if confidences
                else (0.8 if cleaned_text else 0.0)
            )
            avg_conf = max(0.0, min(1.0, avg_conf))

            return RecognitionResult(text=cleaned_text, confidence=avg_conf)
        except Exception:
            # Fallback on raw string
            raw_text = pytesseract.image_to_string(
                pil_img, lang=self.lang, config=self.config
            ).strip()
            return RecognitionResult(text=raw_text, confidence=0.5 if raw_text else 0.0)
