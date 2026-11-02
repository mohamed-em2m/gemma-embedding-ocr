"""Recognition module for EmbeddingOCR."""

from __future__ import annotations

from embedding_ocr.recognition.base import RecognitionEngine, RecognitionResult
from embedding_ocr.recognition.easyocr import EasyOCRRecognizer
from embedding_ocr.recognition.glm_ocr import GLMOCRRecognizer
from embedding_ocr.recognition.paddle import PaddleRecognizer
from embedding_ocr.recognition.tesseract import TesseractRecognizer

__all__ = [
    "EasyOCRRecognizer",
    "GLMOCRRecognizer",
    "PaddleRecognizer",
    "RecognitionEngine",
    "RecognitionResult",
    "TesseractRecognizer",
]
