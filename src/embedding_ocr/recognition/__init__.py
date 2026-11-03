"""Recognition module for EmbeddingOCR."""

from __future__ import annotations

from embedding_ocr.recognition.base import RecognitionEngine, RecognitionResult

__all__ = [
    "RecognitionEngine",
    "RecognitionResult",
]

# Optional backend recognizers (lazy import or guarded to prevent hard dependency crashes)
try:
    from embedding_ocr.recognition.dots_mocr import DotsMOCRRecognizer
    __all__.append("DotsMOCRRecognizer")
except ImportError:
    pass

try:
    from embedding_ocr.recognition.easyocr import EasyOCRRecognizer
    __all__.append("EasyOCRRecognizer")
except ImportError:
    pass

try:
    from embedding_ocr.recognition.glm_ocr import GLMOCRRecognizer
    __all__.append("GLMOCRRecognizer")
except ImportError:
    pass

try:
    from embedding_ocr.recognition.lighton_ocr import LightOnOCRRecognizer
    __all__.append("LightOnOCRRecognizer")
except ImportError:
    pass

try:
    from embedding_ocr.recognition.oxr import OXRRecognizer
    __all__.append("OXRRecognizer")
except ImportError:
    pass

try:
    from embedding_ocr.recognition.paddle import PaddleRecognizer
    __all__.append("PaddleRecognizer")
except ImportError:
    pass

try:
    from embedding_ocr.recognition.tesseract import TesseractRecognizer
    __all__.append("TesseractRecognizer")
except ImportError:
    pass

