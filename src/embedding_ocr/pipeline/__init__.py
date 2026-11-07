"""Preprocessing, detection, cropping, reconstruction, and pipeline orchestration."""

from embedding_ocr.pipeline.crop import crop_regions, normalize_bbox
from embedding_ocr.pipeline.detect import (
    CraftWordDetectorAdapter,
    FullImageDetector,
    PaddleTextDetector,
    TextDetector,
    VisualWordDetector,
    build_detector,
)
from embedding_ocr.pipeline.pipeline import OCRPipeline
from embedding_ocr.pipeline.preprocess import (
    adaptive_binarize,
    deslant_image,
    estimate_char_count_and_width,
    estimate_slant_angle,
    preprocess_handwriting_crop,
)
from embedding_ocr.pipeline.reconstruct import (
    group_lines,
    mean_confidence,
    order_regions,
    reconstruct_text,
)
from embedding_ocr.pipeline.types import (
    OCRResult,
    RecognizedRegion,
    TextRegion,
)

__all__ = [
    "CraftWordDetectorAdapter",
    "FullImageDetector",
    "OCRPipeline",
    "OCRResult",
    "PaddleTextDetector",
    "RecognizedRegion",
    "TextDetector",
    "TextRegion",
    "VisualWordDetector",
    "adaptive_binarize",
    "build_detector",
    "crop_regions",
    "deslant_image",
    "estimate_char_count_and_width",
    "estimate_slant_angle",
    "group_lines",
    "mean_confidence",
    "normalize_bbox",
    "order_regions",
    "preprocess_handwriting_crop",
    "reconstruct_text",
]
