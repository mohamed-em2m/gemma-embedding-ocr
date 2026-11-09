"""Text detection abstractions and backends."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Protocol

import numpy as np

from embedding_ocr.config import DetectorSettings
from embedding_ocr.pipeline.types import TextRegion


class TextDetector(Protocol):
    """Protocol for text detection backends."""

    def detect(self, image: np.ndarray) -> list[TextRegion]:
        """Detect text regions in reading order."""
        ...


class FullImageDetector:
    """Fallback detector that treats the whole image as one region."""

    def detect(self, image: np.ndarray) -> list[TextRegion]:
        height, width = image.shape[:2]
        return [TextRegion(bbox=(0, 0, width, height))]


class VisualWordDetector:
    """Word detector based on connected components and whitespace gaps."""

    def __init__(self, rtl: bool = False, min_component_area: int = 8) -> None:
        from embedding_ocr.recognition.embedding.hierarchical.segmentation import (
            VisualWordSegmenter,
        )

        self.rtl = rtl
        self._segmenter = VisualWordSegmenter(min_component_area=min_component_area)

    def detect(self, image: np.ndarray) -> list[TextRegion]:
        boxes = self._segmenter.extract_word_boxes(image, rtl=self.rtl)
        return [TextRegion(bbox=box) for box in boxes]


class CraftWordDetectorAdapter:
    """Adapter around the optional CRAFT word detector."""

    def __init__(self, rtl: bool = False, cuda: bool = False) -> None:
        from embedding_ocr.recognition.embedding.hierarchical.craft_detector import (
            CraftWordDetector,
        )

        self.rtl = rtl
        self._detector = CraftWordDetector(cuda=cuda)

    def detect(self, image: np.ndarray) -> list[TextRegion]:
        boxes = self._detector.detect_word_boxes(image, rtl=self.rtl)
        return [TextRegion(bbox=box) for box in boxes]


def _poly_to_bbox(poly: Any, width: int, height: int) -> tuple[int, int, int, int]:
    points = np.asarray(poly, dtype=np.float32)
    if points.ndim == 1:
        points = points.reshape(-1, 2)
    if points.size < 4:
        return 0, 0, 0, 0
    x0 = max(0, int(np.floor(points[:, 0].min())))
    y0 = max(0, int(np.floor(points[:, 1].min())))
    x1 = min(width, int(np.ceil(points[:, 0].max())))
    y1 = min(height, int(np.ceil(points[:, 1].max())))
    return x0, y0, max(0, x1 - x0), max(0, y1 - y0)


_LOGGER = logging.getLogger(__name__)


class PaddleTextDetector:
    """PaddleOCR detection backend.

    PaddleOCR 3.x exposes ``predict``; older releases expose ``ocr``. The
    parser accepts both so the adapter can be used with either API. The exact
    result schema is version-dependent and remains **unverified** in this repo.
    """

    def __init__(self, lang: str = "en", rtl: bool = False) -> None:
        # Paddle 3.x can crash in its oneDNN/PIR path on some CPU builds.
        os.environ.setdefault("FLAGS_use_mkldnn", "0")
        try:
            from paddleocr import PaddleOCR  # type: ignore[import-untyped]
        except ImportError as exc:
            msg = (
                "paddleocr is not installed. Install it with "
                "`uv sync --extra baseline` or `uv pip install paddleocr paddlepaddle`."
            )
            raise ImportError(msg) from exc

        self.rtl = rtl
        try:
            self._ocr = PaddleOCR(
                lang=lang,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )
        except TypeError:
            self._ocr = PaddleOCR(lang=lang)

    @staticmethod
    def _iter_polys(result: Any) -> list[Any]:
        if isinstance(result, dict):
            for key in ("dt_polys", "rec_polys", "boxes"):
                if key in result:
                    return list(result[key])
            res = result.get("res")
            if isinstance(res, dict):
                return PaddleTextDetector._iter_polys(res)
            return []
        payload = getattr(result, "json", None)
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except json.JSONDecodeError:
                payload = None
        if isinstance(payload, dict):
            return PaddleTextDetector._iter_polys(payload)
        if isinstance(result, (list, tuple)):
            return list(result)
        return []

    def detect(self, image: np.ndarray) -> list[TextRegion]:
        try:
            if hasattr(self._ocr, "predict"):
                raw_results = list(self._ocr.predict(image))
            else:
                raw_results = [self._ocr.ocr(image, det=True, rec=False)]
        except NotImplementedError as exc:
            _LOGGER.warning(
                "Paddle detector failed in its oneDNN/PIR runtime (%s); "
                "falling back to a full-image region.",
                exc,
            )
            return FullImageDetector().detect(image)

        height, width = image.shape[:2]
        regions: list[TextRegion] = []
        for raw in raw_results:
            for poly in self._iter_polys(raw):
                bbox = _poly_to_bbox(poly, width, height)
                if bbox[2] > 2 and bbox[3] > 2:
                    regions.append(TextRegion(bbox=bbox))

        regions.sort(key=lambda region: region.bbox[0], reverse=self.rtl)
        return regions if regions else FullImageDetector().detect(image)


def build_detector(settings: DetectorSettings, rtl: bool = False) -> TextDetector:
    """Create a detector from configuration."""
    backend = settings.backend.lower()
    resolved_rtl = settings.rtl if settings.rtl is not None else rtl
    if backend == "full":
        return FullImageDetector()
    if backend == "visual":
        return VisualWordDetector(
            rtl=resolved_rtl, min_component_area=settings.min_component_area
        )
    if backend == "craft":
        return CraftWordDetectorAdapter(rtl=resolved_rtl, cuda=settings.craft_cuda)
    if backend == "paddle":
        return PaddleTextDetector(lang=settings.lang, rtl=resolved_rtl)
    msg = f"Unknown detector backend: {settings.backend!r}"
    raise ValueError(msg)
