"""End-to-end OCR pipeline: preprocess -> detect -> crop -> recognize -> join."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np

from embedding_ocr.config import PipelineSettings
from embedding_ocr.pipeline.crop import crop_regions
from embedding_ocr.pipeline.detect import TextDetector
from embedding_ocr.pipeline.preprocess import preprocess_handwriting_crop
from embedding_ocr.pipeline.reconstruct import mean_confidence, reconstruct_text
from embedding_ocr.pipeline.types import (
    OCRResult,
    RecognizedRegion,
    TextRegion,
)
from embedding_ocr.recognition.base import RecognitionEngine


class OCRPipeline:
    """Compose a detector and a recognizer into a reusable OCR pipeline."""

    def __init__(
        self,
        detector: TextDetector,
        recognizer: RecognitionEngine,
        settings: PipelineSettings | None = None,
        rtl: bool = False,
    ) -> None:
        self.detector = detector
        self.recognizer = recognizer
        self.settings = settings or PipelineSettings()
        self.rtl = rtl

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        if not self.settings.preprocess:
            return image
        return preprocess_handwriting_crop(
            image,
            deslant=self.settings.deslant,
            binarize=self.settings.binarize,
            padding=self.settings.padding,
        )

    def _cache_key(self, image: np.ndarray) -> str:
        digest = hashlib.sha256()
        digest.update(str(image.shape).encode())
        digest.update(image.tobytes())
        digest.update(self.detector.__class__.__name__.encode())
        digest.update(str(self.rtl).encode())
        return digest.hexdigest()

    def _cache_path(self, key: str) -> Path | None:
        cache_dir = self.settings.detection_cache
        if cache_dir is None:
            return None
        return Path(cache_dir) / f"{key}.json"

    def _load_regions(self, path: Path) -> list[TextRegion] | None:
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text())
            return [
                TextRegion(
                    bbox=tuple(int(v) for v in item["bbox"]),  # type: ignore[arg-type]
                    confidence=float(item.get("confidence", 1.0)),
                )
                for item in payload["regions"]
            ]
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None

    def _save_regions(self, path: Path, regions: list[TextRegion]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "regions": [
                {"bbox": list(region.bbox), "confidence": region.confidence}
                for region in regions
            ]
        }
        path.write_text(json.dumps(payload, indent=2))

    @staticmethod
    def _ensure_rgb(crop: np.ndarray) -> np.ndarray:
        if crop.ndim == 2:
            return cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
        if crop.shape[2] == 4:
            return cv2.cvtColor(crop, cv2.COLOR_RGBA2RGB)
        return crop

    def recognize(self, image: np.ndarray) -> OCRResult:
        """Run the full pipeline on an image."""
        if image is None or image.size == 0:
            return OCRResult(text="", confidence=0.0)

        started = time.perf_counter()
        processed = self._preprocess(image)
        key = self._cache_key(processed)
        cache_path = self._cache_path(key)
        cached = False

        regions = self._load_regions(cache_path) if cache_path is not None else None
        if regions is None:
            regions = self.detector.detect(processed)
            if cache_path is not None:
                self._save_regions(cache_path, regions)
        else:
            cached = True

        crops = crop_regions(processed, regions)
        recognized: list[RecognizedRegion] = []
        for region, crop in zip(regions, crops, strict=False):
            if crop.size == 0:
                continue
            result = self.recognizer.recognize(self._ensure_rgb(crop))
            recognized.append(
                RecognizedRegion(
                    region=region,
                    text=result.text,
                    confidence=result.confidence,
                )
            )

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return OCRResult(
            text=reconstruct_text(recognized, rtl=self.rtl),
            confidence=mean_confidence(recognized),
            regions=recognized,
            metadata={
                "detector": self.detector.__class__.__name__,
                "recognizer": self.recognizer.__class__.__name__,
                "num_regions": len(regions),
                "detection_cached": cached,
                "latency_ms": elapsed_ms,
            },
        )
