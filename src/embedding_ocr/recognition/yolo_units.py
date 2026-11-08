"""YOLO character/unit detector and recognition engine.

Ultralytics YOLO is AGPL-3.0 or commercially licensed. The import is lazy so
the rest of the project remains usable without the training extra.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from statistics import median
from typing import Any

import cv2
import numpy as np

from embedding_ocr.data.scripts import is_rtl_script, script_for_text
from embedding_ocr.pipeline.types import DetectedUnit
from embedding_ocr.recognition.base import RecognitionEngine, RecognitionResult


def load_classes(path: str | Path | None) -> list[str] | None:
    """Load a YOLO class list from ``classes.json``."""
    if path is None:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    classes = payload.get("classes")
    if not isinstance(classes, list) or not all(
        isinstance(item, str) for item in classes
    ):
        message = f"Invalid classes file: {path}"
        raise ValueError(message)
    return classes


def _to_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value)


class YoloUnitDetector:
    """Adapter around an Ultralytics detection model."""

    def __init__(
        self,
        model: Any,
        classes: Sequence[str] | None = None,
        conf: float = 0.25,
        iou: float = 0.5,
        imgsz: int = 1280,
        device: str | None = None,
        max_det: int = 1000,
        color_order: str = "rgb",
    ) -> None:
        self._model = model
        self.classes = list(classes) if classes is not None else None
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.device = device
        self.max_det = max_det
        self.color_order = color_order

    @classmethod
    def from_checkpoint(
        cls,
        model_path: str | Path,
        classes_path: str | Path | None = None,
        **kwargs: Any,
    ) -> YoloUnitDetector:
        try:
            from ultralytics import YOLO  # type: ignore[import-untyped]
        except ImportError as exc:
            message = (
                "ultralytics is not installed. Install the training extra with "
                "`uv sync --extra train`."
            )
            raise ImportError(message) from exc
        model = YOLO(str(model_path))
        return cls(model=model, classes=load_classes(classes_path), **kwargs)

    def _class_name(self, class_id: int, names: Any) -> str:
        if self.classes is not None and 0 <= class_id < len(self.classes):
            return self.classes[class_id]
        if isinstance(names, dict):
            return str(names.get(class_id, class_id))
        if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
            return str(names[class_id])
        return str(class_id)

    def detect(self, image: np.ndarray) -> list[DetectedUnit]:
        """Detect character units in an RGB image."""
        if image is None or image.size == 0:
            return []
        source = _to_bgr(image) if self.color_order == "rgb" else image
        results = self._model.predict(
            source=source,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            max_det=self.max_det,
            verbose=False,
        )
        if not results:
            return []
        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        height, width = image.shape[:2]
        xyxy = _as_numpy(boxes.xyxy).reshape(-1, 4)
        classes = _as_numpy(boxes.cls).reshape(-1)
        confidences = _as_numpy(boxes.conf).reshape(-1)
        names = getattr(result, "names", None) or getattr(self._model, "names", {})

        units: list[DetectedUnit] = []
        for corners, class_value, confidence in zip(
            xyxy, classes, confidences, strict=False
        ):
            x0 = max(0.0, min(float(width), float(corners[0])))
            y0 = max(0.0, min(float(height), float(corners[1])))
            x1 = max(0.0, min(float(width), float(corners[2])))
            y1 = max(0.0, min(float(height), float(corners[3])))
            box_width = x1 - x0
            box_height = y1 - y0
            if box_width < 1.0 or box_height < 1.0:
                continue
            class_id = int(class_value)
            text = self._class_name(class_id, names)
            units.append(
                DetectedUnit(
                    bbox=(
                        int(round(x0)),
                        int(round(y0)),
                        int(round(box_width)),
                        int(round(box_height)),
                    ),
                    text=text,
                    confidence=float(confidence),
                    script=script_for_text(text),
                )
            )
        return units


def _line_is_rtl(units: Sequence[DetectedUnit]) -> bool:
    arabic = sum(is_rtl_script(unit.script or "") for unit in units)
    latin = sum((unit.script or "") == "latin" for unit in units)
    return arabic > latin


def group_units_by_line(
    units: Sequence[DetectedUnit], rtl: bool | None = None
) -> list[list[DetectedUnit]]:
    """Group boxes into visual lines and order each line by reading direction."""
    if not units:
        return []
    heights = [max(1.0, float(unit.bbox[3])) for unit in units]
    tolerance = max(4.0, median(heights) * 0.6)
    ordered = sorted(
        units,
        key=lambda unit: unit.bbox[1] + unit.bbox[3] / 2.0,
    )
    lines: list[list[DetectedUnit]] = []
    current: list[DetectedUnit] = []
    current_center = 0.0
    for unit in ordered:
        center = unit.bbox[1] + unit.bbox[3] / 2.0
        if current and abs(center - current_center) > tolerance:
            lines.append(current)
            current = []
        if not current:
            current_center = center
        current.append(unit)
        current_center = sum(
            item.bbox[1] + item.bbox[3] / 2.0 for item in current
        ) / len(current)

    if current:
        lines.append(current)
    for line in lines:
        line_rtl = _line_is_rtl(line) if rtl is None else rtl
        line.sort(
            key=lambda unit: unit.bbox[0] + (unit.bbox[2] if line_rtl else 0),
            reverse=line_rtl,
        )
    return lines


def units_to_text(
    lines: Sequence[Sequence[DetectedUnit]], space_gap_ratio: float = 0.9
) -> str:
    """Join ordered units, inserting spaces at gaps larger than a threshold."""
    text_lines: list[str] = []
    for line in lines:
        if not line:
            continue
        line_rtl = _line_is_rtl(line)
        widths = [max(1.0, float(unit.bbox[2])) for unit in line]
        gap_threshold = max(3.0, median(widths) * space_gap_ratio)
        output: list[str] = []
        previous: DetectedUnit | None = None
        for unit in line:
            if previous is not None:
                if line_rtl:
                    gap = previous.bbox[0] - (unit.bbox[0] + unit.bbox[2])
                else:
                    gap = unit.bbox[0] - (previous.bbox[0] + previous.bbox[2])
                if gap > gap_threshold and output:
                    output.append(" ")
            output.append(unit.text)
            previous = unit
        text_lines.append("".join(output))
    return "\n".join(text_lines)


class YoloUnitRecognizer(RecognitionEngine):
    """Recognition engine that reads text directly from YOLO unit classes."""

    def __init__(
        self,
        detector: YoloUnitDetector,
        rtl: bool | None = None,
        space_gap_ratio: float = 0.9,
    ) -> None:
        self.detector = detector
        self.rtl = rtl
        self.space_gap_ratio = space_gap_ratio

    def recognize(self, image: np.ndarray) -> RecognitionResult:
        units = self.detector.detect(image)
        if not units:
            return RecognitionResult(text="", confidence=0.0)
        lines = group_units_by_line(units, rtl=self.rtl)
        text = units_to_text(lines, space_gap_ratio=self.space_gap_ratio)
        confidence = float(np.mean([unit.confidence for unit in units]))
        return RecognitionResult(text=text, confidence=max(0.0, min(1.0, confidence)))
