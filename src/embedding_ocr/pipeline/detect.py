"""Text detection abstractions and backends."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from embedding_ocr.config import DetectorSettings
from embedding_ocr.pipeline.types import TextRegion

_LOGGER = logging.getLogger(__name__)


class TextDetector(Protocol):
    """Protocol for text detection backends."""

    name: str

    def detect(self, image: np.ndarray) -> list[TextRegion]:
        """Detect text regions in reading order."""
        ...


def _box_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    left = max(ax, bx)
    top = max(ay, by)
    right = min(ax + aw, bx + bw)
    bottom = min(ay + ah, by + bh)
    intersection = max(0, right - left) * max(0, bottom - top)
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0 else 0.0


def _dedupe_boxes(
    boxes: list[tuple[int, int, int, int]], iou_threshold: float = 0.6
) -> list[tuple[int, int, int, int]]:
    kept: list[tuple[int, int, int, int]] = []
    for box in sorted(boxes, key=lambda item: item[2] * item[3], reverse=True):
        if any(_box_iou(box, other) >= iou_threshold for other in kept):
            continue
        kept.append(box)
    return kept


def order_boxes(
    boxes: list[tuple[int, int, int, int]], rtl: bool = False
) -> list[tuple[int, int, int, int]]:
    """Group boxes into visual lines, then order each line by direction."""
    lines: list[list[tuple[int, int, int, int]]] = []
    for box in sorted(boxes, key=lambda item: (item[1] + item[3] / 2.0, item[0])):
        top = float(box[1])
        bottom = top + float(box[3])
        best_index: int | None = None
        best_overlap = 0.0
        for index, line in enumerate(lines):
            line_top = min(item[1] for item in line)
            line_bottom = max(item[1] + item[3] for item in line)
            overlap = max(
                0.0, min(bottom, float(line_bottom)) - max(top, float(line_top))
            )
            min_height = min(float(box[3]), float(line_bottom - line_top))
            ratio = overlap / min_height if min_height > 0 else 0.0
            if ratio >= 0.30 and ratio > best_overlap:
                best_index = index
                best_overlap = ratio
        if best_index is None:
            lines.append([box])
        else:
            lines[best_index].append(box)

    lines.sort(
        key=lambda line: (min(item[1] for item in line), min(item[0] for item in line))
    )
    ordered: list[tuple[int, int, int, int]] = []
    for line in lines:
        ordered.extend(sorted(line, key=lambda item: item[0], reverse=rtl))
    return ordered


class FullImageDetector:
    """Fallback detector that treats the whole image as one region."""

    name = "full"

    def detect(self, image: np.ndarray) -> list[TextRegion]:
        height, width = image.shape[:2]
        return [TextRegion(bbox=(0, 0, width, height))]


class VisualWordDetector:
    """Word detector based on connected components and whitespace gaps."""

    name = "visual"

    def __init__(self, rtl: bool = False, min_component_area: int = 8) -> None:
        from embedding_ocr.recognition.embedding.hierarchical.segmentation import (
            VisualWordSegmenter,
        )

        self.rtl = rtl
        self._segmenter = VisualWordSegmenter(min_component_area=min_component_area)

    def detect(self, image: np.ndarray) -> list[TextRegion]:
        boxes = self._segmenter.extract_word_boxes(image, rtl=self.rtl)
        return [TextRegion(bbox=box) for box in boxes]


class VisualLineDetector:
    """Dependency-free line detector with line-aware component grouping."""

    name = "lines"

    def __init__(self, rtl: bool = False, min_component_area: int = 8) -> None:
        from embedding_ocr.recognition.embedding.hierarchical.segmentation import (
            VisualWordSegmenter,
        )

        self.rtl = rtl
        self._segmenter = VisualWordSegmenter(min_component_area=min_component_area)

    def detect(self, image: np.ndarray) -> list[TextRegion]:
        boxes = self._segmenter.extract_line_boxes(image, rtl=self.rtl)
        return [TextRegion(bbox=box) for box in boxes]


class CraftWordDetectorAdapter:
    """Adapter around the optional CRAFT word detector."""

    name = "craft"

    def __init__(
        self,
        rtl: bool = False,
        cuda: bool = False,
        box_padding: int = 2,
    ) -> None:
        from importlib.util import find_spec

        if find_spec("craft_text_detector") is None:
            msg = "craft_text_detector is not installed"
            raise ImportError(msg)

        from embedding_ocr.recognition.embedding.hierarchical.craft_detector import (
            CraftWordDetector,
        )

        self.rtl = rtl
        self.box_padding = box_padding
        self._detector = CraftWordDetector(
            cuda=cuda,
            fallback_full_image=False,
        )

    def detect(self, image: np.ndarray) -> list[TextRegion]:
        height, width = image.shape[:2]
        boxes = self._detector.detect_word_boxes(image, rtl=self.rtl)
        padded: list[tuple[int, int, int, int]] = []
        for x, y, box_width, box_height in boxes:
            x0 = max(0, x - self.box_padding)
            y0 = max(0, y - self.box_padding)
            x1 = min(width, x + box_width + self.box_padding)
            y1 = min(height, y + box_height + self.box_padding)
            if x1 - x0 > 1 and y1 - y0 > 1:
                padded.append((x0, y0, x1 - x0, y1 - y0))
        ordered = order_boxes(_dedupe_boxes(padded), rtl=self.rtl)
        return [TextRegion(bbox=box) for box in ordered]


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value)


class YoloWordDetector:
    """Word-box detector backed by an Ultralytics YOLO checkpoint."""

    name = "yolo_words"

    def __init__(
        self,
        model_path: str | Path | None = None,
        model: Any | None = None,
        conf: float = 0.25,
        iou: float = 0.5,
        imgsz: int = 1280,
        device: str | None = None,
        max_det: int = 1000,
        rtl: bool = False,
        box_padding: int = 2,
        min_box_side: float = 3.0,
        color_order: str = "rgb",
    ) -> None:
        if model is None:
            if model_path is None:
                msg = "model_path is required when model is not provided"
                raise ValueError(msg)
            try:
                from ultralytics import YOLO  # type: ignore[import-untyped]
            except ImportError as exc:
                msg = (
                    "ultralytics is not installed. Install it with "
                    "`uv pip install --python .venv/bin/python -e '.[train]'`."
                )
                raise ImportError(msg) from exc

            self.model_path: Path | None = Path(model_path)
            if not self.model_path.exists():
                msg = f"YOLO word model not found: {self.model_path}"
                raise FileNotFoundError(msg)
            model = YOLO(str(self.model_path))
        else:
            self.model_path = Path(model_path) if model_path is not None else None
        self._model = model
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.device = device
        self.max_det = max_det
        self.rtl = rtl
        self.box_padding = box_padding
        self.min_box_side = min_box_side
        self.color_order = color_order

    @staticmethod
    def _to_bgr(image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return np.repeat(image[:, :, None], 3, axis=2)
        if image.shape[2] == 4:
            return image[:, :, :3]
        return image[:, :, ::-1]

    def detect(self, image: np.ndarray) -> list[TextRegion]:
        """Detect word boxes and return them in reading order."""
        if image is None or image.size == 0:
            return []
        source = self._to_bgr(image) if self.color_order == "rgb" else image
        results: Any = self._model.predict(
            source=np.ascontiguousarray(source),
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
        confidences = _as_numpy(boxes.conf).reshape(-1)
        scores: dict[tuple[int, int, int, int], float] = {}
        for corners, confidence in zip(xyxy, confidences, strict=False):
            x0 = max(0.0, min(float(width), float(corners[0])))
            y0 = max(0.0, min(float(height), float(corners[1])))
            x1 = max(0.0, min(float(width), float(corners[2])))
            y1 = max(0.0, min(float(height), float(corners[3])))
            if x1 - x0 < self.min_box_side or y1 - y0 < self.min_box_side:
                continue
            pad = float(self.box_padding)
            px0 = max(0, int(round(x0 - pad)))
            py0 = max(0, int(round(y0 - pad)))
            px1 = min(width, int(round(x1 + pad)))
            py1 = min(height, int(round(y1 + pad)))
            box = (px0, py0, px1 - px0, py1 - py0)
            if box[2] <= 1 or box[3] <= 1:
                continue
            scores[box] = max(float(confidence), scores.get(box, 0.0))

        ordered = order_boxes(_dedupe_boxes(list(scores)), rtl=self.rtl)
        return [
            TextRegion(bbox=box, confidence=scores.get(box, 1.0)) for box in ordered
        ]


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


def _payload(result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        payload: Any = result
    else:
        raw = getattr(result, "json", None)
        if callable(raw):
            raw = raw()
        if isinstance(raw, str):
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                payload = {}
        else:
            payload = raw
    if not isinstance(payload, dict):
        return {}
    nested = payload.get("res")
    if isinstance(nested, dict):
        return nested
    return payload


def _iter_polys(result: Any) -> list[Any]:
    payload = _payload(result)
    for key in ("dt_polys", "rec_polys", "boxes"):
        value = payload.get(key)
        if value is not None:
            return list(value)
    return []


def _iter_scores(result: Any, count: int) -> list[float]:
    payload = _payload(result)
    raw_scores = payload.get("dt_scores")
    if raw_scores is None:
        return [1.0] * count
    try:
        scores = [float(value) for value in raw_scores]
    except (TypeError, ValueError):
        return [1.0] * count
    if len(scores) < count:
        scores.extend([1.0] * (count - len(scores)))
    return scores[:count]


def _resolve_paddle_device(device: str) -> str:
    if device == "cpu":
        return "cpu"
    if device.startswith("cuda:"):
        return device.replace("cuda:", "gpu:", 1)
    if device == "cuda":
        return "gpu"
    if device == "auto":
        try:
            import paddle

            if (
                paddle.device.is_compiled_with_cuda()
                and paddle.device.cuda.device_count() > 0
            ):
                from embedding_ocr.utils.gpu import resolve_device

                resolved = resolve_device("auto")
                if resolved.startswith("cuda:"):
                    return resolved.replace("cuda:", "gpu:", 1)
                return "gpu"
        except Exception:
            return "cpu"
        return "cpu"
    return device


class PaddleTextDetector:
    """PaddleOCR text-detection backend.

    Uses the standalone ``TextDetection`` module when available so recognition
    models are not loaded and detection can run independently. Older PaddleOCR
    releases fall back to the legacy detection-only ``ocr`` call.
    """

    name = "paddle"

    def __init__(
        self,
        lang: str = "en",
        rtl: bool = False,
        model_name: str = "PP-OCRv5_server_det",
        device: str = "auto",
        min_confidence: float = 0.3,
        box_padding: int = 2,
    ) -> None:
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        self.rtl = rtl
        self.model_name = model_name
        self.min_confidence = min_confidence
        self.box_padding = box_padding
        self._backend = "text_detection"
        self._detector: Any = None
        self._legacy_ocr: Any = None

        try:
            from paddleocr import TextDetection  # type: ignore[import-untyped]
        except ImportError:
            TextDetection = None  # type: ignore[assignment,misc]

        if TextDetection is not None:
            paddle_device = _resolve_paddle_device(device)
            try:
                self._detector = TextDetection(
                    model_name=model_name,
                    device=paddle_device,
                )
            except TypeError:
                self._detector = TextDetection(model_name=model_name)
            return

        try:
            from paddleocr import PaddleOCR  # type: ignore[import-untyped]
        except ImportError as exc:
            msg = (
                "paddleocr is not installed. Install it with "
                "`uv pip install --python .venv/bin/python -e '.[baseline]'` "
                "or run `bash scripts/setup_auto.sh --with-paddle`."
            )
            raise ImportError(msg) from exc

        self._backend = "legacy"
        try:
            self._legacy_ocr = PaddleOCR(
                lang=lang,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )
        except TypeError:
            self._legacy_ocr = PaddleOCR(lang=lang)

    @staticmethod
    def _iter_polys(result: Any) -> list[Any]:
        return _iter_polys(result)

    @staticmethod
    def _iter_scores(result: Any, count: int) -> list[float]:
        return _iter_scores(result, count)

    def _raw_results(self, image: np.ndarray) -> list[Any]:
        if self._backend == "text_detection":
            return list(self._detector.predict(image, batch_size=1))
        if hasattr(self._legacy_ocr, "predict"):
            return list(self._legacy_ocr.predict(image))
        return [self._legacy_ocr.ocr(image, det=True, rec=False)]

    def detect(self, image: np.ndarray) -> list[TextRegion]:
        try:
            raw_results = self._raw_results(image)
        except NotImplementedError as exc:
            _LOGGER.warning(
                "Paddle detector failed in its oneDNN/PIR runtime (%s); "
                "falling back to a visual line detector.",
                exc,
            )
            return VisualLineDetector(rtl=self.rtl).detect(image)
        except Exception as exc:
            _LOGGER.warning(
                "Paddle detection failed (%s); falling back to the visual line detector.",
                exc,
            )
            return VisualLineDetector(rtl=self.rtl).detect(image)

        height, width = image.shape[:2]
        boxes: list[tuple[int, int, int, int]] = []
        for raw in raw_results:
            polys = _iter_polys(raw)
            scores = _iter_scores(raw, len(polys))
            for poly, score in zip(polys, scores, strict=False):
                if score < self.min_confidence:
                    continue
                bbox = _poly_to_bbox(poly, width, height)
                x, y, box_width, box_height = bbox
                if box_width <= 2 or box_height <= 2:
                    continue
                x0 = max(0, x - self.box_padding)
                y0 = max(0, y - self.box_padding)
                x1 = min(width, x + box_width + self.box_padding)
                y1 = min(height, y + box_height + self.box_padding)
                boxes.append((x0, y0, x1 - x0, y1 - y0))

        ordered = order_boxes(_dedupe_boxes(boxes), rtl=self.rtl)
        return [TextRegion(bbox=box) for box in ordered]


class AutoTextDetector:
    """Choose the strongest available detector, degrading gracefully."""

    name = "auto"

    def __init__(self, settings: DetectorSettings, rtl: bool = False) -> None:
        self.settings = settings
        self.rtl = rtl
        self._detector: TextDetector | None = None
        self._resolved_name = "unresolved"

    @property
    def resolved_name(self) -> str:
        return self._resolved_name

    def _build(self) -> TextDetector:
        attempts: list[tuple[str, Any]] = [
            (
                "paddle",
                lambda: PaddleTextDetector(
                    lang=self.settings.lang,
                    rtl=self.rtl,
                    model_name=self.settings.paddle_model,
                    device=self.settings.device,
                    min_confidence=self.settings.min_confidence,
                    box_padding=self.settings.box_padding,
                ),
            ),
            (
                "craft",
                lambda: CraftWordDetectorAdapter(
                    rtl=self.rtl,
                    cuda=self.settings.craft_cuda,
                    box_padding=self.settings.box_padding,
                ),
            ),
            (
                "lines",
                lambda: VisualLineDetector(
                    rtl=self.rtl,
                    min_component_area=self.settings.min_component_area,
                ),
            ),
        ]
        errors: list[str] = []
        for name, factory in attempts:
            try:
                detector = factory()
            except Exception as exc:
                errors.append(f"{name}: {exc}")
                _LOGGER.warning("Detector %s unavailable: %s", name, exc)
                continue
            self._resolved_name = name
            _LOGGER.info("Auto detector selected %s", name)
            return detector
        msg = "no detector backend available: " + "; ".join(errors)
        raise RuntimeError(msg)

    def detect(self, image: np.ndarray) -> list[TextRegion]:
        if self._detector is None:
            self._detector = self._build()
        return self._detector.detect(image)


def build_detector(settings: DetectorSettings, rtl: bool = False) -> TextDetector:
    """Create a detector from configuration.

    When ``detector.line_grouping`` is set, the configured word detector is
    wrapped so it returns one box per text line, which is what the CTC
    recognizer was trained on.
    """
    detector = _build_word_detector(settings, rtl=rtl)
    if not settings.line_grouping:
        return detector
    from embedding_ocr.pipeline.lines import LineGroupingConfig, LineGroupingDetector

    resolved_rtl = settings.rtl if settings.rtl is not None else rtl
    return LineGroupingDetector(
        detector,
        LineGroupingConfig(
            rtl=resolved_rtl,
            vertical_overlap=settings.line_vertical_overlap,
            padding=settings.line_padding,
        ),
    )


def _build_word_detector(settings: DetectorSettings, rtl: bool = False) -> TextDetector:
    """Create the configured word/line detector without grouping."""
    backend = settings.backend.lower()
    resolved_rtl = settings.rtl if settings.rtl is not None else rtl
    if backend == "full":
        return FullImageDetector()
    if backend == "auto":
        return AutoTextDetector(settings, rtl=resolved_rtl)
    if backend == "visual":
        return VisualWordDetector(
            rtl=resolved_rtl, min_component_area=settings.min_component_area
        )
    if backend in {"lines", "line"}:
        return VisualLineDetector(
            rtl=resolved_rtl, min_component_area=settings.min_component_area
        )
    if backend == "craft":
        return CraftWordDetectorAdapter(
            rtl=resolved_rtl,
            cuda=settings.craft_cuda,
            box_padding=settings.box_padding,
        )
    if backend in {"yolo_words", "yolo_word"}:
        if settings.yolo_model is None:
            msg = "detector.yolo_model must point to a YOLO word checkpoint"
            raise ValueError(msg)
        return YoloWordDetector(
            model_path=settings.yolo_model,
            conf=settings.yolo_conf,
            iou=settings.yolo_iou,
            imgsz=settings.yolo_imgsz,
            device=settings.yolo_device,
            max_det=settings.yolo_max_det,
            rtl=resolved_rtl,
            box_padding=settings.box_padding,
        )
    if backend in {"paddle", "paddle_mobile"}:
        model_name = (
            settings.paddle_mobile_model
            if backend == "paddle_mobile"
            else settings.paddle_model
        )
        return PaddleTextDetector(
            lang=settings.lang,
            rtl=resolved_rtl,
            model_name=model_name,
            device=settings.device,
            min_confidence=settings.min_confidence,
            box_padding=settings.box_padding,
        )
    msg = f"Unknown detector backend: {settings.backend!r}"
    raise ValueError(msg)
