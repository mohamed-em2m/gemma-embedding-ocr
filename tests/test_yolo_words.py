"""Tests for the YOLO word detector adapter."""

from __future__ import annotations

import numpy as np
import pytest

from embedding_ocr.pipeline.detect import YoloWordDetector


class _FakeBoxes:
    def __init__(self, boxes: list[list[float]], confidences: list[float]) -> None:
        self.xyxy = np.asarray(boxes, dtype=np.float32)
        self.conf = np.asarray(confidences, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.conf)


class _FakeResult:
    def __init__(self, boxes: _FakeBoxes) -> None:
        self.boxes = boxes


class _FakeModel:
    def __init__(self, boxes: _FakeBoxes) -> None:
        self.boxes = boxes
        self.kwargs: dict[str, object] = {}

    def predict(self, **kwargs: object) -> list[_FakeResult]:
        self.kwargs = kwargs
        return [_FakeResult(self.boxes)]


def _image() -> np.ndarray:
    return np.zeros((100, 200, 3), dtype=np.uint8)


def test_detector_returns_ordered_word_boxes() -> None:
    model = _FakeModel(
        _FakeBoxes(
            boxes=[
                [80, 10, 140, 40],
                [10, 12, 70, 42],
                [10, 70, 60, 95],
                [1, 1, 2, 2],
            ],
            confidences=[0.9, 0.8, 0.7, 0.99],
        )
    )
    detector = YoloWordDetector(model=model, box_padding=0)

    regions = detector.detect(_image())

    assert [region.bbox for region in regions] == [
        (10, 12, 60, 30),
        (80, 10, 60, 30),
        (10, 70, 50, 25),
    ]
    assert [region.confidence for region in regions] == pytest.approx([0.8, 0.9, 0.7])
    assert model.kwargs["conf"] == 0.25
    assert model.kwargs["imgsz"] == 1280


def test_detector_rtl_reverses_each_line() -> None:
    model = _FakeModel(
        _FakeBoxes(
            boxes=[[10, 10, 40, 30], [60, 10, 90, 30]],
            confidences=[0.5, 0.6],
        )
    )
    detector = YoloWordDetector(model=model, box_padding=0, rtl=True)

    regions = detector.detect(_image())

    assert [region.bbox[0] for region in regions] == [60, 10]


def test_detector_requires_model_or_checkpoint() -> None:
    with pytest.raises(ValueError, match="model_path"):
        YoloWordDetector()


def test_detector_empty_result() -> None:
    detector = YoloWordDetector(model=_FakeModel(_FakeBoxes([], [])))
    assert detector.detect(_image()) == []
