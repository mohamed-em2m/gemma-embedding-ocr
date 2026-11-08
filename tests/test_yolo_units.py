"""Tests for YOLO unit parsing, line ordering, and text reconstruction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from embedding_ocr.pipeline.types import DetectedUnit
from embedding_ocr.recognition.base import RecognitionResult
from embedding_ocr.recognition.yolo_units import (
    YoloUnitDetector,
    YoloUnitRecognizer,
    group_units_by_line,
    load_classes,
    units_to_text,
)


class FakeBoxes:
    def __init__(self, xyxy: list[list[float]], cls: list[int], conf: list[float]):
        self.xyxy = np.asarray(xyxy, dtype=np.float32)
        self.cls = np.asarray(cls, dtype=np.float32)
        self.conf = np.asarray(conf, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.xyxy)


class FakeResult:
    def __init__(self, boxes: FakeBoxes, names: dict[int, str]):
        self.boxes = boxes
        self.names = names


class FakeModel:
    def __init__(self, result: FakeResult):
        self.result = result
        self.names = result.names

    def predict(self, **_kwargs: Any) -> list[FakeResult]:
        return [self.result]


class FakeDetector:
    def __init__(self, units: list[DetectedUnit]):
        self.units = units

    def detect(self, _image: np.ndarray) -> list[DetectedUnit]:
        return self.units


def _unit(x: int, y: int, text: str, width: int = 20, height: int = 20) -> DetectedUnit:
    return DetectedUnit(
        bbox=(x, y, width, height),
        text=text,
        confidence=1.0,
        script="arabic" if text in "مرحباكتب" else "latin",
    )


def test_load_classes(tmp_path: Path) -> None:
    path = tmp_path / "classes.json"
    path.write_text(json.dumps({"classes": ["a", "ب"]}), encoding="utf-8")

    assert load_classes(path) == ["a", "ب"]
    assert load_classes(None) is None


def test_detector_parses_ultralytics_boxes() -> None:
    result = FakeResult(
        boxes=FakeBoxes(
            xyxy=[[10, 10, 30, 30], [40, 10, 60, 30]],
            cls=[0, 1],
            conf=[0.9, 0.8],
        ),
        names={0: "A", 1: "ب"},
    )
    detector = YoloUnitDetector(model=FakeModel(result), classes=None)

    units = detector.detect(np.zeros((100, 100, 3), dtype=np.uint8))

    assert [(unit.text, unit.bbox) for unit in units] == [
        ("A", (10, 10, 20, 20)),
        ("ب", (40, 10, 20, 20)),
    ]
    assert units[0].script == "latin"
    assert units[1].script == "arabic"


def test_line_grouping_and_ltr_text() -> None:
    units = [
        _unit(30, 0, "i"),
        _unit(0, 0, "H"),
        _unit(0, 40, "a"),
    ]

    lines = group_units_by_line(units, rtl=False)

    assert [[unit.text for unit in line] for line in lines] == [["H", "i"], ["a"]]
    assert units_to_text(lines) == "Hi\na"


def test_rtl_line_order_and_spacing() -> None:
    units = [
        _unit(60, 0, "ر"),
        _unit(80, 0, "م"),
    ]

    lines = group_units_by_line(units, rtl=None)

    assert [[unit.text for unit in line] for line in lines] == [["م", "ر"]]
    assert units_to_text(lines) == "مر"


def test_gap_inserts_word_space() -> None:
    units = [
        _unit(0, 0, "H", width=20),
        _unit(80, 0, "i", width=20),
    ]

    lines = group_units_by_line(units, rtl=False)

    assert units_to_text(lines) == "H i"


def test_recognizer_returns_text_and_mean_confidence() -> None:
    detector = FakeDetector(
        [
            DetectedUnit((0, 0, 20, 20), "H", 0.8, "latin"),
            DetectedUnit((25, 0, 20, 20), "i", 0.6, "latin"),
        ]
    )
    recognizer = YoloUnitRecognizer(detector=detector, rtl=False)

    result = recognizer.recognize(np.zeros((40, 80, 3), dtype=np.uint8))

    assert isinstance(result, RecognitionResult)
    assert result.text == "Hi"
    assert result.confidence == 0.7
