"""Tests for detection, cropping, reconstruction, and pipeline orchestration."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from embedding_ocr.config import PipelineSettings
from embedding_ocr.pipeline import (
    OCRPipeline,
    RecognizedRegion,
    TextRegion,
    crop_regions,
    group_lines,
    reconstruct_text,
)
from embedding_ocr.recognition.base import RecognitionResult


class TwoRegionDetector:
    def __init__(self) -> None:
        self.calls = 0

    def detect(self, image: np.ndarray) -> list[TextRegion]:
        self.calls += 1
        return [TextRegion((0, 0, 10, 20)), TextRegion((20, 0, 10, 20))]


class SequenceRecognizer:
    def __init__(self) -> None:
        self.index = 0

    def recognize(self, image: np.ndarray) -> RecognitionResult:
        labels = ["A", "B"]
        text = labels[self.index % len(labels)]
        self.index += 1
        return RecognitionResult(text=text, confidence=0.9)


def test_crop_regions_clamps_and_pads() -> None:
    image = np.zeros((20, 30, 3), dtype=np.uint8)
    crops = crop_regions(
        image,
        [TextRegion((0, 0, 10, 10)), TextRegion((25, 15, 10, 10))],
        padding=5,
    )
    assert crops[0].shape == (15, 15, 3)
    assert crops[1].shape == (10, 10, 3)


def test_reconstruct_text_ltr_and_rtl() -> None:
    regions = [
        RecognizedRegion(TextRegion((0, 0, 10, 10)), "left", 1.0),
        RecognizedRegion(TextRegion((20, 0, 10, 10)), "right", 1.0),
    ]
    assert reconstruct_text(regions) == "left right"
    assert reconstruct_text(regions, rtl=True) == "right left"


def test_group_lines_keeps_rows_separate() -> None:
    regions = [
        RecognizedRegion(TextRegion((0, 0, 10, 10)), "top-left", 1.0),
        RecognizedRegion(TextRegion((20, 0, 10, 10)), "top-right", 1.0),
        RecognizedRegion(TextRegion((0, 30, 10, 10)), "bottom", 1.0),
    ]
    groups = group_lines(regions)
    assert len(groups) == 2
    assert [item.text for item in groups[0]] == ["top-left", "top-right"]
    assert reconstruct_text(regions) == "top-left top-right\nbottom"


def test_pipeline_recognizes_and_joins_regions() -> None:
    pipeline = OCRPipeline(
        detector=TwoRegionDetector(),
        recognizer=SequenceRecognizer(),
        settings=PipelineSettings(preprocess=False),
    )
    result = pipeline.recognize(np.zeros((20, 40, 3), dtype=np.uint8))

    assert result.text == "A B"
    assert abs(result.confidence - 0.9) < 1e-9
    assert result.metadata["num_regions"] == 2


class WordBoxRecognizer:
    def __init__(self) -> None:
        self.index = 0

    def recognize(self, image: np.ndarray) -> RecognitionResult:
        index = self.index
        self.index += 1
        return RecognitionResult(
            text=f"W{index}",
            confidence=0.8,
            metadata={
                "word_boxes": [
                    {"bbox": [1, 2, 3, 4], "text": f"W{index}", "confidence": 0.8}
                ]
            },
        )


def test_pipeline_offsets_word_boxes_to_image_coordinates() -> None:
    pipeline = OCRPipeline(
        detector=TwoRegionDetector(),
        recognizer=WordBoxRecognizer(),
        settings=PipelineSettings(preprocess=False),
    )
    result = pipeline.recognize(np.zeros((20, 40, 3), dtype=np.uint8))

    boxes = result.metadata["word_boxes"]
    assert [box["bbox"] for box in boxes] == [[1, 2, 3, 4], [21, 2, 3, 4]]
    assert [box["region_index"] for box in boxes] == [1, 2]
    assert [box["text"] for box in boxes] == ["W0", "W1"]


class ClampedRegionDetector:
    def detect(self, image: np.ndarray) -> list[TextRegion]:
        return [TextRegion((-5, 0, 10, 20))]


def test_pipeline_offsets_word_boxes_from_clamped_crop_origin() -> None:
    pipeline = OCRPipeline(
        detector=ClampedRegionDetector(),
        recognizer=WordBoxRecognizer(),
        settings=PipelineSettings(preprocess=False),
    )
    result = pipeline.recognize(np.zeros((20, 40, 3), dtype=np.uint8))

    boxes = result.metadata["word_boxes"]
    assert boxes[0]["bbox"] == [1, 2, 3, 4]


def test_pipeline_rtl_reading_order() -> None:
    pipeline = OCRPipeline(
        detector=TwoRegionDetector(),
        recognizer=SequenceRecognizer(),
        settings=PipelineSettings(preprocess=False),
        rtl=True,
    )
    result = pipeline.recognize(np.zeros((20, 40, 3), dtype=np.uint8))
    assert result.text == "B A"


def test_detection_cache_reuses_boxes(tmp_path: Path) -> None:
    detector = TwoRegionDetector()
    pipeline = OCRPipeline(
        detector=detector,
        recognizer=SequenceRecognizer(),
        settings=PipelineSettings(preprocess=False, detection_cache=tmp_path / "cache"),
    )
    image = np.zeros((20, 40, 3), dtype=np.uint8)

    first = pipeline.recognize(image)
    second = pipeline.recognize(image)

    assert detector.calls == 1
    assert first.text == second.text
    assert second.metadata["detection_cached"] is True


def test_empty_image_returns_empty_result() -> None:
    pipeline = OCRPipeline(
        detector=TwoRegionDetector(),
        recognizer=SequenceRecognizer(),
        settings=PipelineSettings(preprocess=False),
    )
    result = pipeline.recognize(np.zeros((0, 0, 3), dtype=np.uint8))
    assert result.text == ""
    assert result.confidence == 0.0
