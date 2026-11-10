"""Tests for line-aware segmentation and detection box metrics."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from embedding_ocr.eval.detection import (
    aggregate_metrics,
    detection_metrics,
    match_boxes,
)
from embedding_ocr.pipeline.detect import order_boxes
from embedding_ocr.recognition.embedding.hierarchical.segmentation import (
    VisualWordSegmenter,
)


def _two_line_page() -> np.ndarray:
    image = Image.new("RGB", (300, 140), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.rectangle([10, 10, 59, 39], fill=(0, 0, 0))
    draw.rectangle([80, 10, 129, 39], fill=(0, 0, 0))
    draw.rectangle([10, 80, 59, 109], fill=(0, 0, 0))
    draw.rectangle([80, 80, 129, 109], fill=(0, 0, 0))
    return np.array(image, dtype=np.uint8)


def test_word_segmenter_orders_lines_before_columns() -> None:
    boxes = VisualWordSegmenter().extract_word_boxes(_two_line_page())

    assert len(boxes) == 4
    assert boxes[0][1] < boxes[2][1]
    assert boxes[1][1] < boxes[3][1]
    assert boxes[0][0] < boxes[1][0]
    assert boxes[2][0] < boxes[3][0]


def test_line_segmenter_returns_one_box_per_line() -> None:
    lines = VisualWordSegmenter().extract_line_boxes(_two_line_page())

    assert len(lines) == 2
    assert lines[0][1] < lines[1][1]
    assert lines[0][0] <= 10
    assert lines[1][0] <= 10


def test_word_segmenter_rtl_orders_each_line_right_to_left() -> None:
    boxes = VisualWordSegmenter().extract_word_boxes(_two_line_page(), rtl=True)

    assert len(boxes) == 4
    assert boxes[0][0] > boxes[1][0]
    assert boxes[2][0] > boxes[3][0]
    assert boxes[0][1] < boxes[2][1]


def test_empty_image_returns_no_boxes() -> None:
    blank = np.full((80, 120, 3), 255, dtype=np.uint8)

    assert VisualWordSegmenter().extract_word_boxes(blank) == []
    assert VisualWordSegmenter().extract_line_boxes(blank) == []


def test_order_boxes_groups_uneven_baselines() -> None:
    boxes = [
        (80, 12, 40, 20),
        (10, 10, 40, 20),
        (10, 80, 40, 20),
        (80, 82, 40, 20),
    ]

    assert order_boxes(boxes) == [
        (10, 10, 40, 20),
        (80, 12, 40, 20),
        (10, 80, 40, 20),
        (80, 82, 40, 20),
    ]


def test_detection_metrics_match_misses_and_false_positives() -> None:
    metrics = detection_metrics(
        predictions=[(0, 0, 10, 10), (30, 0, 10, 10), (90, 90, 5, 5)],
        references=[(0, 0, 10, 10), (30, 0, 10, 10), (60, 0, 10, 10)],
    )

    assert metrics.matched == 2
    assert metrics.missed == 1
    assert metrics.false_positives == 1
    assert metrics.precision == 2 / 3
    assert metrics.recall == 2 / 3


def test_aggregate_metrics_sums_counts() -> None:
    first = detection_metrics([(0, 0, 10, 10)], [(0, 0, 10, 10)])
    second = detection_metrics([(0, 0, 10, 10)], [(20, 0, 10, 10)])

    aggregate = aggregate_metrics([first, second])

    assert aggregate.num_predictions == 2
    assert aggregate.num_references == 2
    assert aggregate.matched == 1
    assert aggregate.f1 == 0.5


def test_match_boxes_is_one_to_one() -> None:
    matches, unmatched_predictions, unmatched_references = match_boxes(
        [(0, 0, 10, 10), (0, 0, 10, 10)],
        [(0, 0, 10, 10)],
    )

    assert len(matches) == 1
    assert len(unmatched_predictions) == 1
    assert unmatched_references == []
