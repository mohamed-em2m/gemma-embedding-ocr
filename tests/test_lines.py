"""Tests for grouping detected word boxes into line boxes."""

from __future__ import annotations

import numpy as np
import pytest

from embedding_ocr.pipeline.lines import (
    LineGroupingConfig,
    LineGroupingDetector,
    group_word_regions,
    merge_line_box,
    word_boxes_to_lines,
)
from embedding_ocr.pipeline.types import TextRegion


def _box(x: int, y: int, w: int, h: int, confidence: float = 1.0) -> TextRegion:
    return TextRegion(bbox=(x, y, w, h), confidence=confidence)


def _two_lines() -> list[TextRegion]:
    """Two lines of three words each, given out of reading order."""
    return [
        _box(140, 60, 40, 20),  # second line, third word
        _box(10, 10, 50, 20),  # first line, first word
        _box(80, 58, 50, 22),  # second line, second word
        _box(70, 12, 60, 18),  # first line, second word
        _box(10, 60, 60, 20),  # second line, first word
        _box(150, 8, 30, 24),  # first line, third word
    ]


# --- grouping ----------------------------------------------------------------


def test_grouping_splits_two_lines() -> None:
    lines = group_word_regions(_two_lines())
    assert len(lines) == 2
    assert [[region.bbox[0] for region in line] for line in lines] == [
        [10, 70, 150],
        [10, 80, 140],
    ]


def test_grouping_orders_lines_top_to_bottom() -> None:
    lines = group_word_regions(_two_lines())
    tops = [min(region.bbox[1] for region in line) for line in lines]
    assert tops == sorted(tops)


def test_grouping_keeps_touching_words_on_one_line() -> None:
    regions = [_box(10, 10, 40, 20), _box(50, 10, 40, 20), _box(90, 10, 40, 20)]
    lines = group_word_regions(regions)
    assert len(lines) == 1
    assert [region.bbox[0] for region in lines[0]] == [10, 50, 90]


def test_grouping_handles_single_word() -> None:
    lines = group_word_regions([_box(5, 5, 20, 12)])
    assert len(lines) == 1
    assert lines[0] == [_box(5, 5, 20, 12)]


def test_grouping_handles_empty_input() -> None:
    assert group_word_regions([]) == []


def test_grouping_skips_degenerate_boxes() -> None:
    regions = [_box(0, 0, 0, 20), _box(0, 0, 20, 0), _box(10, 10, 20, 20)]
    lines = group_word_regions(regions)
    assert len(lines) == 1
    assert lines[0] == [_box(10, 10, 20, 20)]


def test_grouping_sorts_rtl_lines_right_to_left() -> None:
    regions = [_box(10, 10, 40, 20), _box(60, 10, 40, 20), _box(110, 10, 40, 20)]
    lines = group_word_regions(regions, LineGroupingConfig(rtl=True))
    assert [region.bbox[0] for region in lines[0]] == [110, 60, 10]


def test_grouping_separates_lines_that_do_not_overlap() -> None:
    regions = [_box(10, 10, 40, 20), _box(10, 40, 40, 20)]
    assert len(group_word_regions(regions)) == 2


def test_grouping_rejects_bad_overlap() -> None:
    for value in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="vertical_overlap"):
            LineGroupingConfig(vertical_overlap=value)


def test_grouping_rejects_negative_padding() -> None:
    with pytest.raises(ValueError, match="padding"):
        LineGroupingConfig(padding=-1)


# --- merging -----------------------------------------------------------------


def test_merge_spans_every_member() -> None:
    line = [_box(10, 10, 40, 20), _box(80, 12, 30, 18)]
    merged = merge_line_box(line)
    assert merged.bbox == (10, 10, 100, 20)


def test_merge_applies_padding() -> None:
    merged = merge_line_box([_box(10, 10, 40, 20)], padding=3)
    assert merged.bbox == (7, 7, 46, 26)


def test_merge_averages_confidence() -> None:
    line = [_box(0, 0, 10, 10, confidence=0.4), _box(20, 0, 10, 10, confidence=0.8)]
    assert merge_line_box(line).confidence == pytest.approx(0.6)


def test_merge_rejects_empty_line() -> None:
    with pytest.raises(ValueError, match="empty line"):
        merge_line_box([])


def test_word_boxes_to_lines_returns_one_box_per_line() -> None:
    boxes = word_boxes_to_lines(_two_lines())
    assert len(boxes) == 2
    assert boxes[0].bbox == (8, 6, 174, 28)  # first line plus 2px padding
    assert boxes[1].bbox == (8, 56, 174, 26)


# --- detector wrapper --------------------------------------------------------


class _StubDetector:
    """Returns fixed word boxes regardless of the image."""

    name = "stub"

    def __init__(self, regions: list[TextRegion]) -> None:
        self.regions = regions

    def detect(self, image: np.ndarray) -> list[TextRegion]:
        return list(self.regions)


def test_detector_wrapper_merges_detected_words() -> None:
    detector = LineGroupingDetector(_StubDetector(_two_lines()))
    lines = detector.detect(np.zeros((100, 200, 3), dtype=np.uint8))
    assert len(lines) == 2
    assert all(isinstance(line, TextRegion) for line in lines)


def test_detector_wrapper_name_reports_wrapped_backend() -> None:
    detector = LineGroupingDetector(_StubDetector([]))
    assert detector.name == "line_grouping(_StubDetector)"


def test_detector_wrapper_passes_config_through() -> None:
    detector = LineGroupingDetector(
        _StubDetector([_box(10, 10, 40, 20)]), LineGroupingConfig(padding=5)
    )
    assert detector.detect(np.zeros((40, 80), dtype=np.uint8))[0].bbox == (
        5,
        5,
        50,
        30,
    )
