"""Group detected word boxes into line boxes.

Word detectors return one box per word. The line recognizer (CTC) expects a
whole text line, because it was trained on line crops and uses the space
between words as a cue. This module bridges the two: it clusters word boxes
into lines by vertical overlap, merges each cluster into a single box, and
exposes the result as a detector so the pipeline needs no special case.

Merging is lossless for reading order: the recognizer emits the whole line in
order, and one line box per line keeps the pipeline's existing
``reconstruct_text`` ordering intact.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean

import numpy as np

from embedding_ocr.pipeline.detect import TextDetector
from embedding_ocr.pipeline.types import TextRegion


@dataclass(frozen=True)
class LineGroupingConfig:
    """How word boxes are clustered into lines.

    Attributes:
        rtl: Right-to-left script. Ordering is irrelevant after merging, but
            it is kept so callers can inspect the grouping.
        vertical_overlap: Minimum overlap, as a fraction of the shorter box
            height, for two boxes to belong to the same line.
        padding: Pixels added around each merged line box.
    """

    rtl: bool = False
    vertical_overlap: float = 0.5
    padding: int = 2

    def __post_init__(self) -> None:
        if not 0.0 < self.vertical_overlap <= 1.0:
            message = "vertical_overlap must be in (0, 1]"
            raise ValueError(message)
        if self.padding < 0:
            message = "padding must be non-negative"
            raise ValueError(message)


def _overlap(a: TextRegion, b: TextRegion) -> float:
    _, ay, _, ah = a.bbox
    _, by, _, bh = b.bbox
    top = max(ay, by)
    bottom = min(ay + ah, by + bh)
    return max(0.0, float(bottom - top))


def _centre_y(region: TextRegion) -> float:
    return region.bbox[1] + region.bbox[3] / 2.0


def group_word_regions(
    regions: list[TextRegion], config: LineGroupingConfig | None = None
) -> list[list[TextRegion]]:
    """Cluster word boxes into lines.

    Boxes are visited top-to-bottom and assigned to the existing line with the
    largest vertical overlap, provided that overlap is at least
    ``vertical_overlap`` of the shorter box. Lines and their members are
    returned in reading order.

    Args:
        regions: Word boxes in image coordinates.
        config: Grouping thresholds.

    Returns:
        One list of regions per line, each sorted left-to-right (or
        right-to-left when ``config.rtl`` is set).
    """
    settings = config or LineGroupingConfig()
    lines: list[list[TextRegion]] = []
    for region in sorted(regions, key=lambda item: (_centre_y(item), item.bbox[0])):
        if region.bbox[3] <= 0 or region.bbox[2] <= 0:
            continue
        best: list[TextRegion] | None = None
        best_overlap = 0.0
        for line in lines:
            overlap = max(_overlap(region, member) for member in line)
            shortest = min(region.bbox[3], min(member.bbox[3] for member in line))
            threshold = settings.vertical_overlap * max(1.0, float(shortest))
            if overlap >= threshold and overlap > best_overlap:
                best_overlap = overlap
                best = line
        if best is None:
            lines.append([region])
        else:
            best.append(region)

    ordered: list[list[TextRegion]] = []
    for line in lines:
        ordered.append(
            sorted(line, key=lambda item: item.bbox[0], reverse=settings.rtl)
        )
    ordered.sort(key=lambda line: (_centre_y(line[0]), line[0].bbox[0]))
    return ordered


def merge_line_box(line: list[TextRegion], *, padding: int = 0) -> TextRegion:
    """Merge a line's word boxes into one box spanning the whole line.

    Args:
        line: Word boxes belonging to one line, in reading order.
        padding: Pixels added on each side.

    Returns:
        A box covering every member, with the mean member confidence.

    Raises:
        ValueError: If ``line`` is empty.
    """
    if not line:
        message = "cannot merge an empty line"
        raise ValueError(message)
    x0 = min(region.bbox[0] for region in line) - padding
    y0 = min(region.bbox[1] for region in line) - padding
    x1 = max(region.bbox[0] + region.bbox[2] for region in line) + padding
    y1 = max(region.bbox[1] + region.bbox[3] for region in line) + padding
    return TextRegion(
        bbox=(int(x0), int(y0), int(max(1, x1 - x0)), int(max(1, y1 - y0))),
        confidence=float(mean(region.confidence for region in line)),
    )


def word_boxes_to_lines(
    regions: list[TextRegion], config: LineGroupingConfig | None = None
) -> list[TextRegion]:
    """Group word boxes and merge each line into a single box."""
    settings = config or LineGroupingConfig()
    return [
        merge_line_box(line, padding=settings.padding)
        for line in group_word_regions(regions, settings)
    ]


class LineGroupingDetector:
    """Wrap a word detector so it returns one box per line.

    Attributes:
        detector: The underlying word detector.
        config: Grouping thresholds.
    """

    def __init__(
        self, detector: TextDetector, config: LineGroupingConfig | None = None
    ) -> None:
        self.detector = detector
        self.config = config or LineGroupingConfig()
        self.name = f"line_grouping({detector.__class__.__name__})"

    def detect(self, image: np.ndarray) -> list[TextRegion]:
        """Detect words, then merge them into line boxes."""
        return word_boxes_to_lines(self.detector.detect(image), self.config)
