"""Reading-order and text reconstruction helpers."""

from __future__ import annotations

from statistics import mean

from embedding_ocr.pipeline.types import RecognizedRegion, TextRegion


def _vertical_overlap(a: TextRegion, b: TextRegion) -> float:
    _, ay, _, ah = a.bbox
    _, by, _, bh = b.bbox
    top = max(ay, by)
    bottom = min(ay + ah, by + bh)
    return max(0.0, float(bottom - top))


def group_lines(
    regions: list[RecognizedRegion],
    rtl: bool = False,
    line_tolerance: float = 0.5,
) -> list[list[RecognizedRegion]]:
    """Group regions into lines by vertical overlap and order within each line."""
    lines: list[list[RecognizedRegion]] = []
    for item in sorted(regions, key=lambda r: (r.region.bbox[1], r.region.bbox[0])):
        best_line: list[RecognizedRegion] | None = None
        best_overlap = 0.0
        for line in lines:
            overlap = max(
                _vertical_overlap(item.region, other.region) for other in line
            )
            min_height = min(item.region.bbox[3], line[0].region.bbox[3])
            if (
                overlap >= line_tolerance * max(1.0, float(min_height))
                and overlap > best_overlap
            ):
                best_overlap = overlap
                best_line = line
        if best_line is None:
            lines.append([item])
        else:
            best_line.append(item)

    for line in lines:
        line.sort(key=lambda r: r.region.bbox[0], reverse=rtl)
    lines.sort(key=lambda line: min(item.region.bbox[1] for item in line))
    return lines


def order_regions(
    regions: list[RecognizedRegion],
    rtl: bool = False,
    line_tolerance: float = 0.5,
) -> list[RecognizedRegion]:
    """Flatten regions into line-major, reading-order sequence."""
    return [
        item
        for line in group_lines(regions, rtl=rtl, line_tolerance=line_tolerance)
        for item in line
    ]


def reconstruct_text(
    regions: list[RecognizedRegion],
    rtl: bool = False,
    line_breaks: bool = True,
) -> str:
    """Join recognized regions into text while preserving reading order."""
    lines = group_lines(regions, rtl=rtl)
    line_texts: list[str] = []
    for line in lines:
        words = [item.text.strip() for item in line if item.text.strip()]
        if words:
            line_texts.append(" ".join(words))
    separator = "\n" if line_breaks else " "
    return separator.join(line_texts).strip()


def mean_confidence(regions: list[RecognizedRegion]) -> float:
    """Mean confidence over regions that produced text."""
    values = [item.confidence for item in regions if item.text.strip()]
    if not values:
        return 0.0
    return max(0.0, min(1.0, float(mean(values))))
