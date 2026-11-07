"""Crop detected text regions from images."""

from __future__ import annotations

import numpy as np

from embedding_ocr.pipeline.types import TextRegion


def normalize_bbox(
    bbox: tuple[int, int, int, int], width: int, height: int
) -> tuple[int, int, int, int]:
    """Clamp a bbox to image bounds and return a non-empty box when possible."""
    x, y, w, h = (int(value) for value in bbox)
    x = max(0, min(x, width))
    y = max(0, min(y, height))
    w = max(0, min(w, width - x))
    h = max(0, min(h, height - y))
    return x, y, w, h


def crop_regions(
    image: np.ndarray, regions: list[TextRegion], padding: int = 0
) -> list[np.ndarray]:
    """Crop each region from an image with optional padding.

    Args:
        image: uint8 array of shape (H, W, C) or (H, W).
        regions: Regions in image coordinates.
        padding: Extra pixels around each bbox (clamped to image bounds).

    Returns:
        List of crops in the same order as ``regions``.
    """
    if image.ndim < 2:
        msg = f"image must be at least 2-D, got shape {image.shape}"
        raise ValueError(msg)

    height, width = image.shape[:2]
    crops: list[np.ndarray] = []
    for region in regions:
        x, y, w, h = normalize_bbox(region.bbox, width, height)
        x0 = max(0, x - padding)
        y0 = max(0, y - padding)
        x1 = min(width, x + w + padding)
        y1 = min(height, y + h + padding)
        crops.append(image[y0:y1, x0:x1])
    return crops
