"""Arabic-aware word-to-unit segmentation.

Arabic letters join along a baseline and carry optional diacritics, so naive
connected-component sorting produces either whole words or diacritic noise.
This segmenter merges small components (dots, harakat) into their nearest base
component and returns units in logical reading order. It intentionally does not
force hard cuts through joined letters; use :mod:`alignment` for that.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class ArabicUnit:
    """A base component with any attached diacritics merged into its bbox."""

    image: np.ndarray  # (target_size, target_size, 3) canvas crop
    bbox: tuple[int, int, int, int]  # x, y, w, h within the word crop
    relative_height: float
    relative_top: float
    relative_bottom: float


class ArabicUnitSegmenter:
    """Connected-component unit segmentation with diacritic merging."""

    def __init__(
        self,
        target_size: int = 64,
        padding: int = 10,
        binarize_thresh: int = 180,
        min_component_area: int = 3,
        diacritic_height_ratio: float = 0.4,
        diacritic_area_ratio: float = 0.2,
    ) -> None:
        self.target_size = target_size
        self.padding = padding
        self.binarize_thresh = binarize_thresh
        self.min_component_area = min_component_area
        self.diacritic_height_ratio = diacritic_height_ratio
        self.diacritic_area_ratio = diacritic_area_ratio

    def _fit_to_canvas(self, crop: np.ndarray) -> np.ndarray:
        if crop.ndim == 3 and crop.shape[2] == 4:
            crop = cv2.cvtColor(crop, cv2.COLOR_RGBA2RGB)
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY) if crop.ndim == 3 else crop

        h, w = gray.shape[:2]
        canvas = np.full((self.target_size, self.target_size), 255, dtype=np.uint8)
        scale = (self.target_size - 2 * self.padding) / max(w, h, 1)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        resized = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
        start_x = (self.target_size - new_w) // 2
        start_y = (self.target_size - new_h) // 2
        canvas[start_y : start_y + new_h, start_x : start_x + new_w] = resized
        return cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGB)

    def segment(self, word_crop: np.ndarray, rtl: bool = True) -> list[ArabicUnit]:
        """Segment a word crop into base units in logical reading order.

        Args:
            word_crop: uint8 numpy array (H, W, 3) or (H, W).
            rtl: If True, return units right-to-left.

        Returns:
            List of ArabicUnit objects.
        """
        if word_crop is None or word_crop.size == 0:
            return []

        if word_crop.ndim == 3 and word_crop.shape[2] == 4:
            word_crop = cv2.cvtColor(word_crop, cv2.COLOR_RGBA2RGB)
        if word_crop.ndim == 3:
            gray = cv2.cvtColor(word_crop, cv2.COLOR_RGB2GRAY)
        else:
            gray = word_crop
        h_img, w_img = gray.shape

        _, binary = cv2.threshold(
            gray, self.binarize_thresh, 255, cv2.THRESH_BINARY_INV
        )
        num_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )

        components: list[tuple[int, int, int, int, int]] = []
        for i in range(1, num_labels):
            x = int(stats[i, cv2.CC_STAT_LEFT])
            y = int(stats[i, cv2.CC_STAT_TOP])
            w = int(stats[i, cv2.CC_STAT_WIDTH])
            h = int(stats[i, cv2.CC_STAT_HEIGHT])
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < self.min_component_area or w < 1 or h < 1:
                continue
            components.append((x, y, w, h, area))

        if not components:
            return []

        median_h = float(np.median([c[3] for c in components]))
        median_area = float(np.median([c[4] for c in components]))
        height_cut = max(2.0, median_h * self.diacritic_height_ratio)
        area_cut = max(2.0, median_area * self.diacritic_area_ratio)

        bases: list[list[int]] = []
        diacritics: list[tuple[int, int, int, int, int]] = []
        for comp in components:
            x, y, w, h, area = comp
            if h < height_cut and area < area_cut:
                diacritics.append(comp)
            else:
                bases.append([x, y, w, h, area])

        if not bases:
            bases = [list(comp) for comp in components]
            diacritics = []

        # Merge each diacritic into the nearest base by horizontal distance.
        for dx, dy, dw, dh, _darea in diacritics:
            d_center = dx + dw / 2.0
            best_idx = min(
                range(len(bases)),
                key=lambda i: abs((bases[i][0] + bases[i][2] / 2.0) - d_center),
            )
            base = bases[best_idx]
            min_x = min(base[0], dx)
            min_y = min(base[1], dy)
            max_x = max(base[0] + base[2], dx + dw)
            max_y = max(base[1] + base[3], dy + dh)
            base[0] = min_x
            base[1] = min_y
            base[2] = max_x - min_x
            base[3] = max_y - min_y

        bases.sort(key=lambda c: c[0], reverse=rtl)

        units: list[ArabicUnit] = []
        for x, y, w, h, _area in bases:
            crop = word_crop[y : y + h, x : x + w]
            units.append(
                ArabicUnit(
                    image=self._fit_to_canvas(crop),
                    bbox=(x, y, w, h),
                    relative_height=float(h) / max(1.0, float(h_img)),
                    relative_top=float(y) / max(1.0, float(h_img)),
                    relative_bottom=float(y + h) / max(1.0, float(h_img)),
                )
            )
        return units
