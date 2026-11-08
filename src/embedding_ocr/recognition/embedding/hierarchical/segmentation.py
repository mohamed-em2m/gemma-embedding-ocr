"""Visual character and subword segmentation for word crops.

Uses whitespace gaps, vertical projection profiles, and connected components to:
1. Extract word proposals from multi-word text regions.
2. Find visual cut boundaries (valleys in vertical projection) for recursive subword splitting.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class CutCandidate:
    """A cheaply scored visual boundary between two units."""

    x: int
    score: float


class VisualWordSegmenter:
    """Handles visual segmentation and cut boundaries on text lines and words."""

    def __init__(
        self,
        binarize_thresh: int = 180,
        min_component_area: int = 8,
    ) -> None:
        self.binarize_thresh = binarize_thresh
        self.min_component_area = min_component_area

    @staticmethod
    def _to_gray(image: np.ndarray) -> np.ndarray:
        """Convert 1/3/4-channel image data to a 2-D grayscale array."""
        if image.ndim == 2:
            return image
        channels = image.shape[2]
        if channels == 4:
            return cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
        if channels == 3:
            return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return image[..., 0]

    def binarize(self, image: np.ndarray) -> np.ndarray:
        """Convert RGB image to inverted binary (white text on black background)."""
        gray = self._to_gray(image)
        _, binary = cv2.threshold(
            gray, self.binarize_thresh, 255, cv2.THRESH_BINARY_INV
        )
        return binary

    def propose_cut_candidates(
        self,
        image: np.ndarray,
        max_candidates: int | None = 32,
        min_spacing: float | None = None,
    ) -> list[CutCandidate]:
        """Propose a small set of likely unit boundaries using cheap geometry.

        Scores vertical-projection valleys, zero-ink gaps, connected-component
        edges, and local projection minima. A non-maximum-suppression pass keeps
        boundaries at least ``min_spacing`` pixels apart.

        Args:
            image: Word crop as uint8 (H, W, 3) or (H, W).
            max_candidates: Optional cap on returned boundaries (endpoints are
                always retained). ``None`` keeps all non-suppressed candidates.
            min_spacing: Minimum distance between selected boundaries. Defaults
                to one quarter of the crop height.

        Returns:
            Boundary candidates sorted by x, including 0 and the crop width.
        """
        binary = self.binarize(image)
        h, w = binary.shape
        if w <= 1:
            return [CutCandidate(0, 0.0), CutCandidate(w, 0.0)]

        ink = (binary > 0).sum(axis=0).astype(np.float32)
        max_ink = float(ink.max())
        if max_ink <= 0:
            return [CutCandidate(0, 0.0), CutCandidate(w, 0.0)]

        valley = 1.0 - ink / max_ink

        gray = self._to_gray(image).astype(np.float32)
        gradient = np.zeros(w, dtype=np.float32)
        if w > 1:
            diff = np.abs(np.diff(gray, axis=1)).mean(axis=0)
            gradient[1:] = diff
            max_gradient = float(gradient.max())
            if max_gradient > 0:
                gradient /= max_gradient

        gap_score = np.zeros(w, dtype=np.float32)
        in_gap = False
        gap_start = 0
        for x, value in enumerate(ink):
            if value <= 0 and not in_gap:
                in_gap = True
                gap_start = x
            elif value > 0 and in_gap:
                gap_score[gap_start:x] = min(1.0, (x - gap_start) / max(2.0, h * 0.5))
                in_gap = False
        if in_gap:
            gap_score[gap_start:w] = min(1.0, (w - gap_start) / max(2.0, h * 0.5))

        edge_mask = np.zeros(w, dtype=bool)
        num_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )
        for i in range(1, num_labels):
            x = int(stats[i, cv2.CC_STAT_LEFT])
            cw = int(stats[i, cv2.CC_STAT_WIDTH])
            ch = int(stats[i, cv2.CC_STAT_HEIGHT])
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < self.min_component_area or cw < 2 or ch < 4:
                continue
            edge_mask[x] = True
            edge_mask[min(w - 1, x + cw - 1)] = True

        local_minima = np.zeros(w, dtype=np.float32)
        for x in range(1, w - 1):
            if ink[x] <= ink[x - 1] and ink[x] <= ink[x + 1]:
                local_minima[x] = 1.0

        scores = (
            0.35 * valley
            + 0.20 * gap_score
            + 0.25 * edge_mask.astype(np.float32)
            + 0.15 * gradient
            + 0.05 * local_minima
        )

        ink_columns = np.flatnonzero(ink > 0)
        if ink_columns.size:
            first_ink = int(ink_columns[0])
            last_ink = int(ink_columns[-1])
        else:
            first_ink = 0
            last_ink = w - 1

        forced = [0, first_ink, last_ink, w]
        spacing = min_spacing if min_spacing is not None else max(2.0, h * 0.25)
        selected: list[int] = []
        for x in sorted(set(forced)):
            if not selected or abs(x - selected[-1]) >= spacing:
                selected.append(x)
        if w not in selected:
            selected.append(w)

        if max_candidates is None:
            order = sorted(
                range(first_ink, last_ink + 1),
                key=lambda x: (-float(scores[x]), x),
            )
            for x in order:
                if any(abs(x - y) < spacing for y in selected):
                    continue
                selected.append(x)
        else:
            remaining = max(0, max_candidates - len(selected))
            span = last_ink - first_ink
            if remaining > 0 and span > 0:
                for bin_index in range(remaining):
                    lo = first_ink + int(round(span * bin_index / remaining))
                    hi = first_ink + int(round(span * (bin_index + 1) / remaining))
                    midpoint = (lo + hi) / 2.0
                    choices = sorted(
                        range(lo, hi + 1),
                        key=lambda x: (-float(scores[x]), abs(x - midpoint)),
                    )
                    for x in choices:
                        if any(abs(x - y) < spacing for y in selected):
                            continue
                        selected.append(x)
                        break

        ordered = sorted(set(selected))
        return [
            CutCandidate(x=x, score=float(scores[x]) if x < w else 0.0) for x in ordered
        ]

    def extract_word_boxes(
        self, image: np.ndarray, rtl: bool = False
    ) -> list[tuple[int, int, int, int]]:
        """Derive word proposals from text-line regions using whitespace gaps and connected components.

        Args:
            image: uint8 numpy array (H, W, 3) or (H, W).
            rtl: If True, return boxes in right-to-left reading order.

        Returns:
            List of (x, y, w, h) bounding boxes for words in reading order.
        """
        binary = self.binarize(image)
        h_img, w_img = binary.shape

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )

        components: list[tuple[int, int, int, int]] = []
        for i in range(1, num_labels):
            x = int(stats[i, cv2.CC_STAT_LEFT])
            y = int(stats[i, cv2.CC_STAT_TOP])
            w = int(stats[i, cv2.CC_STAT_WIDTH])
            h = int(stats[i, cv2.CC_STAT_HEIGHT])
            area = int(stats[i, cv2.CC_STAT_AREA])

            if area < self.min_component_area or w < 2 or h < 4:
                continue
            components.append((x, y, w, h))

        if not components:
            return [(0, 0, w_img, h_img)]

        components.sort(key=lambda c: c[0], reverse=rtl)

        widths = [c[2] for c in components]
        heights = [c[3] for c in components]
        typical_w = float(np.median(widths)) if widths else 20.0
        typical_h = float(np.median(heights)) if heights else 30.0

        # Inter-character gaps in reading order. For RTL, the previous
        # component sits to the right of the current one.
        gaps: list[float] = []
        for i in range(1, len(components)):
            prev_x, _, prev_w, _ = components[i - 1]
            curr_x, _, curr_w, _ = components[i]
            if rtl:
                gap = float(prev_x - (curr_x + curr_w))
            else:
                gap = float(curr_x - (prev_x + prev_w))
            gaps.append(max(0.0, gap))

        # Dynamic inter-word space threshold
        # Inter-character gaps in standard Latin text are typically 1-4 pixels,
        # while word spaces are typically >= 8-10 pixels (or >= 0.35 of line height).
        space_threshold = max(7.0, 0.35 * max(typical_w, typical_h * 0.4))
        if len(gaps) >= 2:
            sorted_gaps = sorted(gaps)
            diffs = [
                sorted_gaps[k + 1] - sorted_gaps[k] for k in range(len(sorted_gaps) - 1)
            ]
            max_jump_idx = int(np.argmax(diffs))
            jump_size = diffs[max_jump_idx]
            cand = (sorted_gaps[max_jump_idx] + sorted_gaps[max_jump_idx + 1]) / 2.0
            if jump_size >= 3.0 and 6.0 <= cand <= max(12.0, typical_w * 2.0):
                space_threshold = cand

        # Group components into words
        word_boxes: list[tuple[int, int, int, int]] = []
        cur_group: list[tuple[int, int, int, int]] = [components[0]]

        for i in range(1, len(components)):
            gap = gaps[i - 1]
            if gap >= space_threshold:
                # Finish current word group
                min_x = min(c[0] for c in cur_group)
                min_y = min(c[1] for c in cur_group)
                max_x = max(c[0] + c[2] for c in cur_group)
                max_y = max(c[1] + c[3] for c in cur_group)
                # Add light padding
                px = 2
                py = 2
                word_boxes.append(
                    (
                        max(0, min_x - px),
                        max(0, min_y - py),
                        min(w_img - min_x, max_x - min_x + 2 * px),
                        min(h_img - min_y, max_y - min_y + 2 * py),
                    )
                )
                cur_group = [components[i]]
            else:
                cur_group.append(components[i])

        if cur_group:
            min_x = min(c[0] for c in cur_group)
            min_y = min(c[1] for c in cur_group)
            max_x = max(c[0] + c[2] for c in cur_group)
            max_y = max(c[1] + c[3] for c in cur_group)
            px = 2
            py = 2
            word_boxes.append(
                (
                    max(0, min_x - px),
                    max(0, min_y - py),
                    min(w_img - min_x, max_x - min_x + 2 * px),
                    min(h_img - min_y, max_y - min_y + 2 * py),
                )
            )

        return word_boxes

    def find_optimal_cut_boundary(
        self,
        image: np.ndarray,
        center_fraction_range: tuple[float, float] = (0.25, 0.75),
    ) -> int:
        """Find the most suitable visual segmentation boundary using vertical projection.

        Rather than blindly dividing by pixel width, finds the deepest valley in the
        ink projection profile within the central region of the image.

        Args:
            image: uint8 numpy array (H, W, 3) or (H, W).
            center_fraction_range: Region within which to search for the cut (default 25% to 75%).

        Returns:
            X-coordinate column index to cut the image into two halves.
        """
        binary = self.binarize(image)
        h, w = binary.shape

        if w <= 8:
            return w // 2

        # Vertical projection profile (sum of ink pixels per column)
        col_ink = np.sum(binary > 0, axis=0)  # shape (w,)

        start_col = int(round(w * center_fraction_range[0]))
        end_col = int(round(w * center_fraction_range[1]))
        start_col = max(1, min(start_col, w - 2))
        end_col = max(start_col + 1, min(end_col, w - 1))

        sub_profile = col_ink[start_col:end_col]

        # Prioritize absolute gaps (ink == 0) closest to the center
        center_idx = (start_col + end_col) // 2
        min_ink = int(np.min(sub_profile))

        candidate_cols = np.where(sub_profile == min_ink)[0] + start_col
        # Pick candidate closest to center
        best_col = int(candidate_cols[np.argmin(np.abs(candidate_cols - center_idx))])

        return best_col

    def split_word_image(
        self,
        image: np.ndarray,
        padding_cut: int = 4,
        rtl: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """Split a word image into two sub-crops using the visual boundary.

        Args:
            image: uint8 numpy array (H, W, 3) or (H, W).
            padding_cut: Horizontal white padding added to each sub-crop.
            rtl: If True, return the right sub-crop first (logical reading order).

        Returns:
            (first_image, second_image, cut_x) where ``first_image`` is the
            left crop for LTR and the right crop for RTL.
        """
        cut_x = self.find_optimal_cut_boundary(image)

        left_crop = image[:, :cut_x]
        right_crop = image[:, cut_x:]

        # Pad both slightly to prevent characters right at the border from clipping
        def pad_crop(crop: np.ndarray) -> np.ndarray:
            ch, cw = crop.shape[:2]
            if cw < 2:
                return crop
            padded = np.full((ch, cw + padding_cut * 2, 3), 255, dtype=np.uint8)
            padded[:, padding_cut : padding_cut + cw] = crop
            return padded

        if rtl:
            return pad_crop(right_crop), pad_crop(left_crop), cut_x
        return pad_crop(left_crop), pad_crop(right_crop), cut_x
