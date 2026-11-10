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

    def _select_polarity(
        self, gray: np.ndarray, dark_ink: np.ndarray, light_ink: np.ndarray
    ) -> np.ndarray:
        dark_fraction = float(np.count_nonzero(dark_ink)) / dark_ink.size
        light_fraction = float(np.count_nonzero(light_ink)) / light_ink.size
        if dark_fraction > 0.5 and light_fraction < dark_fraction:
            return light_ink
        if light_fraction > 0.5 and dark_fraction < light_fraction:
            return dark_ink
        border = np.concatenate((gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]))
        if float(np.median(border)) < self.binarize_thresh / 2:
            return light_ink
        return dark_ink

    def binarize(self, image: np.ndarray) -> np.ndarray:
        """Convert an image to binary ink (foreground white on black).

        White-background crops use inverse thresholding. Dark-background crops
        (scene images, dark slides) use normal thresholding; otherwise the
        background becomes one giant connected component.
        """
        gray = self._to_gray(image)
        _, dark_ink = cv2.threshold(
            gray, self.binarize_thresh, 255, cv2.THRESH_BINARY_INV
        )
        _, light_ink = cv2.threshold(gray, self.binarize_thresh, 255, cv2.THRESH_BINARY)
        return self._select_polarity(gray, dark_ink, light_ink)

    def _otsu_binarize(self, image: np.ndarray) -> np.ndarray:
        gray = self._to_gray(image)
        _, dark_ink = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        _, light_ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return self._select_polarity(gray, dark_ink, light_ink)

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

    def _components(self, binary: np.ndarray) -> list[tuple[int, int, int, int]]:
        """Extract filtered connected components as ``(x, y, w, h)`` boxes."""
        num_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )
        components: list[tuple[int, int, int, int]] = []
        for i in range(1, num_labels):
            x = int(stats[i, cv2.CC_STAT_LEFT])
            y = int(stats[i, cv2.CC_STAT_TOP])
            w = int(stats[i, cv2.CC_STAT_WIDTH])
            h = int(stats[i, cv2.CC_STAT_HEIGHT])
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < self.min_component_area or w < 2 or h < 3:
                continue
            components.append((x, y, w, h))
        return components

    @staticmethod
    def _center_y(box: tuple[int, int, int, int]) -> float:
        return float(box[1]) + float(box[3]) / 2.0

    @classmethod
    def _line_bounds(
        cls, line: list[tuple[int, int, int, int]]
    ) -> tuple[int, int, int, int]:
        min_x = min(box[0] for box in line)
        min_y = min(box[1] for box in line)
        max_x = max(box[0] + box[2] for box in line)
        max_y = max(box[1] + box[3] for box in line)
        return min_x, min_y, max_x - min_x, max_y - min_y

    @classmethod
    def _group_components_into_lines(
        cls, components: list[tuple[int, int, int, int]]
    ) -> list[list[tuple[int, int, int, int]]]:
        """Group components by vertical overlap instead of assuming one line."""
        lines: list[list[tuple[int, int, int, int]]] = []
        for component in sorted(
            components, key=lambda box: (cls._center_y(box), box[0])
        ):
            top = float(component[1])
            bottom = top + float(component[3])
            center = cls._center_y(component)
            best_index: int | None = None
            best_overlap = 0.0
            best_center_distance = float("inf")
            for index, line in enumerate(lines):
                _, line_y, _, line_h = cls._line_bounds(line)
                line_top = float(line_y)
                line_bottom = line_top + float(line_h)
                overlap = max(0.0, min(bottom, line_bottom) - max(top, line_top))
                min_height = min(float(component[3]), float(line_h))
                overlap_ratio = overlap / min_height if min_height > 0 else 0.0
                line_center = line_top + float(line_h) / 2.0
                center_distance = abs(center - line_center)
                if overlap_ratio < 0.30:
                    continue
                if overlap_ratio > best_overlap or (
                    overlap_ratio == best_overlap
                    and center_distance < best_center_distance
                ):
                    best_index = index
                    best_overlap = overlap_ratio
                    best_center_distance = center_distance
            if best_index is None:
                lines.append([component])
            else:
                lines[best_index].append(component)
        lines.sort(
            key=lambda line: (cls._line_bounds(line)[1], cls._line_bounds(line)[0])
        )
        return lines

    def _line_to_word_boxes(
        self,
        line: list[tuple[int, int, int, int]],
        image_width: int,
        image_height: int,
        rtl: bool,
    ) -> list[tuple[int, int, int, int]]:
        """Split one visual line into word boxes using gap statistics."""
        ordered = sorted(line, key=lambda box: box[0], reverse=rtl)
        widths = [box[2] for box in ordered]
        heights = [box[3] for box in ordered]
        typical_w = float(np.median(widths)) if widths else 20.0
        typical_h = float(np.median(heights)) if heights else 30.0

        gaps: list[float] = []
        for index in range(1, len(ordered)):
            prev_x, _, prev_w, _ = ordered[index - 1]
            curr_x, _, curr_w, _ = ordered[index]
            if rtl:
                gap = float(prev_x - (curr_x + curr_w))
            else:
                gap = float(curr_x - (prev_x + prev_w))
            gaps.append(max(0.0, gap))

        space_threshold = max(8.0, 0.40 * typical_h)
        if len(gaps) >= 2:
            sorted_gaps = sorted(gaps)
            diffs = [
                sorted_gaps[k + 1] - sorted_gaps[k] for k in range(len(sorted_gaps) - 1)
            ]
            max_jump_idx = int(np.argmax(diffs))
            jump_size = diffs[max_jump_idx]
            candidate = (
                sorted_gaps[max_jump_idx] + sorted_gaps[max_jump_idx + 1]
            ) / 2.0
            if jump_size >= 4.0 and 6.0 <= candidate <= max(12.0, typical_w * 2.0):
                space_threshold = max(space_threshold, candidate)

        groups: list[list[tuple[int, int, int, int]]] = [[ordered[0]]]
        for index in range(1, len(ordered)):
            if gaps[index - 1] >= space_threshold:
                groups.append([ordered[index]])
            else:
                groups[-1].append(ordered[index])

        boxes: list[tuple[int, int, int, int]] = []
        for group in groups:
            min_x = min(box[0] for box in group)
            min_y = min(box[1] for box in group)
            max_x = max(box[0] + box[2] for box in group)
            max_y = max(box[1] + box[3] for box in group)
            padding_x = 2
            padding_y = 2
            x = max(0, min_x - padding_x)
            y = max(0, min_y - padding_y)
            width = min(image_width - x, max_x - min_x + 2 * padding_x)
            height = min(image_height - y, max_y - min_y + 2 * padding_y)
            if width > 1 and height > 1:
                boxes.append((x, y, width, height))
        return boxes

    def extract_word_boxes(
        self, image: np.ndarray, rtl: bool = False
    ) -> list[tuple[int, int, int, int]]:
        """Derive word proposals from lines using connected components.

        Components are first grouped into visual lines by vertical overlap, then
        each line is split into words using gap statistics. This is required for
        page-level images; the previous single-line assumption merged rows.

        Args:
            image: uint8 numpy array (H, W, 3) or (H, W).
            rtl: If True, order words right-to-left within each line.

        Returns:
            List of (x, y, w, h) word boxes in line-major reading order.
        """
        binary = self.binarize(image)
        h_img, w_img = binary.shape
        components = self._components(binary)
        if not components:
            # Retry with Otsu so low-contrast scans are not silently dropped.
            components = self._components(self._otsu_binarize(image))
        if not components:
            return []

        lines = self._group_components_into_lines(components)
        word_boxes: list[tuple[int, int, int, int]] = []
        for line in lines:
            word_boxes.extend(self._line_to_word_boxes(line, w_img, h_img, rtl=rtl))
        return word_boxes

    def extract_line_boxes(
        self, image: np.ndarray, rtl: bool = False
    ) -> list[tuple[int, int, int, int]]:
        """Return one bounding box per visual text line in reading order."""
        del rtl
        binary = self.binarize(image)
        components = self._components(binary)
        if not components:
            components = self._components(self._otsu_binarize(image))
        if not components:
            return []
        lines = self._group_components_into_lines(components)
        h_img, w_img = binary.shape
        boxes: list[tuple[int, int, int, int]] = []
        for line in lines:
            x, y, width, height = self._line_bounds(line)
            x = max(0, x - 2)
            y = max(0, y - 2)
            width = min(w_img - x, width + 4)
            height = min(h_img - y, height + 4)
            if width > 1 and height > 1:
                boxes.append((x, y, width, height))
        return boxes

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
