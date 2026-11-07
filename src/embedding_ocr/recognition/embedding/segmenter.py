"""Character segmenter using vertical projection and connected components."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class SegmentedCharacter:
    """A single segmented character crop and its geometric bounding box."""

    image: np.ndarray  # (H, W, 3) normalized canvas crop
    bbox: tuple[int, int, int, int]  # x, y, w, h within original crop
    aspect_ratio: float  # h / w
    relative_height: float  # h / line_height
    is_space_before: bool = (
        False  # True if inter-character spacing indicates a word boundary
    )
    relative_top: float = 0.0  # y / line_height (0.0 = top of line)
    relative_bottom: float = 1.0  # (y + h) / line_height (1.0 = bottom of line)


class CharacterSegmenter:
    """Segments a text-line or word crop into isolated character crops."""

    def __init__(
        self,
        target_size: int = 64,
        padding: int = 10,
        binarize_thresh: int = 180,
        space_ratio_thresh: float = 0.45,
    ) -> None:
        self.target_size = target_size
        self.padding = padding
        self.binarize_thresh = binarize_thresh
        self.space_ratio_thresh = space_ratio_thresh

    def segment(self, image: np.ndarray, rtl: bool = False) -> list[SegmentedCharacter]:
        """Segment an input word/line image into individual character crops.

        Args:
            image: uint8 numpy array (H, W, 3) or (H, W).
            rtl: If True, return components in right-to-left reading order.

        Returns:
            List of SegmentedCharacter objects in logical reading order.
        """
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()

        h_img, w_img = gray.shape

        # Binarize (black text on white background)
        _, binary = cv2.threshold(
            gray, self.binarize_thresh, 255, cv2.THRESH_BINARY_INV
        )

        # Connected components with stats
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

            # Filter out tiny noise specks
            if area < 8 or w < 2 or h < 4:
                continue

            components.append((x, y, w, h))

        # Sort in reading order (LTR ascending, RTL descending)
        components.sort(key=lambda c: c[0], reverse=rtl)

        # Estimate average character width and gap distribution
        widths = [c[2] for c in components]
        heights = [c[3] for c in components]
        typical_w = float(np.median(widths)) if widths else 20.0
        typical_h = float(np.median(heights)) if heights else 30.0

        # Compute adjacent inter-character gaps in reading order. For RTL,
        # the previous component sits to the right of the current one.
        gaps: list[float] = []
        for i in range(1, len(components)):
            prev_x, _, prev_w, _ = components[i - 1]
            curr_x, _, curr_w, _ = components[i]
            if rtl:
                gap = float(prev_x - (curr_x + curr_w))
            else:
                gap = float(curr_x - (prev_x + prev_w))
            gaps.append(max(0.0, gap))

        # Dynamic space threshold:
        # If text has multiple words, gaps exhibit a bimodal distribution
        # (small intra-word character spacing vs large inter-word spacing).
        space_threshold = max(
            8.0, self.space_ratio_thresh * max(typical_w, typical_h * 0.4)
        )
        if len(gaps) >= 3:
            sorted_gaps = sorted(gaps)
            # Find largest gap jump between small gaps and large gaps
            diffs = [
                sorted_gaps[k + 1] - sorted_gaps[k] for k in range(len(sorted_gaps) - 1)
            ]
            max_jump_idx = int(np.argmax(diffs))
            jump_size = diffs[max_jump_idx]
            candidate_thresh = (
                sorted_gaps[max_jump_idx] + sorted_gaps[max_jump_idx + 1]
            ) / 2.0
            # If the jump is distinct (at least 4px and candidate threshold is reasonable)
            if jump_size >= 4.0 and 6.0 <= candidate_thresh <= typical_w * 1.5:
                space_threshold = candidate_thresh

        segmented_chars: list[SegmentedCharacter] = []
        for idx, (x, y, w, h) in enumerate(components):
            char_crop = gray[y : y + h, x : x + w]

            # Detect if there is a space before this component (word break)
            is_space_before = False
            if idx > 0:
                gap = gaps[idx - 1]
                if gap >= space_threshold:
                    is_space_before = True

            # Fit into target_size x target_size canvas maintaining aspect ratio
            canvas = np.full((self.target_size, self.target_size), 255, dtype=np.uint8)
            scale = (self.target_size - 2 * self.padding) / max(w, h)
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))

            resized_char = cv2.resize(
                char_crop, (new_w, new_h), interpolation=cv2.INTER_AREA
            )

            start_x = (self.target_size - new_w) // 2
            start_y = (self.target_size - new_h) // 2
            canvas[start_y : start_y + new_h, start_x : start_x + new_w] = resized_char

            # Convert to 3-channel RGB for embedder
            canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGB)

            aspect = float(h) / max(1.0, float(w))
            rel_h = float(h) / max(1.0, float(h_img))
            rel_top = float(y) / max(1.0, float(h_img))
            rel_bottom = float(y + h) / max(1.0, float(h_img))

            segmented_chars.append(
                SegmentedCharacter(
                    image=canvas_rgb,
                    bbox=(x, y, w, h),
                    aspect_ratio=aspect,
                    relative_height=rel_h,
                    is_space_before=is_space_before,
                    relative_top=rel_top,
                    relative_bottom=rel_bottom,
                )
            )

        return segmented_chars
