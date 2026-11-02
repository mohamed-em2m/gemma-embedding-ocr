"""Character segmenter using vertical projection and connected components."""

from __future__ import annotations

from dataclasses import dataclass
import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class SegmentedCharacter:
    """A single segmented character crop and its geometric bounding box."""

    image: np.ndarray  # (H, W, 3) normalized canvas crop
    bbox: tuple[int, int, int, int]  # x, y, w, h within original crop
    aspect_ratio: float  # h / w
    relative_height: float  # h / line_height


class CharacterSegmenter:
    """Segments a text-line or word crop into isolated character crops."""

    def __init__(
        self,
        target_size: int = 64,
        padding: int = 10,
        binarize_thresh: int = 180,
    ) -> None:
        self.target_size = target_size
        self.padding = padding
        self.binarize_thresh = binarize_thresh

    def segment(self, image: np.ndarray) -> list[SegmentedCharacter]:
        """Segment an input word/line image into individual character crops.

        Args:
            image: uint8 numpy array (H, W, 3) or (H, W).

        Returns:
            List of SegmentedCharacter objects ordered left to right.
        """
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()

        h_img, w_img = gray.shape

        # Binarize (black text on white background)
        _, binary = cv2.threshold(gray, self.binarize_thresh, 255, cv2.THRESH_BINARY_INV)

        # Connected components with stats
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

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

        # Sort left to right
        components.sort(key=lambda c: c[0])

        segmented_chars: list[SegmentedCharacter] = []
        for x, y, w, h in components:
            char_crop = gray[y : y + h, x : x + w]

            # Fit into target_size x target_size canvas maintaining aspect ratio
            canvas = np.full((self.target_size, self.target_size), 255, dtype=np.uint8)
            scale = (self.target_size - 2 * self.padding) / max(w, h)
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))

            resized_char = cv2.resize(char_crop, (new_w, new_h), interpolation=cv2.INTER_AREA)

            start_x = (self.target_size - new_w) // 2
            start_y = (self.target_size - new_h) // 2
            canvas[start_y : start_y + new_h, start_x : start_x + new_w] = resized_char

            # Convert to 3-channel RGB for embedder
            canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGB)

            aspect = float(h) / max(1.0, float(w))
            rel_h = float(h) / max(1.0, float(h_img))

            segmented_chars.append(
                SegmentedCharacter(
                    image=canvas_rgb,
                    bbox=(x, y, w, h),
                    aspect_ratio=aspect,
                    relative_height=rel_h,
                )
            )

        return segmented_chars
