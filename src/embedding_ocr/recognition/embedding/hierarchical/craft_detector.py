"""CRAFT Text Detector adapter for word and character box proposals."""

from __future__ import annotations

import numpy as np


class CraftWordDetector:
    """Detects word and character proposals using CRAFT text detector."""

    def __init__(
        self,
        cuda: bool = True,
        refiner: bool = False,  # False produces word-level boxes rather than lines
        text_threshold: float = 0.7,
        link_threshold: float = 0.2,
        low_text: float = 0.4,
        fallback_full_image: bool = True,
    ) -> None:
        self.cuda = cuda
        self.refiner = refiner
        self.text_threshold = text_threshold
        self.link_threshold = link_threshold
        self.low_text = low_text
        self.fallback_full_image = fallback_full_image
        self._craft = None

    def _ensure_loaded(self) -> None:
        if self._craft is None:
            import torchvision.models.vgg as vgg

            if not hasattr(vgg, "model_urls"):
                vgg.model_urls = {
                    "vgg16_bn": "https://download.pytorch.org/models/vgg16_bn-6c64b313.pth"
                }

            from craft_text_detector import Craft

            self._craft = Craft(
                crop_type="box",
                cuda=self.cuda,
                refiner=self.refiner,
                text_threshold=self.text_threshold,
                link_threshold=self.link_threshold,
                low_text=self.low_text,
            )

    def detect_word_boxes(
        self, image: np.ndarray, rtl: bool = False
    ) -> list[tuple[int, int, int, int]]:
        """Detect word bounding boxes (x, y, w, h) in reading order.

        Args:
            image: uint8 numpy array (H, W, 3) or (H, W).
            rtl: If True, order boxes right-to-left instead of left-to-right.

        Returns:
            List of (x, y, w, h) bounding boxes.
        """
        self._ensure_loaded()
        if self._craft is None:
            return []
        if image.ndim == 2:
            import cv2

            img_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = image

        h_img, w_img = img_rgb.shape[:2]
        preds = self._craft.detect_text(img_rgb)
        boxes = preds.get("boxes", [])

        word_boxes: list[tuple[int, int, int, int]] = []
        for poly in boxes:
            pts = np.array(poly, dtype=np.float32)
            min_x = max(0, int(np.floor(np.min(pts[:, 0]))))
            max_x = min(w_img, int(np.ceil(np.max(pts[:, 0]))))
            min_y = max(0, int(np.floor(np.min(pts[:, 1]))))
            max_y = min(h_img, int(np.ceil(np.max(pts[:, 1]))))

            w = max_x - min_x
            h = max_y - min_y
            if w > 2 and h > 2:
                word_boxes.append((min_x, min_y, w, h))

        # Sort in reading order (LTR ascending, RTL descending)
        word_boxes.sort(key=lambda b: b[0], reverse=rtl)
        if word_boxes:
            return word_boxes
        if not self.fallback_full_image:
            return []
        return [(0, 0, w_img, h_img)]
