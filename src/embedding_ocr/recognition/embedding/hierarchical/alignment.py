"""Embedding-guided segmentation-free alignment for cursive/connected scripts.

Instead of hard-cutting a word into characters with projection heuristics, this
module scores overlapping sliding windows with the prototype index and uses
dynamic programming (Viterbi-style) to choose a non-overlapping sequence that
covers the word. It works with any prototype tier (characters, subwords or
words), so the same aligner serves Arabic connected scripts and Latin cursive.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from embedding_ocr.recognition.embedding.prototypes import PrototypeDB


@dataclass(frozen=True)
class AlignedUnit:
    """One aligned visual unit and its best prototype label."""

    image: np.ndarray
    bbox: tuple[int, int, int, int]  # x, y, w, h within the word crop
    text: str
    similarity: float


@dataclass(frozen=True)
class AlignmentResult:
    """Result of segmentation-free word alignment."""

    units: tuple[AlignedUnit, ...]
    score: float

    @property
    def text(self) -> str:
        """Concatenated unit labels in logical order."""
        return "".join(unit.text for unit in self.units)


class EmbeddingAlignmentSegmenter:
    """Sliding-window embedding alignment using a prototype database."""

    def __init__(
        self,
        prototype_db: PrototypeDB,
        canvas_size: int = 64,
        padding: int = 6,
        min_unit_width: int = 8,
        max_unit_width: int | None = None,
        stride: int = 4,
        width_step: int = 4,
        top_k: int = 5,
        min_similarity: float = 0.35,
        unit_penalty: float = 0.1,
        length_penalty: float = 0.0005,
    ) -> None:
        self.prototype_db = prototype_db
        self.canvas_size = canvas_size
        self.padding = padding
        self.min_unit_width = min_unit_width
        self.max_unit_width = max_unit_width
        self.stride = max(1, stride)
        self.width_step = max(1, width_step)
        self.top_k = top_k
        self.min_similarity = min_similarity
        self.unit_penalty = unit_penalty
        self.length_penalty = length_penalty

    def _to_canvas(self, crop: np.ndarray) -> np.ndarray:
        """Fit a crop onto a white square canvas, preserving aspect ratio."""
        if crop.ndim == 3 and crop.shape[2] == 4:
            crop = cv2.cvtColor(crop, cv2.COLOR_RGBA2RGB)
        h, w = crop.shape[:2]
        scale = (self.canvas_size - 2 * self.padding) / max(w, h, 1)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
        start_x = (self.canvas_size - new_w) // 2
        start_y = (self.canvas_size - new_h) // 2
        if resized.ndim == 3:
            canvas = np.full(
                (self.canvas_size, self.canvas_size, 3), 255, dtype=np.uint8
            )
            canvas[start_y : start_y + new_h, start_x : start_x + new_w] = resized
            return canvas
        canvas = np.full((self.canvas_size, self.canvas_size), 255, dtype=np.uint8)
        canvas[start_y : start_y + new_h, start_x : start_x + new_w] = resized
        return cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGB)

    def _candidate_widths(self, height: int, width: int) -> list[int]:
        max_width = self.max_unit_width or max(self.min_unit_width, height * 2)
        max_width = min(width, max_width)
        if max_width < self.min_unit_width:
            return []
        widths = list(range(self.min_unit_width, max_width + 1, self.width_step))
        if widths[-1] != max_width:
            widths.append(max_width)
        return widths

    def align(self, word_crop: np.ndarray, rtl: bool = False) -> AlignmentResult:
        """Align a word crop to prototype units.

        Args:
            word_crop: uint8 numpy array (H, W, 3) or (H, W).
            rtl: If True, return units in right-to-left logical order.

        Returns:
            AlignmentResult with aligned units and the DP score.
        """
        if word_crop is None or word_crop.size == 0:
            return AlignmentResult(units=(), score=0.0)

        h, w = word_crop.shape[:2]
        widths = self._candidate_widths(h, w)
        if not widths:
            return AlignmentResult(units=(), score=0.0)

        starts: set[int] = set(range(0, w, self.stride))
        starts.add(0)
        for width in widths:
            starts.add(max(0, w - width))
        ordered_starts = sorted(start for start in starts if start < w)

        windows: list[tuple[int, int, int, int]] = []  # x, end, y, width
        for start in ordered_starts:
            for width in widths:
                end = start + width
                if end > w:
                    continue
                windows.append((start, end, 0, width))

        if not windows:
            return AlignmentResult(units=(), score=0.0)

        images = [
            self._to_canvas(word_crop[0:h, x:end]) for x, end, _y, _width in windows
        ]
        embeddings = self.prototype_db.embedder.embed_images(images)
        sims, labels = self.prototype_db.search(embeddings, top_k=self.top_k)

        window_scores: list[float] = []
        window_text: list[str] = []
        for row_sims, row_labels in zip(sims, labels, strict=False):
            best_sim = -1.0
            best_label = ""
            for sim, label in zip(row_sims, row_labels, strict=False):
                value = float(sim)
                if value > best_sim:
                    best_sim = value
                    best_label = label
            window_scores.append(best_sim)
            window_text.append(best_label)

        by_start: dict[int, list[int]] = {}
        for idx, (start, _end, _y, _width) in enumerate(windows):
            by_start.setdefault(start, []).append(idx)

        # Minimize the total alignment cost:
        #   sum(1 - similarity) + unit_penalty * n_units + length_penalty * width
        # The per-unit penalty prevents splitting a perfect block into many
        # tiny perfect windows, while the similarity term still prefers real
        # boundaries over a single mixed full-word window.
        dp: dict[int, float] = {0: 0.0}
        back: dict[int, tuple[int, int]] = {}

        positions = sorted(
            set(ordered_starts) | {end for _x, end, _y, _width in windows}
        )
        for start in positions:
            if start not in dp:
                continue
            for idx in by_start.get(start, []):
                _x, end, _y, width = windows[idx]
                if window_scores[idx] < self.min_similarity:
                    continue
                candidate = (
                    dp[start]
                    + (1.0 - window_scores[idx])
                    + self.unit_penalty
                    + self.length_penalty * width
                )
                if end not in dp or candidate < dp[end]:
                    dp[end] = candidate
                    back[end] = (start, idx)

        if w not in dp:
            return AlignmentResult(units=(), score=0.0)

        chosen: list[int] = []
        cursor = w
        while cursor > 0:
            start, idx = back[cursor]
            chosen.append(idx)
            cursor = start
        chosen.reverse()

        units = [
            AlignedUnit(
                image=images[idx],
                bbox=(windows[idx][0], windows[idx][2], windows[idx][3], h),
                text=window_text[idx],
                similarity=window_scores[idx],
            )
            for idx in chosen
        ]
        if rtl:
            units.reverse()

        mean_similarity = (
            float(np.mean([unit.similarity for unit in units])) if units else 0.0
        )
        return AlignmentResult(units=tuple(units), score=mean_similarity)
