"""Embedding-guided segmentation-free alignment for cursive/connected scripts.

Instead of hard-cutting a word into characters with projection heuristics, this
module scores overlapping sliding windows with the prototype index and uses
dynamic programming (Viterbi-style) to choose a non-overlapping sequence that
covers the word. It works with any prototype tier (characters, subwords or
words), so the same aligner serves Arabic connected scripts and Latin cursive.

By default the candidate graph is pruned before any model call: cheap geometry
proposes a small set of likely cut positions, and only edges between those
positions are embedded. The exhaustive stride-by-width graph remains available
as an opt-in strategy and as a conservative fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import cv2
import numpy as np

from embedding_ocr.recognition.embedding.embedder import Embedder
from embedding_ocr.recognition.embedding.hierarchical.segmentation import (
    VisualWordSegmenter,
)


class PrototypeSearchDB(Protocol):
    """Minimal prototype store interface needed by the aligner."""

    embedder: Embedder

    def search(
        self, query_embeddings: np.ndarray, top_k: int = 5
    ) -> tuple[np.ndarray, list[list[str]]]:
        """Return similarities and labels for each query."""
        ...


class VocabularyPrototypeDB:
    """Adapt an ``EmbeddingVocabulary`` character tier to the aligner API."""

    def __init__(self, embedder: Embedder, vocabulary: Any) -> None:
        self.embedder = embedder
        self.vocabulary = vocabulary

    def search(
        self, query_embeddings: np.ndarray, top_k: int = 5
    ) -> tuple[np.ndarray, list[list[str]]]:
        sims, token_rows = self.vocabulary.search_chars(query_embeddings, top_k=top_k)
        labels = [[token.text for token in row] for row in token_rows]
        return sims, labels


@dataclass(frozen=True)
class AlignedUnit:
    """One aligned visual unit and its best prototype label."""

    image: np.ndarray
    bbox: tuple[int, int, int, int]  # x, y, w, h within the word crop
    text: str
    similarity: float


@dataclass(frozen=True)
class AlignmentStats:
    """Counters describing the candidate graph used by one alignment attempt."""

    strategy: str
    boundary_candidates: int
    selected_starts: int
    candidate_windows: int
    embedded_windows: int
    fallback_used: bool = False


@dataclass(frozen=True)
class AlignmentResult:
    """Result of segmentation-free word alignment."""

    units: tuple[AlignedUnit, ...]
    score: float
    stats: AlignmentStats | None = None

    @property
    def text(self) -> str:
        """Concatenated unit labels in logical order."""
        return "".join(unit.text for unit in self.units)


@dataclass(frozen=True)
class _CandidateWindow:
    """One candidate segment (DAG edge) before embedding."""

    start: int
    end: int
    score: float = 0.0


class EmbeddingAlignmentSegmenter:
    """Sliding-window embedding alignment using a prototype database."""

    _STRATEGIES = {"pruned", "exhaustive"}

    def __init__(
        self,
        prototype_db: PrototypeSearchDB,
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
        strategy: str = "pruned",
        max_cut_candidates: int | None = None,
        min_cut_spacing: float | None = None,
        max_edges_per_start: int = 3,
        fallback_exhaustive: bool = True,
    ) -> None:
        if strategy not in self._STRATEGIES:
            msg = (
                f"strategy must be one of {sorted(self._STRATEGIES)}; got {strategy!r}"
            )
            raise ValueError(msg)

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
        self.strategy = strategy
        self.max_cut_candidates = max_cut_candidates
        self.min_cut_spacing = min_cut_spacing
        self.max_edges_per_start = max(1, max_edges_per_start)
        self.fallback_exhaustive = fallback_exhaustive
        self._proposer = VisualWordSegmenter()

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

    def _cut_budget(self, height: int, width: int) -> int:
        if self.max_cut_candidates is not None:
            return max(4, self.max_cut_candidates)
        unit_width = max(float(self.min_unit_width), float(height) * 0.5)
        expected_units = max(1, int(round(float(width) / unit_width)))
        return min(128, max(20, expected_units + 4))

    def _cut_spacing(self, height: int) -> float:
        if self.min_cut_spacing is not None:
            return max(1.0, self.min_cut_spacing)
        unit_width = max(float(self.min_unit_width), float(height) * 0.5)
        return max(2.0, unit_width * 0.5)

    def _target_width(self, height: int) -> float:
        return max(float(self.min_unit_width), float(height) * 0.5)

    def _pruned_windows(
        self, word_crop: np.ndarray, height: int, width: int
    ) -> tuple[list[_CandidateWindow], int, int]:
        cuts = self._proposer.propose_cut_candidates(
            word_crop,
            max_candidates=self._cut_budget(height, width),
            min_spacing=self._cut_spacing(height),
        )
        boundaries = sorted({0, width, *(cut.x for cut in cuts if 0 <= cut.x <= width)})
        scores = {cut.x: cut.score for cut in cuts}
        max_width = min(
            width,
            self.max_unit_width or max(self.min_unit_width, int(round(height * 2))),
        )
        target_width = self._target_width(height)

        windows: dict[tuple[int, int], _CandidateWindow] = {}
        for start in boundaries:
            if start >= width:
                continue
            options: list[_CandidateWindow] = []
            for end in boundaries:
                if end <= start:
                    continue
                window_width = end - start
                if window_width < self.min_unit_width or window_width > max_width:
                    continue
                width_penalty = abs(window_width - target_width) / max(
                    target_width, 1.0
                )
                edge_score = scores.get(end, 0.0) - 0.45 * width_penalty
                options.append(_CandidateWindow(start, end, edge_score))

            options.sort(key=lambda item: (-item.score, item.end))
            for candidate in options[: self.max_edges_per_start]:
                windows[(candidate.start, candidate.end)] = candidate

        selected_starts = sum(1 for boundary in boundaries if boundary < width)
        return (
            sorted(windows.values(), key=lambda item: (item.start, item.end)),
            len(boundaries),
            selected_starts,
        )

    def _exhaustive_windows(
        self, height: int, width: int
    ) -> tuple[list[_CandidateWindow], int, int]:
        widths = self._candidate_widths(height, width)
        if not widths:
            return [], 0, 0

        starts: set[int] = set(range(0, width, self.stride))
        starts.add(0)
        for unit_width in widths:
            starts.add(max(0, width - unit_width))
        ordered_starts = sorted(start for start in starts if start < width)

        windows: list[_CandidateWindow] = []
        for start in ordered_starts:
            for unit_width in widths:
                end = start + unit_width
                if end > width:
                    continue
                windows.append(_CandidateWindow(start, end))

        return windows, len(ordered_starts), len(ordered_starts)

    def _decode(
        self,
        word_crop: np.ndarray,
        windows: list[_CandidateWindow],
        height: int,
        *,
        strategy: str,
        boundary_candidates: int,
        selected_starts: int,
        fallback_used: bool,
        rtl: bool,
    ) -> AlignmentResult:
        stats = AlignmentStats(
            strategy=strategy,
            boundary_candidates=boundary_candidates,
            selected_starts=selected_starts,
            candidate_windows=len(windows),
            embedded_windows=len(windows),
            fallback_used=fallback_used,
        )
        if not windows:
            return AlignmentResult(units=(), score=0.0, stats=stats)

        images = [
            self._to_canvas(word_crop[0:height, window.start : window.end])
            for window in windows
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
        for idx, window in enumerate(windows):
            by_start.setdefault(window.start, []).append(idx)

        # Minimize the total alignment cost:
        #   sum(1 - similarity) + unit_penalty * n_units + length_penalty * width
        # The per-unit penalty prevents splitting a perfect block into many
        # tiny perfect windows, while the similarity term still prefers real
        # boundaries over a single mixed full-word window.
        dp: dict[int, float] = {0: 0.0}
        back: dict[int, tuple[int, int]] = {}

        positions = sorted(
            {window.start for window in windows} | {window.end for window in windows}
        )
        for start in positions:
            if start not in dp:
                continue
            for idx in by_start.get(start, []):
                window = windows[idx]
                if window_scores[idx] < self.min_similarity:
                    continue
                candidate = (
                    dp[start]
                    + (1.0 - window_scores[idx])
                    + self.unit_penalty
                    + self.length_penalty * (window.end - window.start)
                )
                if window.end not in dp or candidate < dp[window.end]:
                    dp[window.end] = candidate
                    back[window.end] = (start, idx)

        if word_crop.shape[1] not in dp:
            return AlignmentResult(units=(), score=0.0, stats=stats)

        chosen: list[int] = []
        cursor = word_crop.shape[1]
        while cursor > 0:
            start, idx = back[cursor]
            chosen.append(idx)
            cursor = start
        chosen.reverse()

        units = [
            AlignedUnit(
                image=images[idx],
                bbox=(
                    windows[idx].start,
                    0,
                    windows[idx].end - windows[idx].start,
                    height,
                ),
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
        return AlignmentResult(units=tuple(units), score=mean_similarity, stats=stats)

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

        height, width = word_crop.shape[:2]
        attempts = [self.strategy]
        if self.strategy == "pruned" and self.fallback_exhaustive:
            attempts.append("exhaustive")

        last_result: AlignmentResult | None = None
        for attempt_index, strategy in enumerate(attempts):
            if strategy == "pruned":
                windows, boundaries, starts = self._pruned_windows(
                    word_crop, height, width
                )
            else:
                windows, boundaries, starts = self._exhaustive_windows(height, width)

            result = self._decode(
                word_crop,
                windows,
                height,
                strategy=strategy,
                boundary_candidates=boundaries,
                selected_starts=starts,
                fallback_used=attempt_index > 0,
                rtl=rtl,
            )
            last_result = result
            if result.units:
                return result

        return last_result or AlignmentResult(units=(), score=0.0)
