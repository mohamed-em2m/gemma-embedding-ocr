"""Character matcher with per-class aggregation and case resolution."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from embedding_ocr.data.glyph_renderer import CASE_PAIRS


@dataclass(frozen=True)
class MatchPrediction:
    """Prediction for a single character."""

    character: str
    confidence: float
    raw_top1_sim: float
    margin: float


class PrototypeMatcher:
    """Aggregates prototype search hits per character class and resolves case."""

    def __init__(self, case_height_thresh: float = 0.65) -> None:
        """Initialize matcher.

        Args:
            case_height_thresh: Relative height threshold to disambiguate lowercase vs uppercase.
        """
        self.case_height_thresh = case_height_thresh
        self.case_map = dict(CASE_PAIRS)
        self.reverse_case_map = {v: k for k, v in CASE_PAIRS}

    def match_character(
        self,
        similarities: np.ndarray,
        labels: list[str],
        relative_height: float | None = None,
    ) -> MatchPrediction:
        """Find the best character match from prototype hits.

        Args:
            similarities: 1D array of cosine similarities for the top-k retrieved prototypes.
            labels: List of character strings corresponding to the similarities.
            relative_height: Relative height of the character in the word line.

        Returns:
            MatchPrediction with predicted character and confidence score.
        """
        # Per-class max aggregation
        class_scores: dict[str, float] = {}
        for sim, lbl in zip(similarities, labels, strict=False):
            score = float(sim)
            if lbl not in class_scores or score > class_scores[lbl]:
                class_scores[lbl] = score

        # Sort classes by max similarity
        sorted_classes = sorted(class_scores.items(), key=lambda item: item[1], reverse=True)
        if not sorted_classes:
            return MatchPrediction(character="", confidence=0.0, raw_top1_sim=0.0, margin=0.0)

        best_char, top1_sim = sorted_classes[0]
        top2_sim = sorted_classes[1][1] if len(sorted_classes) > 1 else 0.0
        margin = max(0.0, top1_sim - top2_sim)

        # Confidence mapped from similarity and margin
        conf = max(0.0, min(1.0, (top1_sim + margin) / 1.5))

        # Disambiguate case pairs using geometric relative height
        if relative_height is not None:
            # If lowercase candidate but glyph is full line height -> promote to uppercase
            if best_char in self.case_map and relative_height >= self.case_height_thresh:
                best_char = self.case_map[best_char]
            # If uppercase candidate but glyph is small (< threshold) -> demote to lowercase
            elif best_char in self.reverse_case_map and relative_height < self.case_height_thresh:
                best_char = self.reverse_case_map[best_char]

        return MatchPrediction(
            character=best_char,
            confidence=conf,
            raw_top1_sim=top1_sim,
            margin=margin,
        )
