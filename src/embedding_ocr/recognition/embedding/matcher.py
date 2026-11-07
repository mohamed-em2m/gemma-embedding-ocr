"""Character matcher with per-class aggregation and case resolution."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from embedding_ocr.data.glyph_renderer import CASE_PAIRS
from embedding_ocr.data.scripts import is_rtl_script, normalize_script_name


@dataclass(frozen=True)
class MatchPrediction:
    """Prediction for a single character."""

    character: str
    confidence: float
    raw_top1_sim: float
    margin: float


class PrototypeMatcher:
    """Aggregates prototype search hits per character class and resolves case."""

    def __init__(
        self,
        case_height_thresh: float = 0.65,
        script: str = "latin",
        case_sensitive: bool | None = None,
    ) -> None:
        """Initialize matcher.

        Args:
            case_height_thresh: Relative height threshold to disambiguate
                lowercase vs uppercase Latin letters.
            script: Script of the prototype classes. RTL scripts such as
                Arabic have no case, so Latin case-pair logic is disabled.
            case_sensitive: Explicit override for case resolution. When
                ``None``, it is enabled only for non-RTL scripts.
        """
        self.case_height_thresh = case_height_thresh
        self.script = normalize_script_name(script)
        self.case_sensitive = (
            case_sensitive
            if case_sensitive is not None
            else not is_rtl_script(self.script)
        )
        self.case_map = dict(CASE_PAIRS)
        self.reverse_case_map = {v: k for k, v in CASE_PAIRS}

    @staticmethod
    def _is_latin_letter(char: str) -> bool:
        return len(char) == 1 and char.isascii() and char.isalpha()

    def match_character(
        self,
        similarities: np.ndarray,
        labels: list[str],
        relative_height: float | None = None,
        relative_top: float | None = None,
        relative_bottom: float | None = None,
        baseline_bottom: float | None = None,
        cap_height_top: float | None = None,
    ) -> MatchPrediction:
        """Find the best character match from prototype hits.

        Args:
            similarities: 1D array of cosine similarities for the top-k retrieved prototypes.
            labels: List of character strings corresponding to the similarities.
            relative_height: Relative height of the character in the word line.
            relative_top: Normalized top coordinate (y / H).
            relative_bottom: Normalized bottom coordinate ((y + h) / H).
            baseline_bottom: Estimated line baseline position.
            cap_height_top: Estimated line uppercase cap-height position.

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
        sorted_classes = sorted(
            class_scores.items(), key=lambda item: item[1], reverse=True
        )
        if not sorted_classes:
            return MatchPrediction(
                character="", confidence=0.0, raw_top1_sim=0.0, margin=0.0
            )

        best_char, top1_sim = sorted_classes[0]
        top2_sim = sorted_classes[1][1] if len(sorted_classes) > 1 else 0.0
        margin = max(0.0, top1_sim - top2_sim)

        # Confidence mapped from similarity and margin
        conf = max(0.0, min(1.0, (top1_sim + margin) / 1.5))

        # Disambiguate case pairs using geometric relative height and baseline
        # cues. This only applies to Latin letters; Arabic/Hebrew and other
        # caseless scripts must never be remapped through CASE_PAIRS.
        is_case_candidate = (
            self.case_sensitive
            and self._is_latin_letter(best_char)
            and (best_char in self.case_map or best_char in self.reverse_case_map)
        )
        if is_case_candidate:
            # Baseline & Cap-height analysis if available
            is_uppercase: bool | None = None
            if relative_top is not None and cap_height_top is not None:
                # If top reaches near cap-height, it's uppercase
                is_uppercase = relative_top <= cap_height_top + 0.08

            if is_uppercase is None and relative_height is not None:
                is_uppercase = relative_height >= self.case_height_thresh

            if is_uppercase is not None:
                if is_uppercase and best_char in self.case_map:
                    best_char = self.case_map[best_char]
                elif not is_uppercase and best_char in self.reverse_case_map:
                    # Special check: descenders like 'p'
                    if (
                        best_char == "P"
                        and relative_bottom is not None
                        and baseline_bottom is not None
                    ):
                        if relative_bottom > baseline_bottom + 0.05:
                            best_char = "p"
                    else:
                        best_char = self.reverse_case_map[best_char]

        return MatchPrediction(
            character=best_char,
            confidence=conf,
            raw_top1_sim=top1_sim,
            margin=margin,
        )
