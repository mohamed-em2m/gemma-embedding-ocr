"""EmbeddingRecognizer implementing RecognitionEngine."""

from __future__ import annotations

import numpy as np

from embedding_ocr.data.scripts import is_rtl_script, normalize_script_name
from embedding_ocr.recognition.base import RecognitionEngine, RecognitionResult
from embedding_ocr.recognition.embedding.matcher import PrototypeMatcher
from embedding_ocr.recognition.embedding.prototypes import PrototypeDB
from embedding_ocr.recognition.embedding.segmenter import CharacterSegmenter


class EmbeddingRecognizer(RecognitionEngine):
    """Recognition engine utilizing visual embeddings + vector prototype search."""

    def __init__(
        self,
        prototype_db: PrototypeDB,
        segmenter: CharacterSegmenter | None = None,
        matcher: PrototypeMatcher | None = None,
        top_k: int = 25,
        script: str | None = None,
        rtl: bool | None = None,
    ) -> None:
        self.prototype_db = prototype_db
        # Default to the prototype DB's script so an Arabic DB does not
        # silently fall back to Latin case/reading-order rules.
        resolved_script = script if script is not None else prototype_db.script
        self.script = normalize_script_name(resolved_script)
        self.rtl = is_rtl_script(self.script) if rtl is None else rtl
        self.segmenter = segmenter or CharacterSegmenter()
        self.matcher = matcher or PrototypeMatcher(script=self.script)
        self.top_k = top_k

    def recognize(self, image: np.ndarray) -> RecognitionResult:
        """Recognize text in an image region using character embedding matching.

        Args:
            image: uint8 numpy array of text region (H, W, 3).

        Returns:
            RecognitionResult with recognized string and mean confidence.
        """
        # 1. Segment into characters in logical reading order
        segments = self.segmenter.segment(image, rtl=self.rtl)
        if not segments:
            return RecognitionResult(text="", confidence=0.0)

        # 2. Extract batch embeddings for all characters in region
        char_images = [seg.image for seg in segments]
        embeddings = self.prototype_db.embedder.embed_images(char_images)

        # 3. Vector prototype search
        sims, retrieved_labels = self.prototype_db.search(embeddings, top_k=self.top_k)

        # 4. Match characters and aggregate
        recognized_chars: list[str] = []
        confidences: list[float] = []

        # Estimate line baseline and cap-height from segmented components
        cap_height_top = min((seg.relative_top for seg in segments), default=0.2)
        baseline_bottom = (
            float(np.median([seg.relative_bottom for seg in segments]))
            if segments
            else 0.8
        )

        for i, seg in enumerate(segments):
            if seg.is_space_before and recognized_chars:
                recognized_chars.append(" ")

            match = self.matcher.match_character(
                similarities=sims[i],
                labels=retrieved_labels[i],
                relative_height=seg.relative_height,
                relative_top=seg.relative_top,
                relative_bottom=seg.relative_bottom,
                baseline_bottom=baseline_bottom,
                cap_height_top=cap_height_top,
            )
            recognized_chars.append(match.character)
            confidences.append(match.confidence)

        text = "".join(recognized_chars)
        avg_confidence = float(np.mean(confidences)) if confidences else 0.0

        return RecognitionResult(
            text=text, confidence=max(0.0, min(1.0, avg_confidence))
        )
