"""Hierarchical Word Recognizer implementing RecognitionEngine.

Executes:
1. Complete-word vector matching with confidence and margin checks.
2. Recursive visual subword splitting when complete-word confidence fails.
3. Fallback to character-level recognition.
4. Word boundary handling and spaces in text lines.
"""

from __future__ import annotations

import numpy as np

from embedding_ocr.pipeline.preprocess import preprocess_handwriting_crop
from embedding_ocr.recognition.base import RecognitionEngine, RecognitionResult
from embedding_ocr.recognition.embedding.embedder import Embedder
from embedding_ocr.recognition.embedding.hierarchical.craft_detector import (
    CraftWordDetector,
)
from embedding_ocr.recognition.embedding.hierarchical.segmentation import (
    VisualWordSegmenter,
)
from embedding_ocr.recognition.embedding.hierarchical.vocabulary import (
    EmbeddingVocabulary,
)


class HierarchicalWordRecognizer(RecognitionEngine):
    """Recognition engine replacing character-only recognition with hierarchical word recognition."""

    def __init__(
        self,
        embedder: Embedder,
        vocabulary: EmbeddingVocabulary,
        segmenter: VisualWordSegmenter | None = None,
        craft_detector: CraftWordDetector | None = None,
        use_craft: bool = False,
        min_word_sim: float = 0.74,
        min_word_margin: float = 0.008,
        min_subword_sim: float = 0.72,
        min_subword_margin: float = 0.005,
        min_char_sim: float = 0.65,
        max_recursion_depth: int = 4,
        top_k: int = 15,
        preprocess_handwriting: bool = False,
    ) -> None:
        self.embedder = embedder
        self.vocabulary = vocabulary
        self.segmenter = segmenter or VisualWordSegmenter()
        self.craft_detector = craft_detector
        self.use_craft = use_craft
        if self.use_craft and self.craft_detector is None:
            self.craft_detector = CraftWordDetector()
        self.min_word_sim = min_word_sim
        self.min_word_margin = min_word_margin
        self.min_subword_sim = min_subword_sim
        self.min_subword_margin = min_subword_margin
        self.min_char_sim = min_char_sim
        self.max_recursion_depth = max_recursion_depth
        self.top_k = top_k
        self.preprocess_handwriting = preprocess_handwriting

        # In-memory LRU-style cache for repeated word/subword visual crops
        self._cache: dict[bytes, tuple[str, float]] = {}

        # Tracking telemetry
        self.stats = {
            "total_words": 0,
            "full_word_hits": 0,
            "subword_fallbacks": 0,
            "char_fallbacks": 0,
            "unresolved_fallbacks": 0,
        }

    def _hash_crop(self, crop: np.ndarray) -> bytes:
        import hashlib

        return hashlib.md5(crop.tobytes()).digest()

    def recognize_word_crop(
        self, word_crop: np.ndarray, depth: int = 0
    ) -> tuple[str, float]:
        """Recognize a single word or subword crop hierarchically."""
        h, w = word_crop.shape[:2]
        if w < 6 or h < 6:
            return "", 0.0

        if depth == 0 and self.preprocess_handwriting:
            word_crop = preprocess_handwriting_crop(word_crop)
            h, w = word_crop.shape[:2]
            if w < 6 or h < 6:
                return "", 0.0

        crop_hash = self._hash_crop(word_crop)
        if crop_hash in self._cache:
            return self._cache[crop_hash]

        # 1. Embed word crop
        query_emb = self.embedder.embed_images([word_crop])

        # If at depth 0, attempt full-word candidate retrieval
        if depth == 0 and self.vocabulary.word_index is not None:
            sims, tokens = self.vocabulary.search_words(query_emb, top_k=self.top_k)
            if len(tokens[0]) >= 2:
                top1_tok = tokens[0][0]
                sim1 = float(sims[0][0])
                sim2 = float(sims[0][1])
                margin = sim1 - sim2

                # Case normalization check: if top1 and top2 are the same word in different case (e.g. system vs System),
                # compute margin against the next truly different word candidate.
                if top1_tok.text.lower() == tokens[0][1].text.lower():
                    for alt_tok, alt_sim in zip(
                        tokens[0][2:], sims[0][2:], strict=False
                    ):
                        if alt_tok.text.lower() != top1_tok.text.lower():
                            margin = sim1 - float(alt_sim)
                            break

                if sim1 >= self.min_word_sim and margin >= self.min_word_margin:
                    self.stats["full_word_hits"] += 1
                    result = (top1_tok.text, sim1)
                    self._cache[crop_hash] = result
                    return result

        # 2. Try subword candidate retrieval if depth > 0 or if full word search fell through
        if self.vocabulary.subword_index is not None:
            sims, tokens = self.vocabulary.search_subwords(query_emb, top_k=self.top_k)
            if len(tokens[0]) >= 2:
                top1_tok = tokens[0][0]
                sim1 = float(sims[0][0])
                sim2 = float(sims[0][1])
                margin = sim1 - sim2

                if top1_tok.text.lower() == tokens[0][1].text.lower():
                    for alt_tok, alt_sim in zip(
                        tokens[0][2:], sims[0][2:], strict=False
                    ):
                        if alt_tok.text.lower() != top1_tok.text.lower():
                            margin = sim1 - float(alt_sim)
                            break

                # Accept subword if high confidence or reached narrow region
                if sim1 >= self.min_subword_sim and margin >= self.min_subword_margin:
                    self.stats["subword_fallbacks"] += 1
                    result = (top1_tok.text, sim1)
                    self._cache[crop_hash] = result
                    return result

        # 3. Recursive splitting if depth limit not reached and width permits
        # Only split if width is wide enough to contain multiple glyphs
        if depth < self.max_recursion_depth and w >= 32:
            left_crop, right_crop, cut_x = self.segmenter.split_word_image(word_crop)

            # Ensure valid cut progress
            if cut_x >= 12 and (w - cut_x) >= 12:
                left_text, left_conf = self.recognize_word_crop(
                    left_crop, depth=depth + 1
                )
                right_text, right_conf = self.recognize_word_crop(
                    right_crop, depth=depth + 1
                )

                if left_text and right_text and left_text != "?" and right_text != "?":
                    combined_text = left_text + right_text
                    combined_conf = (left_conf + right_conf) / 2.0
                    result = (combined_text, combined_conf)
                    self._cache[crop_hash] = result
                    return result

        # 4. Fallback to single character matching
        if self.vocabulary.char_index is not None:
            sims, tokens = self.vocabulary.search_chars(query_emb, top_k=self.top_k)
            if tokens[0]:
                top1_tok = tokens[0][0]
                sim1 = float(sims[0][0])
                if sim1 >= self.min_char_sim:
                    self.stats["char_fallbacks"] += 1
                    result = (top1_tok.text, sim1)
                    self._cache[crop_hash] = result
                    return result

        # If unresolved, do not invent text
        self.stats["unresolved_fallbacks"] += 1
        result = ("?", 0.3)
        self._cache[crop_hash] = result
        return result

    def recognize(self, image: np.ndarray) -> RecognitionResult:
        """Recognize text in a single text region (line or multi-word region).

        Implements RecognitionEngine Protocol.

        Args:
            image: Text-region image as uint8 numpy array (H, W, C).

        Returns:
            RecognitionResult with recognized string and overall confidence.
        """
        if image is None or image.size == 0:
            return RecognitionResult(text="", confidence=0.0)

        # 1. Propose individual word crops using CRAFT or visual gap segmentation
        if self.use_craft and self.craft_detector is not None:
            word_boxes = self.craft_detector.detect_word_boxes(image)
        else:
            word_boxes = self.segmenter.extract_word_boxes(image)

        recognized_words: list[str] = []
        confidences: list[float] = []

        for x, y, w, h in word_boxes:
            word_crop = image[y : y + h, x : x + w]
            self.stats["total_words"] += 1

            text, conf = self.recognize_word_crop(word_crop, depth=0)
            if text:
                recognized_words.append(text)
                confidences.append(conf)

        final_text = " ".join(recognized_words)
        mean_conf = float(np.mean(confidences)) if confidences else 0.0

        return RecognitionResult(
            text=final_text,
            confidence=max(0.0, min(1.0, mean_conf)),
        )
