"""Tests for Arabic vocabulary support, unit segmentation, and alignment."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from embedding_ocr.data.scripts import is_text_in_script
from embedding_ocr.recognition.embedding.hierarchical.alignment import (
    EmbeddingAlignmentSegmenter,
)
from embedding_ocr.recognition.embedding.hierarchical.arabic_units import (
    ArabicUnitSegmenter,
)
from embedding_ocr.recognition.embedding.hierarchical.vocabulary import (
    DEFAULT_ARABIC_FONTS,
    EmbeddingVocabulary,
    render_prototype_image,
)


class DummyEmbedder:
    name = "dummy"
    dim = 8

    def embed_images(self, images: list[np.ndarray]) -> np.ndarray:
        rng = np.random.default_rng(len(images))
        vectors = rng.standard_normal((len(images), self.dim)).astype(np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / norms

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        rng = np.random.default_rng(len(texts))
        vectors = rng.standard_normal((len(texts), self.dim)).astype(np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / norms


class MeanColorEmbedder:
    """Maps an image to its normalized mean RGB vector."""

    name = "mean-color"
    dim = 3

    def embed_images(self, images: list[np.ndarray]) -> np.ndarray:
        vectors = []
        for image in images:
            mean = image.astype(np.float32).reshape(-1, 3).mean(axis=0)
            norm = float(np.linalg.norm(mean))
            vectors.append(mean / norm if norm else mean)
        return np.asarray(vectors, dtype=np.float32)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        return np.zeros((len(texts), self.dim), dtype=np.float32)


class NonWhiteMeanColorEmbedder(MeanColorEmbedder):
    """Mean RGB of non-white pixels, ignoring canvas padding."""

    def embed_images(self, images: list[np.ndarray]) -> np.ndarray:
        vectors = []
        for image in images:
            pixels = image.astype(np.float32)
            mask = (pixels < 250).any(axis=2)
            mean = (
                pixels[mask].mean(axis=0)
                if bool(mask.any())
                else pixels.reshape(-1, 3).mean(axis=0)
            )
            norm = float(np.linalg.norm(mean))
            vectors.append(mean / norm if norm else mean)
        return np.asarray(vectors, dtype=np.float32)


class FakePrototypeDB:
    """Tiny in-memory RGB prototype store for the alignment test."""

    def __init__(self) -> None:
        self.embedder = MeanColorEmbedder()
        self.labels = ["R", "G", "B"]
        self.vectors = np.eye(3, dtype=np.float32)

    def search(
        self, queries: np.ndarray, top_k: int = 5
    ) -> tuple[np.ndarray, list[list[str]]]:
        sims = queries @ self.vectors.T
        ids = np.argsort(-sims, axis=1)[:, :top_k]
        row_sims = np.take_along_axis(sims, ids, axis=1)
        labels = [[self.labels[i] for i in row] for row in ids]
        return row_sims, labels


def test_script_aware_token_filter() -> None:
    assert is_text_in_script("مرحبا", "arabic")
    assert not is_text_in_script("hello", "arabic")
    assert is_text_in_script("hello", "latin")
    assert is_text_in_script("مرحبا", "auto")


def test_embedding_vocabulary_arabic_defaults(tmp_path: Path) -> None:
    vocab = EmbeddingVocabulary(
        embedder=DummyEmbedder(),
        cache_dir=tmp_path / "cache",
        script="arabic",
    )
    assert vocab.script == "arabic"
    installed = [font for font in DEFAULT_ARABIC_FONTS if Path(font).exists()]
    if installed:
        assert vocab.prototype_fonts == installed


def test_render_prototype_image_arabic() -> None:
    font = next(
        (font for font in DEFAULT_ARABIC_FONTS if Path(font).exists()),
        None,
    )
    if font is None:
        pytest.skip("No Arabic font installed")

    image = render_prototype_image("مرحبا", font_path=font, script="arabic")
    assert image.ndim == 3
    assert image.shape[2] == 3
    assert image.min() < 200


def test_alignment_recovers_ltr_and_rtl_units() -> None:
    # Three adjacent full-height color blocks: R, G, B.
    image = np.zeros((20, 60, 3), dtype=np.uint8)
    image[:, 0:20] = (255, 0, 0)
    image[:, 20:40] = (0, 255, 0)
    image[:, 40:60] = (0, 0, 255)

    db = FakePrototypeDB()
    aligner = EmbeddingAlignmentSegmenter(
        prototype_db=db,  # type: ignore[arg-type]
        min_unit_width=8,
        max_unit_width=40,
        stride=1,
        width_step=4,
        min_similarity=0.9,
        unit_penalty=0.1,
    )

    ltr = aligner.align(image, rtl=False)
    assert ltr.text == "RGB"
    assert len(ltr.units) == 3

    rtl = aligner.align(image, rtl=True)
    assert rtl.text == "BGR"
    assert len(rtl.units) == 3


def _long_color_word() -> np.ndarray:
    image = np.full((40, 440, 3), 255, dtype=np.uint8)
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    for i in range(20):
        start = i * 22
        image[:, start : start + 20] = colors[i % 3]
    return image


def test_pruned_alignment_embeds_few_candidates_for_long_word() -> None:
    image = _long_color_word()
    db = FakePrototypeDB()
    db.embedder = NonWhiteMeanColorEmbedder()
    aligner = EmbeddingAlignmentSegmenter(
        prototype_db=db,  # type: ignore[arg-type]
        min_unit_width=8,
        max_unit_width=60,
        stride=4,
        width_step=4,
        min_similarity=0.7,
        unit_penalty=0.1,
        strategy="pruned",
    )

    result = aligner.align(image)

    assert result.text.startswith("RGBRGB")
    assert len(result.units) == 20
    assert result.stats is not None
    assert result.stats.strategy == "pruned"
    assert result.stats.fallback_used is False
    assert result.stats.candidate_windows == result.stats.embedded_windows
    assert result.stats.candidate_windows <= 50
    assert result.stats.embedded_windows < 100


def test_pruned_alignment_falls_back_to_exhaustive_when_no_path() -> None:
    image = np.zeros((20, 60, 3), dtype=np.uint8)
    image[:, 0:20] = (255, 0, 0)
    image[:, 20:40] = (0, 255, 0)
    image[:, 40:60] = (0, 0, 255)

    db = FakePrototypeDB()
    aligner = EmbeddingAlignmentSegmenter(
        prototype_db=db,  # type: ignore[arg-type]
        min_unit_width=8,
        max_unit_width=40,
        stride=1,
        width_step=4,
        min_similarity=0.9,
        unit_penalty=0.1,
        strategy="pruned",
        max_cut_candidates=4,
        fallback_exhaustive=True,
    )

    result = aligner.align(image)

    assert result.text == "RGB"
    assert result.stats is not None
    assert result.stats.strategy == "exhaustive"
    assert result.stats.fallback_used is True


def test_arabic_unit_segmenter_merges_diacritics_and_orders_rtl() -> None:
    image = Image.new("RGB", (80, 40), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)
    # Two base letters.
    draw.rectangle([10, 15, 29, 34], fill=(0, 0, 0))
    draw.rectangle([40, 15, 59, 34], fill=(0, 0, 0))
    # Small diacritic above the right base.
    draw.rectangle([47, 8, 50, 11], fill=(0, 0, 0))

    segmenter = ArabicUnitSegmenter(min_component_area=1)
    units = segmenter.segment(np.array(image, dtype=np.uint8), rtl=True)

    assert len(units) == 2
    assert units[0].bbox[0] > units[1].bbox[0]
    # The right unit must include the merged diacritic above it.
    assert units[0].bbox[1] <= 8
    assert units[0].bbox[3] + units[0].bbox[1] >= 35
    assert units[0].image.shape == (64, 64, 3)


def test_render_prototype_image_falls_back_without_raqm(monkeypatch) -> None:
    font = next(
        (font for font in DEFAULT_ARABIC_FONTS if Path(font).exists()),
        None,
    )
    if font is None:
        pytest.skip("No Arabic font installed")

    original_textbbox = ImageDraw.ImageDraw.textbbox
    original_text = ImageDraw.ImageDraw.text

    def textbbox_without_raqm(self, xy, text, *args, **kwargs):
        if kwargs.get("direction") is not None:
            raise KeyError(
                "setting text direction, language or font features is not "
                "supported without libraqm"
            )
        return original_textbbox(self, xy, text, *args, **kwargs)

    def text_without_raqm(self, xy, text, *args, **kwargs):
        if kwargs.get("direction") is not None:
            raise KeyError(
                "setting text direction, language or font features is not "
                "supported without libraqm"
            )
        return original_text(self, xy, text, *args, **kwargs)

    monkeypatch.setattr(ImageDraw.ImageDraw, "textbbox", textbbox_without_raqm)
    monkeypatch.setattr(ImageDraw.ImageDraw, "text", text_without_raqm)
    image = render_prototype_image("مرحبا", font_path=font, script="arabic")
    assert image.ndim == 3
    assert image.shape[2] == 3
    assert image.min() < 250
