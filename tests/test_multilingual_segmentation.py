"""Tests for script detection, RTL ordering, and multilingual prototype setup."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from embedding_ocr.data.glyph_renderer import (
    ARABIC_CLASSES,
    LATIN_CLASSES,
    classes_for_script,
)
from embedding_ocr.data.scripts import (
    is_rtl_script,
    is_rtl_text,
    normalize_script_name,
    script_for_text,
)
from embedding_ocr.recognition.embedding.hierarchical.craft_detector import (
    CraftWordDetector,
)
from embedding_ocr.recognition.embedding.hierarchical.engine import (
    HierarchicalWordRecognizer,
)
from embedding_ocr.recognition.embedding.hierarchical.segmentation import (
    VisualWordSegmenter,
)
from embedding_ocr.recognition.embedding.index import NumpyIndex
from embedding_ocr.recognition.embedding.matcher import PrototypeMatcher
from embedding_ocr.recognition.embedding.prototypes import PrototypeDB
from embedding_ocr.recognition.embedding.segmenter import CharacterSegmenter


class DummyEmbedder:
    """Deterministic embedder for tests that never downloads a model."""

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


def _two_block_line() -> np.ndarray:
    """Two separated blocks: black on the left, gray on the right."""
    image = Image.new("RGB", (140, 40), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.rectangle([10, 5, 29, 34], fill=(0, 0, 0))
    draw.rectangle([80, 5, 99, 34], fill=(100, 100, 100))
    return np.array(image, dtype=np.uint8)


def test_script_detection_and_aliases() -> None:
    assert script_for_text("hello") == "latin"
    assert script_for_text("مرحبا") == "arabic"
    assert script_for_text("12345") == "unknown"
    assert is_rtl_text("مرحبا") is True
    assert is_rtl_text("hello") is False
    assert is_rtl_script("ar") is True
    assert is_rtl_script("he") is True
    assert is_rtl_script("latin") is False
    assert normalize_script_name("fa") == "arabic"
    assert normalize_script_name("Latin") == "latin"


def test_arabic_classes_are_script_aware() -> None:
    assert "ا" in ARABIC_CLASSES
    assert "ي" in ARABIC_CLASSES
    assert len(ARABIC_CLASSES) >= 28
    assert classes_for_script("arabic") == ARABIC_CLASSES
    assert classes_for_script("latin") == LATIN_CLASSES
    assert classes_for_script("unknown-script") == LATIN_CLASSES


def test_prototype_db_uses_script_classes_when_unspecified() -> None:
    embedder = DummyEmbedder()
    index = NumpyIndex(dim=embedder.dim)
    db = PrototypeDB(embedder=embedder, index=index, script="arabic")
    assert db.script == "arabic"
    assert db.classes == ARABIC_CLASSES


def test_prototype_db_persists_script(tmp_path) -> None:
    embedder = DummyEmbedder()
    index = NumpyIndex(dim=embedder.dim)
    db = PrototypeDB(embedder=embedder, index=index, script="arabic")

    vectors = embedder.embed_images(
        [np.zeros((8, 8, 3), dtype=np.uint8)] * len(db.classes)
    )
    index.build(vectors, np.arange(len(db.classes)))

    path = tmp_path / "arabic_prototypes"
    db.save(path)

    restored = PrototypeDB(
        embedder=embedder,
        index=NumpyIndex(dim=embedder.dim),
        script="latin",
    )
    restored.load_cache(path)
    assert restored.script == "arabic"
    assert restored.classes == ARABIC_CLASSES


def test_matcher_does_not_apply_latin_case_logic_to_arabic() -> None:
    matcher = PrototypeMatcher(script="arabic")
    prediction = matcher.match_character(
        similarities=np.array([0.95, 0.80]),
        labels=["ا", "ب"],
        relative_height=0.95,
    )
    assert prediction.character == "ا"


def test_matcher_still_resolves_latin_case() -> None:
    matcher = PrototypeMatcher(script="latin")
    prediction = matcher.match_character(
        similarities=np.array([0.95, 0.80]),
        labels=["c", "o"],
        relative_height=0.75,
    )
    assert prediction.character == "C"

    disabled = PrototypeMatcher(script="latin", case_sensitive=False)
    assert (
        disabled.match_character(
            similarities=np.array([0.95, 0.80]),
            labels=["c", "o"],
            relative_height=0.75,
        ).character
        == "c"
    )


def test_character_segmenter_returns_rtl_reading_order() -> None:
    image = _two_block_line()
    segmenter = CharacterSegmenter()

    ltr = segmenter.segment(image)
    assert ltr[0].bbox[0] < ltr[1].bbox[0]
    assert ltr[1].is_space_before is True

    rtl = segmenter.segment(image, rtl=True)
    assert rtl[0].bbox[0] > rtl[1].bbox[0]
    assert rtl[1].is_space_before is True


def test_word_segmenter_returns_rtl_reading_order() -> None:
    image = _two_block_line()
    segmenter = VisualWordSegmenter()

    ltr = segmenter.extract_word_boxes(image)
    assert len(ltr) == 2
    assert ltr[0][0] < ltr[1][0]

    rtl = segmenter.extract_word_boxes(image, rtl=True)
    assert len(rtl) == 2
    assert rtl[0][0] > rtl[1][0]


def test_craft_detector_orders_rtl() -> None:
    class FakeCraft:
        def detect_text(self, image: np.ndarray) -> dict[str, list[list[list[int]]]]:
            return {
                "boxes": [
                    [[10, 5], [29, 5], [29, 34], [10, 34]],
                    [[80, 5], [99, 5], [99, 34], [80, 34]],
                ]
            }

    detector = CraftWordDetector(cuda=False)
    detector._craft = FakeCraft()  # type: ignore[assignment]

    image = _two_block_line()
    rtl = detector.detect_word_boxes(image, rtl=True)
    assert rtl[0][0] > rtl[1][0]


def test_prototype_search_clamps_top_k_to_prototype_count() -> None:
    embedder = DummyEmbedder()
    index = NumpyIndex(dim=embedder.dim)
    db = PrototypeDB(
        embedder=embedder,
        index=index,
        classes=["A", "B"],
        script="latin",
    )
    db.labels = list(db.classes)
    vectors = embedder.embed_images(
        [np.zeros((8, 8, 3), dtype=np.uint8)] * len(db.classes)
    )
    index.build(vectors, np.arange(len(db.classes)))

    sims, labels = db.search(vectors[:1], top_k=10)

    assert sims.shape == (1, 2)
    assert len(labels[0]) == 2
    assert set(labels[0]) == {"A", "B"}


def test_hierarchical_recognizer_joins_words_in_reading_order() -> None:
    class DummyVocabulary:
        word_index = None
        subword_index = None
        char_index = None

    def fake_recognize_word_crop(
        word_crop: np.ndarray, depth: int = 0
    ) -> tuple[str, float]:
        # The left block is pure black; the right block is gray.
        return ("LEFT" if int(word_crop.min()) < 50 else "RIGHT", 1.0)

    embedder = DummyEmbedder()
    image = _two_block_line()

    ltr_engine = HierarchicalWordRecognizer(
        embedder=embedder,
        vocabulary=DummyVocabulary(),  # type: ignore[arg-type]
        script="latin",
    )
    ltr_engine.recognize_word_crop = fake_recognize_word_crop  # type: ignore[method-assign]
    assert ltr_engine.recognize(image).text == "LEFT RIGHT"

    rtl_engine = HierarchicalWordRecognizer(
        embedder=embedder,
        vocabulary=DummyVocabulary(),  # type: ignore[arg-type]
        script="arabic",
    )
    rtl_engine.recognize_word_crop = fake_recognize_word_crop  # type: ignore[method-assign]
    assert rtl_engine.recognize(image).text == "RIGHT LEFT"
