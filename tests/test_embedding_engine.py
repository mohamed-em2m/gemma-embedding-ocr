"""Unit tests for Phase 4 EmbeddingRecognizer pipeline components."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from embedding_ocr.recognition.base import RecognitionResult
from embedding_ocr.recognition.embedding.engine import EmbeddingRecognizer
from embedding_ocr.recognition.embedding.index import NumpyIndex
from embedding_ocr.recognition.embedding.matcher import PrototypeMatcher
from embedding_ocr.recognition.embedding.prototypes import PrototypeDB
from embedding_ocr.recognition.embedding.segmenter import CharacterSegmenter


class DummyEmbedder:
    """Mock embedder returning deterministic dummy embeddings."""

    name = "dummy"
    dim = 32

    def embed_images(self, images: list[np.ndarray]) -> np.ndarray:
        n = len(images)
        # Generate predictable vectors based on mean pixel intensities
        rng = np.random.default_rng(len(images))
        vecs = rng.standard_normal((n, self.dim)).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / norms

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        n = len(texts)
        rng = np.random.default_rng(len(texts))
        vecs = rng.standard_normal((n, self.dim)).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / norms


def test_character_segmenter() -> None:
    # Render two separated letters 'A B' with a gap
    img = Image.new("RGB", (140, 50), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([10, 10, 30, 40], fill=(0, 0, 0))
    draw.rectangle([70, 10, 90, 40], fill=(0, 0, 0))

    segmenter = CharacterSegmenter()
    segments = segmenter.segment(np.array(img))

    assert len(segments) == 2
    assert segments[0].bbox[0] < segments[1].bbox[0]
    assert segments[0].image.shape == (64, 64, 3)
    assert not segments[0].is_space_before
    assert segments[1].is_space_before


def test_prototype_matcher_case_resolution() -> None:
    matcher = PrototypeMatcher(case_height_thresh=0.65)
    sims = np.array([0.95, 0.80])
    labels = ["c", "o"]

    # When relative height is tall, 'c' promotes to 'C'
    res_tall = matcher.match_character(sims, labels, relative_height=0.75)
    assert res_tall.character == "C"

    # When relative height is short, 'c' stays 'c'
    res_short = matcher.match_character(sims, labels, relative_height=0.45)
    assert res_short.character == "c"


def test_embedding_recognizer_end_to_end() -> None:
    embedder = DummyEmbedder()
    index = NumpyIndex(dim=embedder.dim)
    db = PrototypeDB(embedder=embedder, index=index, classes=["A", "B", "C"])

    # Build DB with dummy vectors
    vecs = embedder.embed_images([np.zeros((64, 64, 3), dtype=np.uint8)] * 3)
    db.labels = ["A", "B", "C"]
    index.build(vecs, np.arange(3))

    recognizer = EmbeddingRecognizer(prototype_db=db)
    test_img = np.full((50, 140, 3), 255, dtype=np.uint8)
    test_img[10:40, 15:35] = 0
    test_img[10:40, 75:95] = 0

    res = recognizer.recognize(test_img)
    assert isinstance(res, RecognitionResult)
    # 2 letters + 1 space in between
    assert len(res.text) == 3
    assert res.text[1] == " "
    assert 0.0 <= res.confidence <= 1.0
