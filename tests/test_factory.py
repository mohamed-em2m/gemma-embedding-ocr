"""Tests for configuration-driven factories."""

from __future__ import annotations

import numpy as np
import pytest

from embedding_ocr.config import AppConfig, DetectorSettings, IndexSettings
from embedding_ocr.recognition.factory import (
    build_detector,
    build_index,
    build_recognizer,
    resolve_rtl,
)


def test_build_index_numpy() -> None:
    index = build_index(IndexSettings(backend="numpy", dim=4), dim=4)
    vectors = np.eye(4, dtype=np.float32)
    index.build(vectors, np.arange(4))
    sims, ids = index.search(vectors[:1], k=1)
    assert sims.shape == (1, 1)
    assert int(ids[0, 0]) == 0


def test_build_full_detector() -> None:
    detector = build_detector(DetectorSettings(backend="full"))
    regions = detector.detect(np.zeros((10, 20, 3), dtype=np.uint8))
    assert regions[0].bbox == (0, 0, 20, 10)


def test_resolve_rtl_from_script() -> None:
    config = AppConfig()
    config.recognition.script = "arabic"
    assert resolve_rtl(config) is True
    config.recognition.script = "latin"
    assert resolve_rtl(config) is False


def test_unknown_backends_raise() -> None:
    with pytest.raises(ValueError, match="Unknown detector"):
        build_detector(DetectorSettings(backend="nope"))
    config = AppConfig()
    config.recognition.backend = "nope"
    with pytest.raises(ValueError, match="Unknown recognizer"):
        build_recognizer(config)


class _DummyEmbedder:
    name = "dummy"
    dim = 4

    def embed_images(self, images: list[np.ndarray]) -> np.ndarray:
        return np.ones((len(images), self.dim), dtype=np.float32)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        return np.ones((len(texts), self.dim), dtype=np.float32)


def test_load_prototype_db_validates_metadata(tmp_path) -> None:
    from embedding_ocr.recognition.embedding.index import NumpyIndex
    from embedding_ocr.recognition.embedding.prototypes import PrototypeDB
    from embedding_ocr.recognition.factory import load_prototype_db

    embedder = _DummyEmbedder()
    index = NumpyIndex(dim=4, embedder_name="dummy")
    index.build(np.ones((1, 4), dtype=np.float32), np.array([0]))
    db = PrototypeDB(embedder=embedder, index=index, classes=["A"], script="latin")
    db.labels = ["A"]
    path = tmp_path / "prototypes"
    db.save(path)

    config = AppConfig()
    config.embedder.dim = 4
    config.index.backend = "numpy"
    config.index.path = path
    config.recognition.script = "latin"

    loaded = load_prototype_db(config, embedder)  # type: ignore[arg-type]
    assert loaded.labels == ["A"]

    other = _DummyEmbedder()
    other.name = "other"
    with pytest.raises(ValueError, match="active embedder"):
        load_prototype_db(config, other)  # type: ignore[arg-type]


def test_polygon_helpers() -> None:
    from embedding_ocr.pipeline.detect import PaddleTextDetector, _poly_to_bbox

    assert _poly_to_bbox([[1, 2], [9, 2], [9, 8], [1, 8]], 20, 20) == (1, 2, 8, 6)
    assert PaddleTextDetector._iter_polys({"dt_polys": [[1, 2, 3, 4]]}) == [
        [1, 2, 3, 4]
    ]
    assert PaddleTextDetector._iter_polys({"res": {"dt_polys": [[1, 2, 3, 4]]}}) == [
        [1, 2, 3, 4]
    ]

    class Result:
        json = '{"dt_polys": [[5, 6, 7, 8]]}'

    assert PaddleTextDetector._iter_polys(Result()) == [[5, 6, 7, 8]]


def test_load_prototype_db_missing_index_is_actionable(tmp_path) -> None:
    from embedding_ocr.recognition.factory import load_prototype_db

    config = AppConfig()
    config.index.backend = "numpy"
    config.index.path = tmp_path / "missing"
    config.recognition.script = "latin"

    with pytest.raises(FileNotFoundError, match="build_prototype_cache"):
        load_prototype_db(config, _DummyEmbedder())  # type: ignore[arg-type]


def test_paddle_detector_falls_back_on_onednn_error() -> None:
    from embedding_ocr.pipeline.detect import PaddleTextDetector

    class BrokenPaddle:
        def predict(self, image: np.ndarray) -> list[object]:
            raise NotImplementedError(
                "ConvertPirAttribute2RuntimeAttribute not support"
            )

    detector = object.__new__(PaddleTextDetector)
    detector.rtl = False
    detector._ocr = BrokenPaddle()  # type: ignore[attr-defined]

    regions = detector.detect(np.zeros((12, 20, 3), dtype=np.uint8))
    assert regions[0].bbox == (0, 0, 20, 12)
