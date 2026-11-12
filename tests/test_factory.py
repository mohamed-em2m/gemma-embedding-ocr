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


def test_build_yolo_word_detector_requires_model() -> None:
    with pytest.raises(ValueError, match="yolo_model"):
        build_detector(DetectorSettings(backend="yolo_words"))


def test_build_line_detector() -> None:
    detector = build_detector(DetectorSettings(backend="lines"))
    blank = np.full((10, 20, 3), 255, dtype=np.uint8)
    assert detector.detect(blank) == []


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
    detector._backend = "legacy"
    detector._legacy_ocr = BrokenPaddle()

    regions = detector.detect(np.full((12, 20, 3), 255, dtype=np.uint8))
    assert regions == []


def test_build_ctc_recognizer_requires_checkpoint() -> None:
    config = AppConfig()
    config.recognition.backend = "ctc"
    with pytest.raises(ValueError, match="ctc_checkpoint"):
        build_recognizer(config)


def test_build_ctc_recognizer_loads_a_checkpoint(tmp_path) -> None:
    from embedding_ocr.recognition.ctc.charset import CharSet
    from embedding_ocr.recognition.ctc.checkpoint import save_checkpoint
    from embedding_ocr.recognition.ctc.model import CTCSeq2Seq, ModelConfig

    charset = CharSet(("a", "b", " "))
    model = CTCSeq2Seq(
        ModelConfig(
            vocab_size=charset.size,
            image_height=8,
            channels=(8, 16, 32),
            rnn_hidden=16,
            rnn_layers=1,
            dropout=0.0,
            decoder_layers=1,
            decoder_dim=32,
            decoder_heads=4,
            decoder_ffn=64,
            max_decoder_len=16,
        )
    )
    path = tmp_path / "ctc.pt"
    save_checkpoint(path, model, charset=charset)

    config = AppConfig()
    config.recognition.backend = "ctc"
    config.recognition.ctc_checkpoint = path
    config.recognition.ctc_device = "cpu"
    recognizer = build_recognizer(config)

    image = np.full((24, 64), 255, dtype=np.uint8)
    result = recognizer.recognize(image)
    assert set(result.text) <= set(charset.chars)


def test_build_ctc_recognizer_attaches_a_lexicon(tmp_path) -> None:
    from embedding_ocr.recognition.ctc.charset import CharSet
    from embedding_ocr.recognition.ctc.checkpoint import save_checkpoint
    from embedding_ocr.recognition.ctc.engine import CTCRecognizer
    from embedding_ocr.recognition.ctc.model import CTCSeq2Seq, ModelConfig

    charset = CharSet(("a", "b", " "))
    model = CTCSeq2Seq(
        ModelConfig(
            vocab_size=charset.size,
            image_height=8,
            channels=(8, 16, 32),
            rnn_hidden=16,
            rnn_layers=1,
            dropout=0.0,
            decoder_layers=1,
            decoder_dim=32,
            decoder_heads=4,
            decoder_ffn=64,
            max_decoder_len=16,
        )
    )
    checkpoint = tmp_path / "ctc.pt"
    save_checkpoint(checkpoint, model, charset=charset)
    lexicon = tmp_path / "words.txt"
    lexicon.write_text("ab\nab\nba\nba\n", encoding="utf-8")

    config = AppConfig()
    config.recognition.backend = "ctc"
    config.recognition.ctc_checkpoint = checkpoint
    config.recognition.ctc_device = "cpu"
    config.recognition.ctc_beam_width = 4
    config.recognition.ctc_lexicon = lexicon
    config.recognition.ctc_lexicon_weight = 1.0
    recognizer = build_recognizer(config)

    assert isinstance(recognizer, CTCRecognizer)
    assert recognizer.scorer is not None
    assert recognizer.decode.beam_width == 4
    assert len(recognizer.scorer.lexicon) == 2


def test_line_grouping_wraps_the_word_detector() -> None:
    from embedding_ocr.pipeline.lines import LineGroupingDetector

    detector = build_detector(DetectorSettings(backend="full", line_grouping=True))
    assert isinstance(detector, LineGroupingDetector)
    regions = detector.detect(np.zeros((10, 20, 3), dtype=np.uint8))
    assert len(regions) == 1


def test_line_grouping_is_off_by_default() -> None:
    from embedding_ocr.pipeline.lines import LineGroupingDetector

    detector = build_detector(DetectorSettings(backend="full"))
    assert not isinstance(detector, LineGroupingDetector)
