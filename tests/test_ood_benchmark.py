"""Tests for the OOD benchmark dispatch and metric helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from PIL import Image

from embedding_ocr.data.ood_dataset import OODSample
from embedding_ocr.recognition.base import RecognitionResult


def _load_script() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_ood_ocr.py"
    spec = importlib.util.spec_from_file_location("benchmark_ood_ocr", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class DummyEngine:
    def __init__(self, text: str) -> None:
        self.text = text

    def recognize(self, image: np.ndarray) -> RecognitionResult:
        del image
        return RecognitionResult(text=self.text, confidence=0.9)


def _sample(tmp_path: Path, script: str, category: str) -> OODSample:
    image_path = tmp_path / f"{script}_{category}.png"
    Image.fromarray(np.full((16, 16, 3), 255, dtype=np.uint8)).save(image_path)
    return OODSample(
        sample_id=f"{script}_{category}",
        image_path=image_path,
        text="abc",
        script=script,
        category=category,
        font="/fake/font.ttf",
        condition="clean",
    )


def test_hybrid_predictor_dispatches_glyphs_and_words(tmp_path: Path) -> None:
    module = _load_script()
    predictor = module.HybridEmbeddingPredictor(
        name="hybrid",
        mode="hybrid",
        char_engines={
            "latin": DummyEngine("char-latin"),
            "arabic": DummyEngine("char-arabic"),
        },
        hier_engines={
            "latin": DummyEngine("hier-latin"),
            "arabic": DummyEngine("hier-arabic"),
        },
    )
    image = np.zeros((8, 8, 3), dtype=np.uint8)

    assert predictor.predict(_sample(tmp_path, "latin", "glyph"), image)[0] == (
        "char-latin"
    )
    assert predictor.predict(_sample(tmp_path, "arabic", "glyph"), image)[0] == (
        "char-arabic"
    )
    assert predictor.predict(_sample(tmp_path, "latin", "word"), image)[0] == (
        "hier-latin"
    )
    assert predictor.predict(_sample(tmp_path, "arabic", "phrase"), image)[0] == (
        "hier-arabic"
    )


def test_evaluate_predictor_reports_groups(tmp_path: Path) -> None:
    module = _load_script()
    samples = [
        _sample(tmp_path, "latin", "word"),
        _sample(tmp_path, "arabic", "word"),
    ]
    report = module.evaluate_predictor(
        module.EnginePredictor("dummy", DummyEngine("abc")), samples
    )

    assert report.status == "ok"
    assert report.metrics["cer"] == pytest.approx(0.0)
    assert report.metrics["exact_match"] == pytest.approx(1.0)
    assert "script:latin" in report.groups
    assert "script:arabic" in report.groups
    assert "condition:clean" in report.groups
    assert len(report.samples) == 2
