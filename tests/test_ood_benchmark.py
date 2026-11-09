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


class WordAwareEngine(DummyEngine):
    def recognize_word_crop(self, image: np.ndarray, depth: int = 0):
        del image, depth
        return RecognitionResult(text="word-path", confidence=0.8)


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


def test_word_samples_use_word_crop_path(tmp_path: Path) -> None:
    module = _load_script()
    engine = WordAwareEngine("line-path")
    predictor = module.HybridEmbeddingPredictor(
        name="hier",
        mode="hier",
        char_engines={"latin": engine, "arabic": engine},
        hier_engines={"latin": engine, "arabic": engine},
    )
    image = np.zeros((8, 8, 3), dtype=np.uint8)

    assert (
        predictor.predict(_sample(tmp_path, "latin", "word"), image)[0] == "word-path"
    )
    assert (
        predictor.predict(_sample(tmp_path, "arabic", "diacritics"), image)[0]
        == "word-path"
    )
    assert (
        predictor.predict(_sample(tmp_path, "arabic", "ligature"), image)[0]
        == "word-path"
    )
    assert predictor.predict(_sample(tmp_path, "latin", "phrase"), image)[0] == (
        "line-path"
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


def test_ensure_dataset_fails_without_autobuild(tmp_path: Path) -> None:
    module = _load_script()
    with pytest.raises(FileNotFoundError):
        module._ensure_dataset(tmp_path / "manifest.jsonl", auto_build=False)


def test_ensure_dataset_builds_when_missing(monkeypatch, tmp_path: Path) -> None:
    module = _load_script()

    def fake_build(out: Path, overwrite: bool = False) -> Path:
        del overwrite
        out.mkdir(parents=True, exist_ok=True)
        manifest = out / "manifest.jsonl"
        manifest.write_text("")
        (out / "dataset.json").write_text("{}")
        return manifest

    monkeypatch.setattr("embedding_ocr.data.ood_dataset.build_ood_dataset", fake_build)
    manifest = tmp_path / "manifest.jsonl"
    module._ensure_dataset(manifest, auto_build=True)
    assert manifest.exists()
    assert (tmp_path / "dataset.json").exists()


def test_limit_per_script_is_stratified(tmp_path: Path) -> None:
    import json

    module = _load_script()
    manifest = tmp_path / "manifest.jsonl"
    rows = []
    for script in ("latin", "arabic"):
        for category in ("word", "glyph"):
            for condition in ("clean", "hard"):
                rows.append(
                    {
                        "id": f"{script}_{category}_{condition}",
                        "image": f"{script}_{category}_{condition}.png",
                        "text": "x",
                        "script": script,
                        "category": category,
                        "font": "/fake/font.ttf",
                        "condition": condition,
                    }
                )
    manifest.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )

    selected = module._load_samples(manifest, limit_per_script=2)
    assert len(selected) == 4
    assert {sample.category for sample in selected} == {"word", "glyph"}
