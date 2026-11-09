"""Tests for the Gradio UI helpers (Gradio itself is optional)."""

from __future__ import annotations

import numpy as np
import pytest

from embedding_ocr.config import AppConfig, load_config
from embedding_ocr.pipeline.types import OCRResult, RecognizedRegion, TextRegion
from embedding_ocr.webui import (
    UiOverrides,
    annotate_result,
    apply_ui_overrides,
    format_details,
    list_config_profiles,
    resolve_config_path,
)


def test_config_profiles_are_listed_and_resolved() -> None:
    profiles = list_config_profiles("configs")
    assert "default.yaml" in profiles
    assert resolve_config_path("default.yaml").name == "default.yaml"
    assert resolve_config_path(None).name in profiles


def test_config_resolution_rejects_unknown_and_traversal() -> None:
    with pytest.raises(ValueError, match="invalid profile"):
        resolve_config_path("../default.yaml")
    with pytest.raises(ValueError, match="unknown profile"):
        resolve_config_path("does-not-exist.yaml")


def test_quick_mode_caps_hierarchical_vocabulary() -> None:
    config = load_config("configs/default.yaml")
    overrides = UiOverrides(quick=True, script="arabic")
    updated = apply_ui_overrides(config, overrides)

    assert updated.recognition.script == "arabic"
    assert updated.recognition.rtl is None
    assert updated.recognition.vocabulary.max_words is not None
    assert updated.recognition.vocabulary.max_words <= 4000
    assert updated.recognition.vocabulary.max_subwords is not None
    assert updated.recognition.vocabulary.max_subwords <= 3000


def test_quick_mode_does_not_touch_non_hierarchical_config() -> None:
    config = AppConfig()
    config.recognition.backend = "tesseract"
    config.recognition.vocabulary.max_words = 25000
    apply_ui_overrides(config, UiOverrides(quick=True))
    assert config.recognition.vocabulary.max_words == 25000


def test_annotate_result_draws_boxes() -> None:
    image = np.full((60, 100, 3), 255, dtype=np.uint8)
    result = OCRResult(
        text="ab",
        confidence=0.9,
        regions=[
            RecognizedRegion(
                region=TextRegion(bbox=(10, 10, 20, 30)),
                text="a",
                confidence=0.9,
            )
        ],
    )
    annotated = annotate_result(image, result)
    assert annotated.shape == image.shape
    assert not np.array_equal(annotated, image)


def test_format_details_is_serializable() -> None:
    result = OCRResult(
        text="x",
        confidence=0.5,
        metadata={"latency_ms": 1.0},
        regions=[
            RecognizedRegion(
                region=TextRegion(bbox=(0, 0, 4, 4)),
                text="x",
                confidence=0.5,
            )
        ],
    )
    details = format_details("default.yaml", UiOverrides(), result)
    assert details["profile"] == "default.yaml"
    assert details["regions"][0]["text"] == "x"
    assert details["overrides"]["device"] == "auto"


def test_run_ocr_uses_pipeline(monkeypatch) -> None:
    import embedding_ocr.webui as webui

    class DummyPipeline:
        def recognize(self, image: np.ndarray) -> OCRResult:
            return OCRResult(
                text="hello",
                confidence=0.75,
                regions=[
                    RecognizedRegion(
                        region=TextRegion(bbox=(1, 1, 10, 10)),
                        text="hello",
                        confidence=0.75,
                    )
                ],
                metadata={
                    "detector": "dummy",
                    "recognizer": "dummy",
                    "latency_ms": 2.5,
                },
            )

    monkeypatch.setattr(webui, "get_pipeline", lambda *args, **kwargs: DummyPipeline())
    image = np.full((32, 32, 3), 255, dtype=np.uint8)
    text, confidence, details, annotated, status = webui.run_ocr(
        image,
        "default.yaml",
        "cpu",
        "config default",
        "config default",
        "config default",
        True,
    )

    assert text == "hello"
    assert confidence == pytest.approx(0.75)
    assert details["text"] == "hello"
    assert annotated is not None
    assert "Done" in status
