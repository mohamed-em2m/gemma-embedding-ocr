"""Tests for the Gradio UI helpers (Gradio itself is optional)."""

from __future__ import annotations

import sys
from pathlib import Path

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
    parse_args,
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


def test_run_ocr_clears_pipeline_cache_after_cuda_oom(monkeypatch) -> None:
    import embedding_ocr.webui as webui

    class OutOfMemoryError(Exception):
        pass

    def raise_oom(*args, **kwargs):
        raise OutOfMemoryError("CUDA out of memory: tried to allocate 1 GiB")

    cleared: list[bool] = []
    monkeypatch.setattr(webui, "get_pipeline", raise_oom)
    monkeypatch.setattr(webui, "clear_pipeline_cache", lambda: cleared.append(True))
    image = np.full((32, 32, 3), 255, dtype=np.uint8)
    _text, _confidence, details, annotated, status = webui.run_ocr(
        image,
        "default.yaml",
        "cuda",
        "config default",
        "config default",
        "config default",
        True,
    )

    assert cleared == [True]
    assert "OutOfMemoryError" in status
    assert "error" in details
    assert annotated is not None


def test_public_share_link_is_enabled_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["embedding-ocr-ui"])
    assert parse_args().share is True

    monkeypatch.setattr(sys, "argv", ["embedding-ocr-ui", "--no-share"])
    assert parse_args().share is False


def test_bundled_example_images_are_available() -> None:
    from embedding_ocr.webui import list_example_images

    examples = list_example_images()
    assert len(examples) >= 3
    assert all(Path(path).is_file() for path in examples)


def test_example_rows_select_arabic_profile_and_script() -> None:
    from embedding_ocr.webui import example_rows

    rows = example_rows(["default.yaml", "embedding_align.yaml"])
    assert len(rows) >= 3
    arabic_rows = [row for row in rows if str(row[0]).endswith("arabic_marhaba.png")]
    assert arabic_rows
    assert arabic_rows[0][1] == "embedding_align.yaml"
    assert arabic_rows[0][3] == "hierarchical"
    assert arabic_rows[0][4] == "full"
    assert arabic_rows[0][5] == "arabic"
    assert all(len(row) == 7 for row in rows)


def test_pipeline_cache_keeps_only_one_model(monkeypatch) -> None:
    import embedding_ocr.webui as webui

    class DummyPipeline:
        released = 0

        def __del__(self) -> None:
            type(self).released += 1

    webui.clear_pipeline_cache()
    monkeypatch.setattr(webui, "build_pipeline", lambda config: DummyPipeline())

    webui.get_pipeline(
        "default.yaml",
        webui.UiOverrides(device="cpu", quick=True),
    )
    webui.get_pipeline(
        "default.yaml",
        webui.UiOverrides(device="auto", quick=True),
    )

    assert len(webui._PIPELINE_CACHE) == 1
    assert DummyPipeline.released >= 1
    webui.clear_pipeline_cache()
