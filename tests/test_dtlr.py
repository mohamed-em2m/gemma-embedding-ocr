"""Tests for the optional DTLR adapter (no external checkout required)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from embedding_ocr.recognition.dtlr import DTLRRecognizer, _decode_detections


def _outputs(logits: list[list[float]], xs: list[float]) -> dict[str, torch.Tensor]:
    boxes = [[[x, 0.5, 0.1, 0.2] for x in xs]]
    return {
        "pred_logits": torch.tensor([logits], dtype=torch.float32),
        "pred_boxes": torch.tensor(boxes, dtype=torch.float32),
    }


def test_decode_orders_characters_by_box_centre() -> None:
    outputs = _outputs(
        logits=[
            [8.0, -8.0, -8.0],
            [-8.0, 8.0, -8.0],
            [-8.0, -8.0, 8.0],
            [-8.0, -8.0, -8.0],
        ],
        xs=[0.7, 0.1, 0.4, 0.9],
    )

    text, confidence = _decode_detections(outputs, ("A", "B", "C"))

    assert text == "BCA"
    assert 0.9 < confidence <= 1.0


def test_decode_skips_background_queries() -> None:
    outputs = _outputs(
        logits=[
            [8.0, -8.0],
            [-8.0, -8.0],
            [-8.0, 8.0],
        ],
        xs=[0.3, 0.1, 0.6],
    )

    text, confidence = _decode_detections(outputs, ("A", "B"))

    assert text == "AB"
    assert confidence > 0.9


def test_decode_empty_detections() -> None:
    outputs = {
        "pred_logits": torch.empty((1, 0, 3)),
        "pred_boxes": torch.empty((1, 0, 4)),
    }

    assert _decode_detections(outputs, ("A", "B", "C")) == ("", 0.0)


def test_dtlr_requires_checkout_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DTLR_ROOT", raising=False)

    with pytest.raises(ValueError, match="DTLR_ROOT"):
        DTLRRecognizer()


def test_dtlr_rejects_missing_checkout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("DTLR_ROOT", raising=False)

    with pytest.raises(FileNotFoundError, match="not a directory"):
        DTLRRecognizer(root=tmp_path / "missing")


def test_dtlr_requires_weights(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("DTLR_ROOT", raising=False)
    monkeypatch.delenv("DTLR_WEIGHTS", raising=False)
    root = tmp_path / "DTLR"
    dino = root / "models" / "dino"
    dino.mkdir(parents=True)
    (dino / "dino.py").write_text("")

    with pytest.raises(ValueError, match="DTLR_WEIGHTS"):
        DTLRRecognizer(root=root)
