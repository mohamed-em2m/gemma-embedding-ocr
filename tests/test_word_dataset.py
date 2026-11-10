"""Tests for synthetic word-level YOLO dataset generation."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytest
import yaml

from embedding_ocr.training.letter_dataset import (
    FontInfo,
    generate_dataset,
    make_image,
)


def _font_path(candidates: list[str]) -> str:
    for candidate in candidates:
        if Path(candidate).exists():
            return candidate
    pytest.skip("Required test font is not installed")
    raise AssertionError("unreachable")


def _fonts() -> list[FontInfo]:
    latin = _font_path(
        [
            "/usr/share/fonts/liberation-sans-fonts/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        ]
    )
    arabic = _font_path(
        [
            "/usr/share/fonts/google-noto-vf/NotoNaskhArabic[wght].ttf",
            "/usr/share/fonts/truetype/noto/NotoNaskhArabic-Regular.ttf",
        ]
    )
    return [
        FontInfo(Path(latin), ("latin",)),
        FontInfo(Path(arabic), ("arabic",)),
    ]


def test_make_image_word_labels_are_one_box_per_word() -> None:
    pytest.importorskip("uharfbuzz")
    image, labels = make_image(
        640,
        _fonts(),
        random.Random(3),
        np.random.default_rng(3),
        layout="mixed",
        min_font_size=24,
        max_font_size=64,
        max_words=4,
        max_lines=3,
        empty_fraction=0.0,
        label_level="word",
    )

    assert labels
    assert all(label.class_id == 0 for label in labels)
    assert len(labels) <= 4 * 3
    for label in labels:
        x0, y0, x1, y1 = label.bbox
        assert 0 <= x0 < x1 <= image.width
        assert 0 <= y0 < y1 <= image.height
        assert x1 - x0 > 8
        assert y1 - y0 > 8


def test_generate_word_dataset_writes_single_class(tmp_path: Path) -> None:
    pytest.importorskip("uharfbuzz")
    manifest = generate_dataset(
        tmp_path,
        _fonts(),
        n=4,
        size=320,
        val_fraction=0.0,
        test_fraction=0.0,
        seed=13,
        layout="mixed",
        min_font_size=20,
        max_font_size=44,
        max_words=4,
        max_lines=2,
        preview_count=2,
        label_level="word",
    )

    assert manifest["label_level"] == "word"
    assert manifest["classes"] == ["word"]
    assert manifest["image_counts"]["train"] == 4
    assert manifest["label_counts"]["train"] > 0

    data = yaml.safe_load((tmp_path / "data.yaml").read_text(encoding="utf-8"))
    assert data["nc"] == 1
    assert data["names"] == {0: "word"}

    classes = json.loads((tmp_path / "classes.json").read_text(encoding="utf-8"))
    assert classes["classes"] == ["word"]
    assert classes["class_to_id"] == {"word": 0}

    for label_file in sorted((tmp_path / "labels" / "train").glob("*.txt")):
        for line in label_file.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            assert len(parts) == 5
            assert parts[0] == "0"
            assert all(0.0 <= float(value) <= 1.0 for value in parts[1:])


def test_unknown_label_level_raises() -> None:
    with pytest.raises(ValueError, match="label_level"):
        make_image(
            320,
            [],
            random.Random(0),
            np.random.default_rng(0),
            label_level="line",
        )
