"""Tests for synthetic YOLO letter-dataset generation."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
import yaml

from embedding_ocr.training.letter_dataset import (
    ARABIC_LIGATURES,
    CLASS_TO_ID,
    DETECTION_CLASSES,
    FontInfo,
    generate_dataset,
    make_image,
    shape_word,
    split_fonts,
)


def test_detection_classes_are_unique_and_include_arabic() -> None:
    assert len(DETECTION_CLASSES) == len(set(DETECTION_CLASSES))
    assert "a" in CLASS_TO_ID
    assert "م" in CLASS_TO_ID
    assert ARABIC_LIGATURES[0] in CLASS_TO_ID


def test_split_fonts_has_no_leakage() -> None:
    fonts = [FontInfo(Path(f"latin-{index}.ttf"), ("latin",)) for index in range(8)]
    fonts += [FontInfo(Path(f"arabic-{index}.ttf"), ("arabic",)) for index in range(8)]

    split = split_fonts(fonts, val_fraction=0.2, test_fraction=0.2, seed=7)
    train = {font.path for font in split["train"]}
    val = {font.path for font in split["val"]}
    test = {font.path for font in split["test"]}

    assert train
    assert val
    assert test
    assert not train & val
    assert not train & test
    assert not val & test
    assert train | val | test == {font.path for font in fonts}


def _font_path(candidates: list[str]) -> str:
    for candidate in candidates:
        if Path(candidate).exists():
            return candidate
    pytest.skip("Required test font is not installed")
    raise AssertionError("unreachable")


def test_shape_latin_returns_one_box_per_character() -> None:
    pytest.importorskip("uharfbuzz")
    path = _font_path(
        [
            "/usr/share/fonts/liberation-sans-fonts/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        ]
    )

    rendered = shape_word("Hello", path, font_size=48, script="latin")

    assert rendered is not None
    assert [unit.text for unit in rendered.units] == ["H", "e", "l", "l", "o"]
    for unit in rendered.units:
        x0, y0, x1, y1 = unit.bbox
        assert 0 <= x0 < x1 <= rendered.width
        assert 0 <= y0 < y1 <= rendered.height


def test_shape_arabic_merges_diacritics_into_base_units() -> None:
    pytest.importorskip("uharfbuzz")
    path = _font_path(
        [
            "/usr/share/fonts/google-noto-vf/NotoNaskhArabic[wght].ttf",
            "/usr/share/fonts/truetype/noto/NotoNaskhArabic-Regular.ttf",
        ]
    )

    rendered = shape_word("بَ", path, font_size=48, script="arabic")

    assert rendered is not None
    assert [unit.text for unit in rendered.units] == ["ب"]


def test_line_layout_uses_available_scripts_and_spreads_lines() -> None:
    pytest.importorskip("uharfbuzz")
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
    fonts = [
        FontInfo(Path(latin), ("latin",)),
        FontInfo(Path(arabic), ("arabic",)),
    ]

    _image, labels = make_image(
        640,
        fonts,
        random.Random(0),
        np.random.default_rng(0),
        layout="line",
        min_font_size=24,
        max_font_size=72,
        max_words=4,
        max_lines=2,
        empty_fraction=0.0,
    )

    text = "".join(label.text for label in labels)
    assert any(char.isascii() and char.isalnum() for char in text)
    assert any("\u0600" <= char <= "\u06ff" for char in text)
    centers = [(label.bbox[1] + label.bbox[3]) / 2.0 for label in labels]
    assert max(centers) - min(centers) > 128


def test_generate_dataset_writes_valid_yolo_files(tmp_path: Path) -> None:
    pytest.importorskip("uharfbuzz")
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
    fonts = [
        FontInfo(Path(latin), ("latin",)),
        FontInfo(Path(arabic), ("arabic",)),
    ]

    manifest = generate_dataset(
        tmp_path,
        fonts,
        n=4,
        size=320,
        val_fraction=0.0,
        test_fraction=0.0,
        seed=11,
        layout="mixed",
        min_font_size=20,
        max_font_size=44,
        max_words=4,
        max_lines=2,
        preview_count=2,
    )

    assert manifest["image_counts"]["train"] == 4
    assert manifest["label_counts"]["train"] > 0
    assert (tmp_path / "data.yaml").exists()
    assert (tmp_path / "classes.json").exists()
    assert (tmp_path / "manifest.json").exists()
    assert len(list((tmp_path / "previews").glob("*.jpg"))) == 2

    data = yaml.safe_load((tmp_path / "data.yaml").read_text(encoding="utf-8"))
    assert data["nc"] == len(DETECTION_CLASSES)
    assert len(data["names"]) == len(DETECTION_CLASSES)

    label_files = sorted((tmp_path / "labels" / "train").glob("*.txt"))
    assert len(label_files) == 4
    for label_file in label_files:
        for line in label_file.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            assert len(parts) == 5
            class_id = int(parts[0])
            assert 0 <= class_id < len(DETECTION_CLASSES)
            values = [float(value) for value in parts[1:]]
            assert all(0.0 <= value <= 1.0 for value in values)
            assert values[1] > 0.0
            assert values[2] > 0.0
            assert values[3] > 0.0
