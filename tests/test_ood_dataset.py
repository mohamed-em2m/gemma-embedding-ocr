"""Tests for the held-out Latin/Arabic OOD OCR dataset."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from embedding_ocr.data.ood_dataset import (
    OOD_INDEX_FONTS,
    OOD_TEST_FONTS,
    build_ood_dataset,
    load_ood_dataset,
)

COUNTS = {
    "latin": {"word": 1, "phrase": 1, "glyph": 2},
    "arabic": {
        "word": 1,
        "phrase": 1,
        "diacritics": 1,
        "ligature": 1,
        "glyph": 2,
    },
}


def _available(fonts: tuple[str, ...]) -> list[str]:
    return [font for font in fonts if Path(font).exists()]


def _available_split() -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    index = {script: _available(fonts) for script, fonts in OOD_INDEX_FONTS.items()}
    test = {script: _available(fonts) for script, fonts in OOD_TEST_FONTS.items()}
    if not index.get("latin") or not test.get("latin"):
        pytest.skip("Latin index/test fonts are not installed")
    if not index.get("arabic") or not test.get("arabic"):
        pytest.skip("Arabic index/test fonts are not installed")
    # Keep the test small but preserve the disjoint-font contract.
    return (
        {script: fonts[:1] for script, fonts in index.items()},
        {script: fonts[:1] for script, fonts in test.items()},
    )


def _build(tmp_path: Path, name: str) -> Path:
    index_fonts, test_fonts = _available_split()
    return build_ood_dataset(
        tmp_path / name,
        seed=1234,
        conditions=("clean", "hard"),
        counts=COUNTS,
        index_fonts=index_fonts,
        test_fonts=test_fonts,
        font_size=30,
    )


def test_build_writes_expected_manifest_and_images(tmp_path: Path) -> None:
    manifest = _build(tmp_path, "dataset")
    samples = load_ood_dataset(manifest)

    expected = (4 + 6) * 2
    assert len(samples) == expected
    assert all(sample.image_path.exists() for sample in samples)
    assert {"latin", "arabic"} == {sample.script for sample in samples}
    assert {"clean", "hard"} == {sample.condition for sample in samples}
    assert any(sample.category == "diacritics" for sample in samples)

    for sample in samples:
        array = np.asarray(Image.open(sample.image_path).convert("RGB"))
        assert array.ndim == 3
        assert array.shape[2] == 3
        assert array.max() > array.min()

    meta = json.loads((manifest.parent / "dataset.json").read_text())
    for script in ("latin", "arabic"):
        assert set(meta["index_fonts"][script]).isdisjoint(meta["test_fonts"][script])


def test_build_is_deterministic(tmp_path: Path) -> None:
    first = load_ood_dataset(_build(tmp_path, "first"))
    second = load_ood_dataset(_build(tmp_path, "second"))

    assert [(sample.sample_id, sample.text) for sample in first] == [
        (sample.sample_id, sample.text) for sample in second
    ]
    for left, right in zip(first, second, strict=True):
        left_hash = hashlib.sha256(left.image_path.read_bytes()).hexdigest()
        right_hash = hashlib.sha256(right.image_path.read_bytes()).hexdigest()
        assert left_hash == right_hash


def test_build_refuses_to_overwrite_manifest(tmp_path: Path) -> None:
    manifest = _build(tmp_path, "dataset")
    assert manifest.exists()
    with pytest.raises(FileExistsError):
        _build(tmp_path, "dataset")
