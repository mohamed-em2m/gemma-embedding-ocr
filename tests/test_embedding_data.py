"""Tests for font splits, short-text sampling, and rendering."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from embedding_ocr.training.fonts import (
    FontFamily,
    discover_font_families,
    split_font_families,
)
from embedding_ocr.training.render import (
    augment_image,
    render_text,
    render_variants,
    resolve_default_font,
)
from embedding_ocr.training.short_texts import (
    TextSample,
    dedupe_texts,
    edit_negative,
    hard_negative,
    is_valid_text,
    lookalike_negative,
    make_synthetic_texts,
    normalize_text,
    words_from_document,
)


def test_normalize_and_validate_text() -> None:
    assert normalize_text("  Hello\n\tWorld  ") == "Hello World"
    assert is_valid_text("Hello World")
    assert not is_valid_text("")
    assert not is_valid_text("   ")
    assert not is_valid_text("word " * 21)


def test_text_sample_rejects_invalid() -> None:
    assert TextSample.from_text("") is None
    sample = TextSample.from_text("Invoice 42")
    assert sample is not None
    assert sample.text == "Invoice 42"
    assert len(sample.text_id) == 40


def test_dedupe_texts_preserves_order_and_limit() -> None:
    samples = dedupe_texts(["Hello", "Hello", "World", "  "], limit=2)
    assert [sample.text for sample in samples] == ["Hello", "World"]


def test_words_from_document_is_bounded() -> None:
    text = words_from_document("one two three four five", rng=random.Random(0))
    assert text is not None
    assert 1 <= len(text.split()) <= 5


def test_lookalike_and_edit_negatives() -> None:
    rng = random.Random(0)
    assert lookalike_negative("O0", rng=rng) != "O0"
    for _ in range(10):
        assert edit_negative("hello", rng=rng) != "hello"


def test_hard_negative_uses_lexicon() -> None:
    negative = hard_negative("hello", ["hello", "hallo", "world"], rng=random.Random(0))
    assert negative == "hallo"


def test_synthetic_texts_are_valid_and_unique() -> None:
    samples = make_synthetic_texts(count=50, seed=0)
    assert len(samples) == 50
    assert len({sample.text_id for sample in samples}) == 50
    assert all(is_valid_text(sample.text) for sample in samples)


def test_discover_and_split_font_families(tmp_path: Path) -> None:
    root = tmp_path / "fonts"
    for index in range(6):
        family = root / "ofl" / f"family{index}"
        family.mkdir(parents=True)
        (family / f"Family{index}-Regular.ttf").write_bytes(b"font")
        (family / f"Family{index}-Bold.ttf").write_bytes(b"font")

    families = discover_font_families(root)
    assert len(families) == 6
    train, heldout = split_font_families(families, holdout_fraction=0.3, seed=0)
    assert len(train) == 4
    assert len(heldout) == 2
    assert not {family.name for family in train} & {family.name for family in heldout}


def test_split_font_families_requires_two() -> None:
    family = FontFamily(name="a", paths=(Path("a.ttf"),))
    with pytest.raises(ValueError, match="two font families"):
        split_font_families([family])


def test_render_text_falls_back_for_broken_font() -> None:
    image = render_text("Invoice 42", "/tmp/does-not-exist.ttf", canvas_size=96, seed=0)
    assert image.size == (96, 96)
    assert np.asarray(image).min() < 128


def test_render_text_shape_and_ink() -> None:
    font = resolve_default_font()
    image = render_text("Invoice 42", font, canvas_size=96, seed=0)
    assert image.size == (96, 96)
    assert image.mode == "RGB"
    assert np.array(image).min() < 200


def test_render_text_wraps_long_labels() -> None:
    font = resolve_default_font()
    long_text = "His sons Gary and Ryan thought it would be awesome to honour their father today"
    image = render_text(long_text, font, canvas_size=128, seed=0)
    gray = np.asarray(image.convert("L"))
    assert (gray < 200).mean() > 0.02
    # Wrapped text occupies several text rows instead of one unreadable line.
    assert int((gray < 200).any(axis=1).sum()) > 40


def test_augment_image_returns_same_shape() -> None:
    font = resolve_default_font()
    image = render_text("Hello", font, canvas_size=96, seed=0)
    augmented = augment_image(image, hard=True, rng=random.Random(0))
    assert augmented.size == image.size
    assert augmented.mode == "RGB"


def test_render_variants_count() -> None:
    font = resolve_default_font()
    variants = render_variants(
        "Hello",
        [font],
        count=3,
        augment=False,
        canvas_size=64,
        seed=0,
    )
    assert len(variants) == 3
    assert all(isinstance(image, Image.Image) for image in variants)


def test_build_synthetic_dataset_manifests(tmp_path: Path) -> None:
    from embedding_ocr.training.embedding_dataset import (
        EmbeddingDatasetConfig,
        build_synthetic_dataset,
        load_manifest,
    )

    font = Path(resolve_default_font())
    train = [
        FontFamily(name="train-a", paths=(font,)),
        FontFamily(name="train-b", paths=(font,)),
    ]
    heldout = [FontFamily(name="heldout", paths=(font,))]
    samples = make_synthetic_texts(count=20, seed=1)
    built = build_synthetic_dataset(
        samples,
        train,
        heldout,
        EmbeddingDatasetConfig(
            output_dir=tmp_path / "embedding",
            text_image_rows=6,
            image_image_rows=3,
            val_fraction=0.34,
            canvas_size=48,
            augment=False,
            seed=0,
        ),
    )

    assert built.counts["train"] > 0
    assert built.counts["val"] > 0
    for mode in ("text_image", "image_image"):
        train_rows = load_manifest(built.manifests[f"train/{mode}"])
        val_rows = load_manifest(built.manifests[f"val/{mode}"])
        assert train_rows
        assert val_rows
        assert all(row["split"] == "train" for row in train_rows)
        assert all(row["split"] == "val" for row in val_rows)
        assert all(Path(str(row["positive"])).is_absolute() for row in train_rows)
        assert all(Path(str(row["positive"])).exists() for row in train_rows)
        if mode == "image_image":
            assert all(
                Path(str(row["anchor_path"])).is_absolute() for row in train_rows
            )
    assert (built.root / "summary.json").exists()


def test_load_and_save_real_crops(tmp_path: Path) -> None:
    from embedding_ocr.training.real_text_crops import (
        load_real_crops,
        save_real_crops,
    )

    image = Image.new("RGB", (16, 16), "white")
    manifest = save_real_crops(
        [(image, "Invoice 42"), (image, "   ")],
        tmp_path / "real",
        source="unit",
    )
    crops = load_real_crops(manifest)
    assert len(crops) == 1
    assert crops[0].text == "Invoice 42"
    assert crops[0].source == "unit"
    assert crops[0].image_path.is_absolute()
    assert crops[0].image_path.exists()

    # The shell wrapper passes the dataset directory, not the manifest path.
    assert [crop.text for crop in load_real_crops(manifest.parent)] == ["Invoice 42"]

    with pytest.raises(FileNotFoundError):
        load_real_crops(tmp_path / "missing")


def test_builder_appends_real_rows(tmp_path: Path) -> None:
    from embedding_ocr.training.embedding_dataset import (
        EmbeddingDatasetConfig,
        build_synthetic_dataset,
        load_manifest,
    )
    from embedding_ocr.training.real_text_crops import RealCrop

    font = Path(resolve_default_font())
    train = [
        FontFamily(name="train-a", paths=(font,)),
        FontFamily(name="train-b", paths=(font,)),
    ]
    heldout = [FontFamily(name="heldout", paths=(font,))]
    image = Image.new("RGB", (24, 24), "white")
    crop_path = tmp_path / "crop.png"
    image.save(crop_path)
    real_crops = [RealCrop(image_path=crop_path, text="Real Invoice 42")]

    built = build_synthetic_dataset(
        make_synthetic_texts(count=10, seed=2),
        train,
        heldout,
        EmbeddingDatasetConfig(
            output_dir=tmp_path / "with-real",
            text_image_rows=2,
            image_image_rows=1,
            real_rows=3,
            val_fraction=0.0,
            canvas_size=48,
            augment=False,
            seed=0,
        ),
        real_crops=real_crops,
    )
    rows = load_manifest(built.manifests["train/text_image"])
    assert any(row["text"] == "Real Invoice 42" for row in rows)
