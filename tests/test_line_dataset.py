"""Tests for line-level manifests, the character vocabulary, and normalization."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from embedding_ocr.recognition.ctc.charset import CharSet, UnknownCharacterError
from embedding_ocr.recognition.ctc.normalize import (
    detect_script,
    normalize_for_cer,
    normalize_text,
)
from embedding_ocr.training.line_dataset import (
    FilterConfig,
    LineRecord,
    LineSource,
    ctc_timesteps,
    filter_reason,
    import_local_lines,
    inspect_local_dataset,
    prepare_line_datasets,
    read_label_table,
    read_manifest,
    write_manifest,
)


def _row(text: str, shade: int) -> dict[str, Any]:
    """A fake Hugging Face row: a distinct 160x32 image plus a label."""
    return {"image": Image.new("RGB", (160, 32), (shade, 0, 0)), "text": text}


def _loader(
    rows: Mapping[str, Sequence[Mapping[str, Any]]],
) -> Any:
    def load(hf_id: str, splits: Sequence[str]) -> dict[str, Any]:
        del hf_id
        return {split: rows[split] for split in splits if split in rows}

    return load


# --- normalization -----------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("ﻣﺤﻤﺪ", "محمد"),
        ("مُحَمَّد", "محمد"),
        ("مـــحــمــد", "محمد"),
        ("أحمد إبراهيم آدم", "احمد ابراهيم ادم"),
        ("  hello \n  world  ", "hello world"),
        ("Hello", "Hello"),
    ],
)
def test_normalize_text(raw: str, expected: str) -> None:
    assert normalize_text(raw) == expected


def test_normalize_can_keep_diacritics() -> None:
    assert normalize_text("مُحَمَّد", strip_diacritics=False) == "مُحَمَّد"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("hello", "latin"),
        ("مرحبا", "arabic"),
        ("hello مرحبا", "mixed"),
        ("123 !?", "unknown"),
    ],
)
def test_detect_script(text: str, expected: str) -> None:
    assert detect_script(text) == expected


def test_normalize_for_cer_keeps_latin_case() -> None:
    assert normalize_for_cer("Hello") != normalize_for_cer("hello")
    assert normalize_for_cer("Hello", case_sensitive=False) == normalize_for_cer(
        "hello", case_sensitive=False
    )


def test_normalize_for_cer_folds_arabic_consistently() -> None:
    assert normalize_for_cer("أَحْمَد") == normalize_for_cer("احمد")


# --- charset -----------------------------------------------------------------


def test_charset_round_trip() -> None:
    charset = CharSet.from_texts(["abc", "abd"])
    assert charset.chars == ("a", "b", "c", "d")
    assert charset.size == 5
    ids = charset.encode("cab")
    assert ids == [3, 1, 2]
    assert charset.decode(ids) == "cab"
    assert charset.decode([0, 3, 0, 1, 2]) == "cab"


def test_charset_rejects_unknown_characters() -> None:
    charset = CharSet.from_texts(["ab"])
    with pytest.raises(UnknownCharacterError, match="not in the charset"):
        charset.encode("z")
    assert charset.encode("azb", strict=False) == [1, 2]
    assert charset.unknown("azb") == {"z"}


def test_charset_save_load(tmp_path: Path) -> None:
    charset = CharSet.from_texts(["abc"])
    path = tmp_path / "charset.json"
    charset.save(path)
    assert CharSet.load(path) == charset
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["chars"] == ["a", "b", "c"]


def test_charset_rejects_empty_vocabulary() -> None:
    with pytest.raises(ValueError, match="at least one character"):
        CharSet(())
    with pytest.raises(ValueError, match="duplicate"):
        CharSet(("a", "a"))


# --- manifest ----------------------------------------------------------------


def test_manifest_round_trip(tmp_path: Path) -> None:
    image = tmp_path / "crop.png"
    Image.new("L", (64, 16), 255).save(image)
    record = LineRecord(
        image=image,
        text="Hello world",
        text_norm="Hello world",
        script="latin",
        split="train",
        source="fake",
        orig_id="train/0",
        width=64,
        height=16,
    )
    manifest = tmp_path / "manifest.jsonl"
    write_manifest([record], manifest)

    loaded = read_manifest(manifest)
    assert len(loaded) == 1
    assert loaded[0].image == image
    assert loaded[0].text_norm == "Hello world"
    payload = json.loads(manifest.read_text(encoding="utf-8").strip())
    assert payload["image"] == "crop.png"


def test_filter_reason() -> None:
    config = FilterConfig()
    assert filter_reason("", 100, 32, config) == "empty_text"
    assert filter_reason("x" * 97, 100, 32, config) == "too_long"
    assert filter_reason("ab", 100, 2, config) == "too_small"
    assert filter_reason("ab", 4000, 32, config) == "extreme_aspect"
    assert filter_reason("ab", 100, 32, config) is None


def test_ctc_timesteps_matches_encoder_stride() -> None:
    config = FilterConfig(image_height=32, width_stride=2)
    assert ctc_timesteps(160, 32, config) == 80
    assert ctc_timesteps(160, 0, config) == 0


def test_filter_reason_rejects_labels_longer_than_time_axis() -> None:
    config = FilterConfig(max_chars=80)
    # 40 characters need 79 CTC steps; 100 px at height 32 yields only 50.
    assert filter_reason("x" * 40, 100, 32, config) == "ctc_infeasible"
    assert filter_reason("x" * 10, 100, 32, config) is None


# --- preparation -------------------------------------------------------------


def test_prepare_line_datasets_writes_manifest(tmp_path: Path) -> None:
    rows = {
        "train": [_row("hello", 10), _row("world", 20)],
        "val": [_row("held", 30)],
        "test": [_row("red", 40)],
    }
    output = tmp_path / "lines"
    stats = prepare_line_datasets(
        [LineSource("fake", "fake/id", script="latin")],
        output,
        loader=_loader(rows),
    )

    assert stats["totals"]["by_split"] == {"train": 2, "val": 1, "test": 1}
    assert stats["totals"]["by_script"] == {"latin": 4}
    records = read_manifest(output / "manifest.jsonl")
    assert len(records) == 4
    assert all(record.image.is_file() for record in records)
    assert CharSet.load(output / "charset.json").chars == (
        "d",
        "e",
        "h",
        "l",
        "o",
        "r",
        "w",
    )


def test_prepare_dry_run_writes_nothing(tmp_path: Path) -> None:
    rows = {"train": [_row("hello", 10)]}
    output = tmp_path / "lines"
    stats = prepare_line_datasets(
        [LineSource("fake", "fake/id")],
        output,
        loader=_loader(rows),
        dry_run=True,
    )
    assert stats["dry_run"] is True
    assert not (output / "manifest.jsonl").exists()


def test_prepare_dedupes_identical_images(tmp_path: Path) -> None:
    rows = {
        "train": [_row("hello", 10), _row("world", 10)],
        "test": [_row("hello", 10)],
    }
    output = tmp_path / "lines"
    stats = prepare_line_datasets(
        [LineSource("fake", "fake/id")],
        output,
        loader=_loader(rows),
    )
    assert stats["totals"]["by_split"] == {"train": 1, "val": 0, "test": 0}
    assert stats["totals"]["dedupe"] == {"test": 1, "train": 1}


def test_prepare_drops_out_of_vocabulary_eval_rows(tmp_path: Path) -> None:
    rows = {
        "train": [_row("abc", 10)],
        "val": [_row("abz", 20)],
    }
    output = tmp_path / "lines"
    stats = prepare_line_datasets(
        [LineSource("fake", "fake/id")],
        output,
        loader=_loader(rows),
    )
    assert stats["totals"]["coverage_dropped"] == {"val": 1}
    assert stats["totals"]["by_split"] == {"train": 1, "val": 0, "test": 0}


def test_prepare_requires_train_rows(tmp_path: Path) -> None:
    rows = {"val": [_row("abc", 10)]}
    with pytest.raises(ValueError, match="no usable training rows"):
        prepare_line_datasets(
            [LineSource("fake", "fake/id")],
            tmp_path / "lines",
            loader=_loader(rows),
        )


def test_prepare_refuses_non_empty_output(tmp_path: Path) -> None:
    output = tmp_path / "lines"
    output.mkdir()
    (output / "stale.txt").write_text("x", encoding="utf-8")
    with pytest.raises(FileExistsError, match="not empty"):
        prepare_line_datasets(
            [LineSource("fake", "fake/id")],
            output,
            loader=_loader({"train": [_row("abc", 10)]}),
        )


# --- local imports -----------------------------------------------------------


def test_read_label_table_csv(tmp_path: Path) -> None:
    labels = tmp_path / "labels.csv"
    labels.write_text("image,text\n1.png,hello\n2.png,world\n", encoding="utf-8")
    table = read_label_table(labels, image_column="image", text_column="text")
    assert table == {"1.png": "hello", "2.png": "world"}


def test_import_local_lines_with_label_table(tmp_path: Path) -> None:
    images = tmp_path / "images"
    images.mkdir()
    for name in ("1.png", "2.png"):
        Image.new("L", (160, 32), 255).save(images / name)
    labels = tmp_path / "labels.csv"
    labels.write_text("image,text\n1.png,hello\n2.png,world\n", encoding="utf-8")

    pairs = import_local_lines(tmp_path, labels_file=labels, image_dir=images)
    assert [text for _, text in pairs] == ["hello", "world"]


def test_import_local_lines_uses_sidecar_transcripts(tmp_path: Path) -> None:
    Image.new("L", (160, 32), 255).save(tmp_path / "line.png")
    (tmp_path / "line.txt").write_text("مرحبا", encoding="utf-8")
    pairs = import_local_lines(tmp_path)
    assert pairs == [(tmp_path / "line.png", "مرحبا")]


def test_inspect_local_dataset(tmp_path: Path) -> None:
    (tmp_path / "images").mkdir()
    Image.new("L", (64, 16), 255).save(tmp_path / "images" / "a.png")
    (tmp_path / "labels.csv").write_text("image,text\na.png,hi\n", encoding="utf-8")

    summary = inspect_local_dataset(tmp_path)
    assert summary["total_images"] == 1
    assert summary["image_counts"] == {"images": 1}
    assert summary["tables"][0]["sample"] == [["image", "text"], ["a.png", "hi"]]
