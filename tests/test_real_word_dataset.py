"""Tests for merging real Roboflow-style word datasets."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest
import yaml
from PIL import Image

from embedding_ocr.training.real_dataset import (
    SourceSpec,
    detect_dataset_kind,
    load_source_specs,
    merge_datasets,
    parse_yolo_label_line,
    read_class_names,
)


def _image(path: Path, color: tuple[int, int, int] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if color is None:
        digest = sum(path.name.encode("utf-8"))
        color = (digest % 256, (digest * 7) % 256, (digest * 13) % 256)
    Image.new("RGB", (32, 32), color).save(path)


def _detection_source(
    root: Path,
    *,
    names: list[str],
    train: int = 2,
    valid: int = 1,
    class_id: int = 0,
    prefix: str = "img",
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "data.yaml").write_text(
        yaml.safe_dump({"nc": len(names), "names": names}), encoding="utf-8"
    )
    for split, count in (("train", train), ("valid", valid)):
        for index in range(count):
            stem = f"{prefix}_{split}_{index}"
            _image(root / split / "images" / f"{stem}.jpg")
            (root / split / "labels").mkdir(parents=True, exist_ok=True)
            (root / split / "labels" / f"{stem}.txt").write_text(
                f"{class_id} 0.5 0.5 0.2 0.2\n", encoding="utf-8"
            )
    return root


def _read_labels(root: Path, split: str) -> list[str]:
    lines: list[str] = []
    for path in sorted((root / "labels" / split).glob("*.txt")):
        lines.extend(line for line in path.read_text().splitlines() if line)
    return lines


def test_merges_texto_and_word_into_one_class(tmp_path: Path) -> None:
    word_source = _detection_source(tmp_path / "word", names=["word"], prefix="word")
    texto_source = _detection_source(
        tmp_path / "texto", names=["texto"], prefix="texto"
    )

    manifest = merge_datasets(
        [SourceSpec(path=word_source), SourceSpec(path=texto_source)],
        tmp_path / "merged",
        copy_mode="copy",
    )

    data = yaml.safe_load((tmp_path / "merged" / "data.yaml").read_text())
    assert data["nc"] == 1
    assert data["names"] == ["word"]
    assert manifest["totals"]["image_counts"] == {"train": 4, "val": 2, "test": 0}
    assert all(
        line.startswith("0 ") for line in _read_labels(tmp_path / "merged", "train")
    )
    assert manifest["sources"][1]["class_names"] == ["texto"]


def test_classification_folders_become_full_image_boxes(tmp_path: Path) -> None:
    source = tmp_path / "classification"
    _image(source / "train" / "texto" / "a.jpg")
    _image(source / "train" / "texto" / "b.jpg")
    _image(source / "valid" / "texto" / "c.jpg")

    assert detect_dataset_kind(source) == "classification"
    manifest = merge_datasets(
        [SourceSpec(path=source)], tmp_path / "merged", copy_mode="copy"
    )

    assert manifest["sources"][0]["kind"] == "classification"
    lines = _read_labels(tmp_path / "merged", "train")
    assert lines == [
        "0 0.500000 0.500000 1.000000 1.000000",
        "0 0.500000 0.500000 1.000000 1.000000",
    ]


def test_missing_validation_split_is_carved_from_train(tmp_path: Path) -> None:
    source = _detection_source(
        tmp_path / "train_only", names=["word"], train=5, valid=0
    )
    manifest = merge_datasets(
        [SourceSpec(path=source)], tmp_path / "merged", copy_mode="copy"
    )

    assert manifest["totals"]["image_counts"] == {"train": 4, "val": 1, "test": 0}


def test_duplicate_images_across_sources_are_removed(tmp_path: Path) -> None:
    first = _detection_source(tmp_path / "first", names=["word"], train=2, valid=1)
    second = tmp_path / "second"
    for split in ("train", "valid"):
        (second / split / "images").mkdir(parents=True)
        (second / split / "labels").mkdir(parents=True)
    # Copy exact bytes from the first source into the second source.
    for split, count in (("train", 2), ("valid", 1)):
        for index in range(count):
            source_image = first / split / "images" / f"img_{split}_{index}.jpg"
            (second / split / "images" / f"dup_{split}_{index}.jpg").write_bytes(
                source_image.read_bytes()
            )
            (second / split / "labels" / f"dup_{split}_{index}.txt").write_text(
                "0 0.5 0.5 0.2 0.2\n"
            )

    manifest = merge_datasets(
        [SourceSpec(path=first), SourceSpec(path=second)],
        tmp_path / "merged",
        copy_mode="copy",
    )

    assert manifest["totals"]["duplicates_removed"] == 3
    assert manifest["totals"]["image_counts"]["train"] == 2
    assert manifest["totals"]["image_counts"]["val"] == 1


def test_filename_collisions_are_prefixed_per_source(tmp_path: Path) -> None:
    first = _detection_source(tmp_path / "first", names=["word"], prefix="same")
    second = _detection_source(tmp_path / "second", names=["word"], prefix="same")

    merge_datasets(
        [SourceSpec(path=first), SourceSpec(path=second)],
        tmp_path / "merged",
        copy_mode="copy",
        dedupe=False,
    )

    names = sorted(
        path.name for path in (tmp_path / "merged" / "images" / "train").iterdir()
    )
    assert len(names) == len(set(names))
    assert all(name.startswith(("first__", "second__")) for name in names)


def test_class_filter_keeps_only_requested_classes(tmp_path: Path) -> None:
    source = tmp_path / "mixed_classes"
    source.mkdir(parents=True, exist_ok=True)
    (source / "data.yaml").write_text(
        yaml.safe_dump({"nc": 2, "names": ["word", "line"]}), encoding="utf-8"
    )
    for split, count in (("train", 2), ("valid", 1)):
        for index in range(count):
            stem = f"{split}_{index}"
            _image(source / split / "images" / f"{stem}.jpg")
            (source / split / "labels").mkdir(parents=True, exist_ok=True)
            (source / split / "labels" / f"{stem}.txt").write_text(
                "0 0.5 0.5 0.2 0.2\n1 0.5 0.5 0.2 0.2\n", encoding="utf-8"
            )

    manifest = merge_datasets(
        [SourceSpec(path=source, include_classes=("word",))],
        tmp_path / "merged",
        copy_mode="copy",
    )

    assert manifest["sources"][0]["boxes_kept"] == 3
    assert manifest["sources"][0]["boxes_dropped_class_filter"] == 3


def test_read_class_names_accepts_string_keys(tmp_path: Path) -> None:
    root = tmp_path / "string_keys"
    root.mkdir(parents=True)
    (root / "data.yaml").write_text(
        yaml.safe_dump({"nc": 2, "names": {"0": "word", "1": "texto"}}),
        encoding="utf-8",
    )
    assert read_class_names(root) == ["word", "texto"]


def test_polygon_and_confidence_lines_parse_to_boxes() -> None:
    polygon = parse_yolo_label_line("0 0.1 0.1 0.9 0.1 0.9 0.9 0.1 0.9")
    assert polygon == (0, pytest.approx((0.1, 0.1, 0.9, 0.9)))

    with_confidence = parse_yolo_label_line("0 0.5 0.5 0.2 0.2 0.93")
    assert with_confidence == (0, pytest.approx((0.4, 0.4, 0.6, 0.6)))

    absolute = parse_yolo_label_line("0 50 25 20 10", image_size=(100, 50))
    assert absolute == (0, pytest.approx((0.4, 0.4, 0.6, 0.6)))


def test_load_source_specs_and_zip_source(tmp_path: Path) -> None:
    inner = tmp_path / "inner" / "dataset"
    _detection_source(inner, names=["texto"], train=2, valid=1)
    archive = tmp_path / "export.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        for path in inner.rglob("*"):
            if path.is_file():
                handle.write(path, path.relative_to(inner.parent))

    config = tmp_path / "sources.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "sources": [
                    {
                        "name": "zipped",
                        "path": str(archive),
                        "kind": "auto",
                        "exclude_classes": [],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    specs = load_source_specs(config)
    assert specs[0].name == "zipped"

    manifest = merge_datasets(specs, tmp_path / "merged", copy_mode="copy")
    assert manifest["sources"][0]["kind"] == "detection"
    assert manifest["totals"]["image_counts"]["val"] == 1
    assert (tmp_path / "merged" / "manifest.json").is_file()
    assert json.loads((tmp_path / "merged" / "classes.json").read_text())[
        "classes"
    ] == ["word"]


def test_refuses_to_overwrite_without_flag(tmp_path: Path) -> None:
    source = _detection_source(tmp_path / "source", names=["word"])
    output = tmp_path / "merged"
    merge_datasets([SourceSpec(path=source)], output, copy_mode="copy")
    with pytest.raises(FileExistsError):
        merge_datasets([SourceSpec(path=source)], output, copy_mode="copy")
    merge_datasets([SourceSpec(path=source)], output, copy_mode="copy", overwrite=True)
