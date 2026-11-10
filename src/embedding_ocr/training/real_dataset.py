"""Merge real word-detection datasets into one single-class YOLO dataset.

Roboflow Universe exports are not uniform: some are object-detection projects
with classes such as ``word`` or ``texto``, some are classification projects
whose class folders contain cropped words, and some only ship a ``train``
split.  This module normalises all of them to one class (``word``) so they can
be trained together.

Supported input layouts (auto-detected):

* YOLO detection export: ``<root>/{train,valid,test}/images|labels``.
* YOLO detection export with split sub-folders: ``<root>/images/{train,...}``.
* Flat detection folder: ``<root>/images`` + ``<root>/labels`` or images and
  ``.txt`` files side by side.
* Classification export: ``<root>/{train,valid,test}/<class>/*.jpg`` (also the
  single-split ``<root>/<class>/*.jpg`` form).  Each image becomes one
  full-image ``word`` box.

Labels are always rewritten to class ``0`` (``word``).  Class names are read
from ``data.yaml`` when present, otherwise inferred from the label files.
Images are copied (or linked) with a source prefix to avoid collisions, and
duplicate images across sources are removed.

Example:
    uv run --no-sync python scripts/build_real_word_dataset.py \
        --source artifacts/real_sources/devanagari-word-detection \
        --source artifacts/real_sources/words-5tolc \
        --out artifacts/yolo_words_real
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import shutil
import zipfile
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import yaml

MERGER_VERSION = "1.0"
TARGET_CLASS = "word"
SPLITS = ("train", "val", "test")
SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "valid": "val",
    "val": "val",
    "validation": "val",
    "test": "test",
    "testing": "test",
}
IMAGE_EXTENSIONS = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
)
DatasetKind = Literal["auto", "detection", "classification"]
CopyMode = Literal["copy", "hardlink", "symlink"]
_SPLIT_PRIORITY = {split: index for index, split in enumerate(SPLITS)}


@dataclass(frozen=True)
class SourceSpec:
    """One dataset to merge.

    Args:
        path: Dataset directory or a ``.zip`` export.
        name: Short unique name used as the filename prefix and in reports.
        kind: ``auto`` (default), ``detection`` or ``classification``.
        include_classes: If set, only these source class names are kept.
        exclude_classes: Source class names to drop.
    """

    path: Path
    name: str | None = None
    kind: DatasetKind = "auto"
    include_classes: tuple[str, ...] | None = None
    exclude_classes: tuple[str, ...] = ()
    class_map: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class _Record:
    image: Path
    split: str
    source_index: int
    source_name: str
    source_class: str | None
    label_lines: tuple[str, ...]
    empty: bool


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return cleaned.strip("._-") or "source"


def _unique_source_names(specs: Sequence[SourceSpec]) -> list[str]:
    names: list[str] = []
    seen: dict[str, int] = {}
    for spec in specs:
        base = _slug(spec.name or spec.path.stem)
        count = seen.get(base, 0)
        seen[base] = count + 1
        names.append(base if count == 0 else f"{base}_{count + 1}")
    return names


def _image_files(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _has_images(directory: Path) -> bool:
    try:
        return any(
            path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            for path in directory.iterdir()
        )
    except (FileNotFoundError, NotADirectoryError):
        return False


def _read_data_yaml(root: Path) -> dict[str, Any] | None:
    for name in ("data.yaml", "data.yml", "dataset.yaml", "dataset.yml"):
        candidate = root / name
        if candidate.is_file():
            loaded = yaml.safe_load(candidate.read_text(encoding="utf-8"))
            return loaded if isinstance(loaded, dict) else {}
    for candidate in sorted(root.rglob("data.yaml")):
        loaded = yaml.safe_load(candidate.read_text(encoding="utf-8"))
        return loaded if isinstance(loaded, dict) else {}
    return None


def read_class_names(root: Path) -> list[str]:
    """Read ``names`` from a YOLO ``data.yaml``; return ``[]`` if absent."""
    data = _read_data_yaml(root)
    if not data:
        return []
    names = data.get("names")
    if isinstance(names, Mapping):
        normalised: dict[int, str] = {}
        for key, value in names.items():
            try:
                normalised[int(key)] = str(value)
            except (TypeError, ValueError):
                continue
        if not normalised:
            return []
        return [
            normalised.get(index, f"class_{index}")
            for index in range(max(normalised) + 1)
        ]
    if isinstance(names, (list, tuple)):
        return [str(name) for name in names]
    return []


def _split_dirs(root: Path) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for child in sorted(root.iterdir()):
        if child.is_dir() and child.name.lower() in SPLIT_ALIASES:
            found[SPLIT_ALIASES[child.name.lower()]] = child
    if not found:
        found["train"] = root
    return found


def _unwrap_single_root(root: Path) -> Path:
    """Return the single top-level directory of a zip export, if any."""
    entries = [path for path in root.iterdir() if not path.name.startswith(".")]
    if (
        len(entries) == 1
        and entries[0].is_dir()
        and not _has_images(root)
        and not (root / "images").exists()
    ):
        return entries[0]
    return root


def detect_dataset_kind(root: Path) -> DatasetKind:
    """Detect whether a dataset directory holds detections or classification."""
    if not root.is_dir():
        message = f"Dataset directory not found: {root}"
        raise FileNotFoundError(message)

    if (root / "images").is_dir():
        return "detection"
    if any(path.is_dir() and path.name.lower() == "labels" for path in root.rglob("*")):
        return "detection"
    if _read_data_yaml(root) is not None:
        for split_dir in _split_dirs(root).values():
            if (split_dir / "images").is_dir():
                return "detection"
    for split_dir in _split_dirs(root).values():
        if any(child.is_dir() and _has_images(child) for child in split_dir.iterdir()):
            return "classification"
    if _has_images(root):
        return "classification"
    message = (
        f"Could not detect a YOLO detection or classification layout in {root}. "
        "Expected train/valid/test folders with images and labels, or class folders."
    )
    raise ValueError(message)


def _detection_layout(root: Path) -> dict[str, tuple[Path, Path]]:
    layout: dict[str, tuple[Path, Path]] = {}

    images_root = root / "images"
    if images_root.is_dir():
        labels_root = root / "labels"
        split_images = {
            SPLIT_ALIASES[child.name.lower()]: child
            for child in sorted(images_root.iterdir())
            if child.is_dir() and child.name.lower() in SPLIT_ALIASES
        }
        if split_images:
            for split, images_dir in split_images.items():
                candidate = labels_root / images_dir.name
                if not candidate.is_dir():
                    candidate = labels_root / split
                layout[split] = (
                    images_dir,
                    candidate if candidate.is_dir() else labels_root,
                )
            return layout
        layout["train"] = (
            images_root,
            labels_root if labels_root.is_dir() else images_root,
        )
        return layout

    for split, split_dir in _split_dirs(root).items():
        images_dir = (
            split_dir / "images" if (split_dir / "images").is_dir() else split_dir
        )
        labels_dir = (
            split_dir / "labels" if (split_dir / "labels").is_dir() else split_dir
        )
        layout[split] = (images_dir, labels_dir)
    return layout


def _classification_layout(root: Path) -> dict[str, list[tuple[str | None, Path]]]:
    layout: dict[str, list[tuple[str | None, Path]]] = {}
    for split, split_dir in _split_dirs(root).items():
        groups: list[tuple[str | None, Path]] = []
        direct_images = _has_images(split_dir)
        if direct_images:
            groups.append((None, split_dir))
        for child in sorted(split_dir.iterdir()):
            if child.is_dir() and _has_images(child):
                groups.append((child.name, child))
        layout[split] = groups
    return layout


def _infer_class_count(label_dirs: Iterable[Path]) -> int:
    max_id = -1
    for directory in label_dirs:
        if not directory.is_dir():
            continue
        for path in directory.glob("*.txt"):
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for line in text.splitlines():
                parts = line.split()
                if not parts:
                    continue
                try:
                    max_id = max(max_id, int(float(parts[0])))
                except ValueError:
                    continue
    return max_id + 1


def _line_needs_image_size(line: str) -> bool:
    for token in line.split()[1:]:
        try:
            if abs(float(token)) > 1.5:
                return True
        except ValueError:
            continue
    return False


def _image_size(path: Path) -> tuple[int, int] | None:
    try:
        from PIL import Image

        with Image.open(path) as image:
            return image.size
    except (OSError, ValueError):
        return None


def parse_yolo_label_line(
    line: str, image_size: tuple[int, int] | None = None
) -> tuple[int, tuple[float, float, float, float]] | None:
    """Parse a YOLO box, polygon or box+confidence line into a normalised box.

    Returns ``(class_id, (x1, y1, x2, y2))`` with all coordinates in ``[0, 1]``
    or ``None`` when the line is malformed or degenerate.
    """
    parts = line.split()
    if len(parts) < 5:
        return None
    try:
        class_id = int(float(parts[0]))
        values = [float(token) for token in parts[1:]]
    except ValueError:
        return None
    if not values or not all(
        value == value and abs(value) != float("inf") for value in values
    ):
        return None

    width, height = image_size if image_size else (1, 1)
    absolute = any(abs(value) > 1.5 for value in values)

    if len(values) == 4 or (len(values) > 4 and len(values) % 2 == 1):
        cx, cy, box_w, box_h = values[:4]
        if absolute:
            cx, box_w = cx / width, box_w / width
            cy, box_h = cy / height, box_h / height
        x1, y1 = cx - box_w / 2, cy - box_h / 2
        x2, y2 = cx + box_w / 2, cy + box_h / 2
    elif len(values) >= 6 and len(values) % 2 == 0:
        xs = values[0::2]
        ys = values[1::2]
        if absolute:
            xs = [value / width for value in xs]
            ys = [value / height for value in ys]
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    else:
        return None

    x1, x2 = max(0.0, min(1.0, x1)), max(0.0, min(1.0, x2))
    y1, y2 = max(0.0, min(1.0, y1)), max(0.0, min(1.0, y2))
    if x2 - x1 <= 1e-6 or y2 - y1 <= 1e-6:
        return None
    return class_id, (x1, y1, x2, y2)


def _box_to_line(box: tuple[float, float, float, float], class_id: int = 0) -> str:
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    width, height = x2 - x1, y2 - y1
    return f"{class_id} {cx:.6f} {cy:.6f} {width:.6f} {height:.6f}"


def _full_image_line() -> str:
    return _box_to_line((0.0, 0.0, 1.0, 1.0))


def _class_allowed(
    name: str,
    include_classes: frozenset[str] | None,
    exclude_classes: frozenset[str],
) -> bool:
    if name in exclude_classes:
        return False
    return include_classes is None or name in include_classes


def _collect_detection(
    spec: SourceSpec,
    source_index: int,
    source_name: str,
    root: Path,
    target_class: str,
    keep_empty: bool,
) -> tuple[list[_Record], dict[str, Any]]:
    layout = _detection_layout(root)
    class_names = read_class_names(root)
    if not class_names:
        inferred = _infer_class_count(labels for _, labels in layout.values())
        class_names = [f"class_{index}" for index in range(inferred)]
    if not class_names:
        class_names = [target_class]

    include = (
        frozenset(spec.include_classes) if spec.include_classes is not None else None
    )
    exclude = frozenset(spec.exclude_classes)
    class_map = dict(spec.class_map)
    records: list[_Record] = []
    stats: dict[str, Any] = {
        "kind": "detection",
        "class_names": class_names,
        "include_classes": list(spec.include_classes) if spec.include_classes else None,
        "exclude_classes": list(spec.exclude_classes),
        "class_map": class_map,
        "image_counts": {},
        "boxes_kept": 0,
        "boxes_dropped_class_filter": 0,
        "boxes_dropped_invalid": 0,
        "missing_label_files": 0,
        "empty_images": 0,
        "skipped_images": 0,
    }

    for split, (images_dir, labels_dir) in layout.items():
        if not images_dir.is_dir():
            continue
        images = _image_files(images_dir)
        stats["image_counts"][split] = len(images)
        for image_path in images:
            label_path = labels_dir / f"{image_path.stem}.txt"
            if not label_path.is_file():
                stats["missing_label_files"] += 1
                if not keep_empty:
                    stats["skipped_images"] += 1
                    continue
                records.append(
                    _Record(
                        image=image_path,
                        split=split,
                        source_index=source_index,
                        source_name=source_name,
                        source_class=None,
                        label_lines=(),
                        empty=True,
                    )
                )
                stats["empty_images"] += 1
                continue

            text = label_path.read_text(encoding="utf-8", errors="ignore")
            lines = [line for line in text.splitlines() if line.strip()]
            image_size = None
            if any(_line_needs_image_size(line) for line in lines):
                image_size = _image_size(image_path)

            kept: list[str] = []
            for line in lines:
                if image_size is None and _line_needs_image_size(line):
                    stats["boxes_dropped_invalid"] += 1
                    continue
                parsed = parse_yolo_label_line(line, image_size)
                if parsed is None:
                    stats["boxes_dropped_invalid"] += 1
                    continue
                class_id, box = parsed
                source_class = (
                    class_names[class_id]
                    if 0 <= class_id < len(class_names)
                    else f"class_{class_id}"
                )
                if not _class_allowed(source_class, include, exclude):
                    stats["boxes_dropped_class_filter"] += 1
                    continue
                mapped = class_map.get(source_class, target_class)
                if mapped != target_class:
                    stats["boxes_dropped_class_filter"] += 1
                    continue
                kept.append(_box_to_line(box, 0))
                stats["boxes_kept"] += 1

            if not kept and not keep_empty:
                stats["skipped_images"] += 1
                continue
            records.append(
                _Record(
                    image=image_path,
                    split=split,
                    source_index=source_index,
                    source_name=source_name,
                    source_class=None,
                    label_lines=tuple(kept),
                    empty=not kept,
                )
            )
            if not kept:
                stats["empty_images"] += 1
    return records, stats


def _collect_classification(
    spec: SourceSpec,
    source_index: int,
    source_name: str,
    root: Path,
    target_class: str,
    keep_empty: bool,
) -> tuple[list[_Record], dict[str, Any]]:
    layout = _classification_layout(root)
    include = (
        frozenset(spec.include_classes) if spec.include_classes is not None else None
    )
    exclude = frozenset(spec.exclude_classes)
    records: list[_Record] = []
    stats: dict[str, Any] = {
        "kind": "classification",
        "class_names": [],
        "include_classes": list(spec.include_classes) if spec.include_classes else None,
        "exclude_classes": list(spec.exclude_classes),
        "class_map": dict(spec.class_map),
        "image_counts": {},
        "boxes_kept": 0,
        "boxes_dropped_class_filter": 0,
        "boxes_dropped_invalid": 0,
        "missing_label_files": 0,
        "empty_images": 0,
        "skipped_images": 0,
    }
    class_names: set[str] = set()
    for split, groups in layout.items():
        count = 0
        for source_class, directory in groups:
            name = source_class or target_class
            class_names.add(name)
            if not _class_allowed(name, include, exclude):
                continue
            for image_path in _image_files(directory):
                count += 1
                records.append(
                    _Record(
                        image=image_path,
                        split=split,
                        source_index=source_index,
                        source_name=source_name,
                        source_class=name,
                        label_lines=(_full_image_line(),),
                        empty=False,
                    )
                )
                stats["boxes_kept"] += 1
        stats["image_counts"][split] = count
    stats["class_names"] = sorted(class_names)
    return records, stats


def _collect_source(
    spec: SourceSpec,
    source_index: int,
    source_name: str,
    root: Path,
    target_class: str,
    keep_empty: bool,
) -> tuple[list[_Record], dict[str, Any]]:
    kind = spec.kind
    if kind == "auto":
        kind = detect_dataset_kind(root)
    stats: dict[str, Any]
    if kind == "detection":
        records, stats = _collect_detection(
            spec, source_index, source_name, root, target_class, keep_empty
        )
    else:
        records, stats = _collect_classification(
            spec, source_index, source_name, root, target_class, keep_empty
        )
    stats.update(
        {
            "name": source_name,
            "root": str(root),
            "kind": kind,
        }
    )
    return records, stats


def _file_digest(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _dedupe(records: Sequence[_Record]) -> tuple[list[_Record], int]:
    ordered = sorted(
        records,
        key=lambda record: (
            _SPLIT_PRIORITY.get(record.split, 99),
            record.source_index,
            str(record.image),
        ),
    )
    seen: set[str] = set()
    kept: list[_Record] = []
    removed = 0
    for record in ordered:
        try:
            digest = _file_digest(record.image)
        except OSError:
            removed += 1
            continue
        if digest in seen:
            removed += 1
            continue
        seen.add(digest)
        kept.append(record)
    return kept, removed


def _cap_per_source(
    records: Sequence[_Record], max_per_source: int | None, seed: int
) -> tuple[list[_Record], dict[int, int]]:
    if max_per_source is None:
        return list(records), {}
    grouped: dict[int, list[_Record]] = defaultdict(list)
    for record in records:
        grouped[record.source_index].append(record)
    kept: list[_Record] = []
    dropped: dict[int, int] = {}
    for source_index in sorted(grouped):
        items = grouped[source_index]
        if len(items) <= max_per_source:
            kept.extend(items)
            continue
        rng = random.Random(seed + source_index * 7919)
        kept.extend(rng.sample(items, max_per_source))
        dropped[source_index] = len(items) - max_per_source
    return kept, dropped


def _assign_splits(
    records: Sequence[_Record],
    *,
    preserve_splits: bool,
    val_fraction: float,
    test_fraction: float,
    seed: int,
    ensure_val: bool,
) -> list[_Record]:
    grouped: dict[int, list[_Record]] = defaultdict(list)
    for record in records:
        grouped[record.source_index].append(record)

    assigned: list[_Record] = []
    for source_index in sorted(grouped):
        items = grouped[source_index]
        rng = random.Random(seed + source_index * 104_729)
        if not preserve_splits:
            pool = list(items)
            rng.shuffle(pool)
            total = len(pool)
            n_test = min(round(total * test_fraction), max(0, total - 1))
            n_val = min(round(total * val_fraction), max(0, total - n_test - 1))
            if ensure_val and n_val == 0 and total - n_test >= 2:
                n_val = 1
            buckets = {
                "test": pool[:n_test],
                "val": pool[n_test : n_test + n_val],
                "train": pool[n_test + n_val :],
            }
        else:
            buckets = {split: [] for split in SPLITS}
            for record in items:
                buckets[record.split if record.split in buckets else "train"].append(
                    record
                )
            if not buckets["train"]:
                buckets["train"] = buckets["val"] + buckets["test"]
                buckets["val"] = []
                buckets["test"] = []
            if ensure_val and not buckets["val"] and len(buckets["train"]) >= 2:
                rng.shuffle(buckets["train"])
                n_val = max(1, round(len(buckets["train"]) * val_fraction))
                n_val = min(n_val, len(buckets["train"]) - 1)
                buckets["val"] = buckets["train"][:n_val]
                buckets["train"] = buckets["train"][n_val:]
            if test_fraction > 0 and not buckets["test"] and len(buckets["train"]) >= 2:
                rng.shuffle(buckets["train"])
                n_test = min(
                    round(len(buckets["train"]) * test_fraction),
                    len(buckets["train"]) - 1,
                )
                buckets["test"] = buckets["train"][:n_test]
                buckets["train"] = buckets["train"][n_test:]

        for split in SPLITS:
            for record in buckets[split]:
                assigned.append(replace(record, split=split))
    return assigned


def _transfer(source: Path, destination: Path, mode: CopyMode) -> None:
    if mode == "symlink":
        try:
            os.symlink(source.resolve(), destination)
            return
        except OSError:
            pass
    elif mode == "hardlink":
        try:
            os.link(source, destination)
            return
        except OSError:
            pass
    shutil.copy2(source, destination)


def _unique_stem(base: str, used: set[str]) -> str:
    candidate = base
    index = 1
    while candidate in used:
        candidate = f"{base}_{index}"
        index += 1
    used.add(candidate)
    return candidate


def _write_dataset(
    records: Sequence[_Record],
    output_dir: Path,
    target_class: str,
    copy_mode: CopyMode,
) -> dict[str, Any]:
    for split in SPLITS:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    used: dict[str, set[str]] = {split: set() for split in SPLITS}
    image_counts = {split: 0 for split in SPLITS}
    box_counts = {split: 0 for split in SPLITS}
    empty_counts = {split: 0 for split in SPLITS}

    ordered = sorted(
        records,
        key=lambda record: (
            _SPLIT_PRIORITY.get(record.split, 99),
            record.source_name,
            str(record.image),
        ),
    )
    for record in ordered:
        extension = record.image.suffix.lower()
        if extension not in IMAGE_EXTENSIONS:
            extension = ".jpg"
        stem = _unique_stem(
            f"{record.source_name}__{_slug(record.image.stem)}", used[record.split]
        )
        image_destination = output_dir / "images" / record.split / f"{stem}{extension}"
        label_destination = output_dir / "labels" / record.split / f"{stem}.txt"
        _transfer(record.image, image_destination, copy_mode)
        label_destination.write_text(
            "\n".join(record.label_lines) + ("\n" if record.label_lines else ""),
            encoding="utf-8",
        )
        image_counts[record.split] += 1
        box_counts[record.split] += len(record.label_lines)
        if record.empty:
            empty_counts[record.split] += 1

    data: dict[str, Any] = {
        "train": "images/train",
        "val": "images/val",
        "nc": 1,
        "names": [target_class],
    }
    if image_counts["test"] > 0:
        data["test"] = "images/test"
    (output_dir / "data.yaml").write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    (output_dir / "classes.json").write_text(
        json.dumps(
            {"classes": [target_class], "class_to_id": {target_class: 0}},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "image_counts": image_counts,
        "box_counts": box_counts,
        "empty_counts": empty_counts,
    }


def _safe_extract(archive: zipfile.ZipFile, destination: Path) -> None:
    root = destination.resolve()
    for member in archive.infolist():
        target = (destination / member.filename).resolve()
        if not target.is_relative_to(root):
            message = f"Unsafe path in zip archive: {member.filename}"
            raise ValueError(message)
        archive.extract(member, destination)


def _prepare_source_path(
    spec: SourceSpec, source_index: int, source_name: str, cache_dir: Path
) -> Path:
    path = spec.path.expanduser()
    if not path.exists():
        message = f"Dataset source not found: {path}"
        raise FileNotFoundError(message)
    if path.suffix.lower() == ".zip":
        extracted = cache_dir / f"{source_index:02d}_{source_name}"
        if not extracted.exists() or not any(extracted.iterdir()):
            extracted.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(path) as archive:
                _safe_extract(archive, extracted)
        path = extracted
    return _unwrap_single_root(path)


def load_source_specs(config_path: str | Path) -> list[SourceSpec]:
    """Load a list of :class:`SourceSpec` entries from a YAML config."""
    path = Path(config_path)
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, Mapping):
        message = f"Invalid source config (expected a mapping): {path}"
        raise ValueError(message)
    entries = loaded.get("sources", [])
    if not isinstance(entries, list):
        message = f"Invalid 'sources' list in {path}"
        raise ValueError(message)
    specs: list[SourceSpec] = []
    for entry in entries:
        if not isinstance(entry, Mapping) or "path" not in entry:
            continue
        include = entry.get("include_classes")
        exclude = entry.get("exclude_classes", [])
        kind = entry.get("kind", "auto")
        if kind not in {"auto", "detection", "classification"}:
            message = f"Invalid kind {kind!r} in {path}"
            raise ValueError(message)
        specs.append(
            SourceSpec(
                path=Path(str(entry["path"])),
                name=str(entry["name"]) if entry.get("name") else None,
                kind=kind,
                include_classes=tuple(str(item) for item in include)
                if isinstance(include, list)
                else None,
                exclude_classes=tuple(str(item) for item in exclude)
                if isinstance(exclude, list)
                else (),
                class_map={
                    str(key): str(value)
                    for key, value in (entry.get("class_map") or {}).items()
                },
            )
        )
    return specs


def merge_datasets(
    sources: Sequence[SourceSpec],
    output_dir: str | Path,
    *,
    target_class: str = TARGET_CLASS,
    val_fraction: float = 0.1,
    test_fraction: float = 0.0,
    seed: int = 42,
    preserve_splits: bool = True,
    ensure_val: bool = True,
    keep_empty: bool = True,
    dedupe: bool = True,
    copy_mode: CopyMode = "copy",
    overwrite: bool = False,
    max_per_source: int | None = None,
    allow_empty_val: bool = False,
) -> dict[str, Any]:
    """Merge detection/classification sources into one single-class YOLO dataset.

    Returns the written manifest as a dictionary.
    """
    if not sources:
        message = "At least one source dataset is required."
        raise ValueError(message)
    if not 0.0 <= val_fraction < 1.0:
        message = "val_fraction must be in [0, 1)."
        raise ValueError(message)
    if not 0.0 <= test_fraction < 1.0:
        message = "test_fraction must be in [0, 1)."
        raise ValueError(message)
    if val_fraction + test_fraction >= 1.0:
        message = "val_fraction + test_fraction must be < 1."
        raise ValueError(message)
    if max_per_source is not None and max_per_source <= 0:
        message = "max_per_source must be positive."
        raise ValueError(message)
    if copy_mode not in {"copy", "hardlink", "symlink"}:
        message = f"Unknown copy mode: {copy_mode!r}"
        raise ValueError(message)

    output = Path(output_dir).expanduser()
    if output.exists() and any(output.iterdir()):
        if not overwrite:
            message = (
                f"Output directory is not empty: {output}. "
                "Pass overwrite=True (or --overwrite) to replace it."
            )
            raise FileExistsError(message)
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    cache_dir = output.parent / f".{output.name}_sources"
    cache_dir.mkdir(parents=True, exist_ok=True)

    names = _unique_source_names(sources)
    all_records: list[_Record] = []
    source_stats: list[dict[str, Any]] = []
    for index, (spec, name) in enumerate(zip(sources, names, strict=True)):
        root = _prepare_source_path(spec, index, name, cache_dir)
        records, stats = _collect_source(
            spec, index, name, root, target_class, keep_empty
        )
        all_records.extend(records)
        source_stats.append(stats)

    records, capped = _cap_per_source(all_records, max_per_source, seed)
    for source_index, dropped in capped.items():
        source_stats[source_index]["capped_images"] = dropped

    duplicates = 0
    if dedupe:
        records, duplicates = _dedupe(records)

    records = _assign_splits(
        records,
        preserve_splits=preserve_splits,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        seed=seed,
        ensure_val=ensure_val,
    )
    planned_val = sum(1 for record in records if record.split == "val")
    if planned_val == 0 and not allow_empty_val:
        message = (
            "No validation images were produced. Add more images, adjust "
            "val_fraction, or pass allow_empty_val=True."
        )
        raise ValueError(message)
    counts = _write_dataset(records, output, target_class, copy_mode)

    manifest: dict[str, Any] = {
        "version": MERGER_VERSION,
        "created": datetime.now(UTC).isoformat(),
        "target_class": target_class,
        "classes": [target_class],
        "seed": seed,
        "val_fraction": val_fraction,
        "test_fraction": test_fraction,
        "preserve_splits": preserve_splits,
        "ensure_val": ensure_val,
        "keep_empty": keep_empty,
        "dedupe": dedupe,
        "copy_mode": copy_mode,
        "sources": source_stats,
        "totals": {
            **counts,
            "duplicates_removed": duplicates,
            "source_images": sum(
                sum(stats["image_counts"].values()) for stats in source_stats
            ),
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest
