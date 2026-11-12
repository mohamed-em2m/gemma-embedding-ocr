"""Build and read line-level OCR manifests for the CTC recognizer.

A manifest is a JSONL file with one record per training example::

    {"image": "iam_line/images/train/iam_line_train_000000.png",
     "text": "A MOVE to stop Mr Gaitskell",
     "text_norm": "A MOVE to stop Mr Gaitskell",
     "script": "latin", "split": "train", "source": "iam_line",
     "orig_id": "train/0", "width": 1720, "height": 64}

Image paths are stored relative to the manifest directory so a prepared
dataset can be moved or mounted somewhere else. The vocabulary is built from
the **train** split only; validation and test rows whose normalized text
contains a character outside it are dropped and counted, never silently
mapped to a neighbouring class.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import shutil
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from itertools import islice
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

from embedding_ocr.recognition.ctc.charset import CharSet
from embedding_ocr.recognition.ctc.normalize import detect_script, normalize_text
from embedding_ocr.utils.io import git_hash

SPLITS = ("train", "val", "test")
_SPLIT_PRIORITY = {split: index for index, split in enumerate(SPLITS)}
# Dataset mirrors disagree: IAM-line calls its split "validation", others "valid".
_SPLIT_ALIASES = {
    "train": ("train", "training"),
    "val": ("val", "validation", "valid", "dev"),
    "test": ("test", "testing"),
}
MANIFEST_NAME = "manifest.jsonl"
STATS_NAME = "stats.json"
IMAGE_EXTENSIONS = frozenset(
    {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
)
_LABEL_SUFFIXES = {".csv", ".tsv", ".jsonl", ".json"}

HfLoader = Callable[[str, Sequence[str]], Mapping[str, Sequence[Mapping[str, Any]]]]


@dataclass(frozen=True)
class FilterConfig:
    """Reject labels and crops that cannot be trained on reliably.

    Attributes:
        min_chars: Minimum number of normalized characters.
        max_chars: Maximum label length; longer lines are truncated by the
            CTC time axis at 32 px height and are not worth padding for.
        max_aspect: Maximum width/height ratio after detection cropping.
        min_side: Minimum crop width and height in pixels.
        image_height: Height the trainer resizes crops to.
        width_stride: Total horizontal downsampling of the encoder. The CTC
            time axis is ``resized_width // width_stride`` and must be at
            least ``2 * len(label) - 1``, otherwise the label cannot be
            aligned and training fails.
    """

    min_chars: int = 1
    max_chars: int = 96
    max_aspect: float = 40.0
    min_side: int = 8
    image_height: int = 32
    width_stride: int = 2


@dataclass(frozen=True)
class LineRecord:
    """One line crop with its label."""

    image: Path
    text: str
    text_norm: str
    script: str
    split: str
    source: str
    orig_id: str
    width: int
    height: int


@dataclass(frozen=True)
class LineSource:
    """A Hugging Face line-image dataset to prepare.

    Attributes:
        name: Short unique name used for directories and record ids.
        hf_id: Hugging Face dataset id, e.g. ``Teklia/IAM-line``.
        script: Force a script instead of detecting it per line.
        splits: Splits to read from the dataset.
    """

    name: str
    hf_id: str
    script: str | None = None
    splits: tuple[str, ...] = SPLITS


def load_hf_splits(
    hf_id: str, splits: Sequence[str]
) -> dict[str, Sequence[Mapping[str, Any]]]:
    """Load splits from the Hugging Face Hub with a lazy ``datasets`` import.

    Requires the ``finetune`` (or ``experiments``) extra. Missing splits are
    skipped instead of failing, because dataset mirrors differ.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - environment dependent
        message = (
            "the 'datasets' package is required to download line datasets; "
            "install it with: uv pip install --python .venv/bin/python -e '.[finetune]'"
        )
        raise RuntimeError(message) from exc

    loaded: dict[str, Sequence[Mapping[str, Any]]] = {}
    for split in splits:
        for candidate in _SPLIT_ALIASES.get(split, (split,)):
            try:
                loaded[split] = load_dataset(hf_id, split=candidate)
            except (ValueError, FileNotFoundError):
                continue
            break
    if not loaded:
        message = f"no usable splits found in {hf_id!r} (looked for {list(splits)})"
        raise ValueError(message)
    return loaded


def _as_image(value: Any) -> Image.Image:
    """Convert a ``datasets`` image cell into a PIL image."""
    if isinstance(value, Image.Image):
        return value
    if isinstance(value, Mapping):
        raw = value.get("bytes")
        if raw:
            return Image.open(io.BytesIO(raw))
        path = value.get("path")
        if path:
            return Image.open(path)
    message = f"unsupported image value of type {type(value).__name__}"
    raise ValueError(message)


def _save_image(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if image.mode not in {"L", "RGB"}:
        image = image.convert("RGB")
    image.save(path, format="PNG")


def filter_reason(
    text: str, width: int, height: int, config: FilterConfig
) -> str | None:
    """Return why a record is unusable, or ``None`` when it passes."""
    if not text:
        return "empty_text"
    if len(text) < config.min_chars:
        return "too_short"
    if len(text) > config.max_chars:
        return "too_long"
    if width < config.min_side or height < config.min_side:
        return "too_small"
    if height <= 0 or width / height > config.max_aspect:
        return "extreme_aspect"
    if (
        config.width_stride > 0
        and ctc_timesteps(width, height, config) < 2 * len(text) - 1
    ):
        return "ctc_infeasible"
    return None


def ctc_timesteps(width: int, height: int, config: FilterConfig) -> int:
    """Return the encoder time steps for a crop resized to ``image_height``."""
    if height <= 0 or config.width_stride <= 0:
        return 0
    resized_width = int(width * config.image_height / height)
    return resized_width // config.width_stride


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def dedupe_records(
    records: Sequence[LineRecord],
) -> tuple[list[LineRecord], dict[str, int]]:
    """Drop exact duplicate crops, keeping the train copy when one exists.

    Keeping the training copy and dropping the evaluation copy removes the
    leak while leaving the evaluation split clean of images the model has
    already seen.
    """
    ordered = sorted(
        records,
        key=lambda record: (
            _SPLIT_PRIORITY.get(record.split, 99),
            record.source,
            record.orig_id,
        ),
    )
    seen: set[str] = set()
    kept: list[LineRecord] = []
    removed: Counter[str] = Counter()
    for record in ordered:
        try:
            digest = _digest(record.image)
        except OSError:
            removed["unreadable"] += 1
            continue
        if digest in seen:
            removed[record.split] += 1
            continue
        seen.add(digest)
        kept.append(record)
    return kept, dict(removed)


def write_manifest(records: Sequence[LineRecord], path: Path) -> None:
    """Write records as JSONL with image paths relative to ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    root = path.parent.resolve()
    ordered = sorted(
        records,
        key=lambda record: (
            _SPLIT_PRIORITY.get(record.split, 99),
            record.source,
            record.orig_id,
        ),
    )
    with path.open("w", encoding="utf-8") as handle:
        for record in ordered:
            try:
                image = str(record.image.resolve().relative_to(root))
            except ValueError:
                image = str(record.image)
            payload = {
                "image": image,
                "text": record.text,
                "text_norm": record.text_norm,
                "script": record.script,
                "split": record.split,
                "source": record.source,
                "orig_id": record.orig_id,
                "width": record.width,
                "height": record.height,
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def read_manifest(path: Path) -> list[LineRecord]:
    """Read a manifest written by :func:`write_manifest`."""
    root = Path(path).parent
    records: list[LineRecord] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
                image = Path(str(payload["image"]))
                records.append(
                    LineRecord(
                        image=image if image.is_absolute() else root / image,
                        text=str(payload["text"]),
                        text_norm=str(payload.get("text_norm", payload["text"])),
                        script=str(payload.get("script", "unknown")),
                        split=str(payload.get("split", "train")),
                        source=str(payload.get("source", "unknown")),
                        orig_id=str(payload.get("orig_id", line_number)),
                        width=int(payload.get("width", 0)),
                        height=int(payload.get("height", 0)),
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                message = f"invalid manifest record at {path}:{line_number}: {exc}"
                raise ValueError(message) from exc
    return records


def split_counts(records: Iterable[LineRecord]) -> dict[str, int]:
    """Count records per split, always returning every known split."""
    counts = Counter(record.split for record in records)
    return {split: counts.get(split, 0) for split in SPLITS}


def prepare_line_datasets(
    sources: Sequence[LineSource],
    output_dir: Path,
    *,
    filters: FilterConfig | None = None,
    loader: HfLoader = load_hf_splits,
    max_per_source: int | None = None,
    limit_per_split: int | None = None,
    dry_run: bool = False,
    overwrite: bool = False,
    strip_diacritics: bool = True,
    fold_alefs: bool = True,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Download line datasets, write crops, and build a manifest.

    Args:
        sources: Datasets to prepare, in priority order.
        output_dir: Destination root. Gets ``manifest.jsonl``,
            ``charset.json``, ``stats.json`` and one directory per source.
        filters: Crop and label filters.
        loader: Injection point for tests; defaults to the Hub loader.
        max_per_source: Keep the first N rows per source.
        limit_per_split: Keep at most this many rows per split (debug runs).
        dry_run: Decode and filter but write no images or manifest.
        overwrite: Replace a non-empty output directory.
        strip_diacritics: Arabic harakat/tatweel policy.
        fold_alefs: Arabic alef folding policy.
        log: Optional log sink.

    Returns:
        The statistics dictionary that is also written to ``stats.json``.
    """
    emit = log or (lambda message: None)
    config = filters or FilterConfig()
    output = Path(output_dir).expanduser()

    if output.exists() and any(output.iterdir()):
        if not overwrite:
            message = (
                f"output directory is not empty: {output}. "
                "Pass overwrite=True (or --overwrite) to replace it."
            )
            raise FileExistsError(message)
        shutil.rmtree(output)
    if not dry_run:
        output.mkdir(parents=True, exist_ok=True)

    records: list[LineRecord] = []
    source_stats: list[dict[str, Any]] = []

    for source in sources:
        payload = loader(source.hf_id, source.splits)
        per_split: dict[str, dict[str, Any]] = {}
        collected: list[LineRecord] = []
        reached_cap = False
        for split in SPLITS:
            if split not in payload or reached_cap:
                continue
            kept = 0
            examined = 0
            split_dropped: Counter[str] = Counter()
            rows = payload[split]
            for index, row in enumerate(rows):
                examined += 1
                if limit_per_split is not None and kept >= limit_per_split:
                    break
                if max_per_source is not None and len(collected) >= max_per_source:
                    reached_cap = True
                    emit(f"{source.name}: stopped at max_per_source={max_per_source}")
                    break
                raw_text = row.get("text")
                if not isinstance(raw_text, str):
                    split_dropped["missing_text"] += 1
                    continue
                text = normalize_text(
                    raw_text,
                    strip_diacritics=strip_diacritics,
                    fold_alefs=fold_alefs,
                )
                try:
                    image = _as_image(row.get("image"))
                except (ValueError, UnidentifiedImageError, OSError):
                    split_dropped["bad_image"] += 1
                    continue
                width, height = image.size
                reason = filter_reason(text, width, height, config)
                if reason is not None:
                    split_dropped[reason] += 1
                    continue
                script = source.script or detect_script(text)
                orig_id = f"{split}/{index}"
                destination = (
                    output
                    / source.name
                    / "images"
                    / split
                    / f"{source.name}_{split}_{kept:06d}.png"
                )
                if not dry_run:
                    _save_image(image, destination)
                collected.append(
                    LineRecord(
                        image=destination,
                        text=raw_text.strip(),
                        text_norm=text,
                        script=script,
                        split=split,
                        source=source.name,
                        orig_id=orig_id,
                        width=width,
                        height=height,
                    )
                )
                kept += 1
            per_split[split] = {
                "available": len(rows),
                "examined": examined,
                "kept": kept,
                "dropped": dict(split_dropped),
            }
            emit(f"{source.name}/{split}: kept {kept} of {examined} examined rows")
        records.extend(collected)
        source_stats.append(
            {
                "name": source.name,
                "hf_id": source.hf_id,
                "script": source.script,
                "kept": len(collected),
                "splits": per_split,
            }
        )

    train_texts = [record.text_norm for record in records if record.split == "train"]
    if not train_texts:
        message = "no usable training rows were collected; refusing to build a charset"
        raise ValueError(message)
    charset = CharSet.from_texts(train_texts)
    emit(f"charset: {len(charset.chars)} characters from {len(train_texts)} train rows")

    coverage_dropped: Counter[str] = Counter()
    covered: list[LineRecord] = []
    for record in records:
        missing = charset.unknown(record.text_norm)
        if missing:
            coverage_dropped[record.split] += 1
            continue
        covered.append(record)
    if coverage_dropped:
        emit(
            f"dropped rows with out-of-vocabulary characters: {dict(coverage_dropped)}"
        )

    dedupe_stats: dict[str, int] = {}
    if dry_run:
        deduped = covered
    else:
        deduped, dedupe_stats = dedupe_records(covered)
        if dedupe_stats:
            emit(f"deduplicated crops: {dedupe_stats}")

    text_splits: dict[str, set[str]] = defaultdict(set)
    for record in deduped:
        text_splits[record.text_norm].add(record.split)
    text_duplicates = sum(1 for splits in text_splits.values() if len(splits) > 1)

    counts = split_counts(deduped)
    by_script = Counter(record.script for record in deduped)
    stats: dict[str, Any] = {
        "created": datetime.now(UTC).isoformat(),
        "git_hash": git_hash(),
        "output": str(output),
        "filters": asdict(config),
        "normalization": {
            "strip_diacritics": strip_diacritics,
            "fold_alefs": fold_alefs,
        },
        "sources": source_stats,
        "totals": {
            "raw_rows": sum(
                split_info["available"]
                for source_stat in source_stats
                for split_info in source_stat["splits"].values()
            ),
            "collected": len(records),
            "kept": len(deduped),
            "by_split": counts,
            "by_script": dict(sorted(by_script.items())),
            "charset_size": charset.size,
            "characters": len(charset.chars),
            "coverage_dropped": dict(coverage_dropped),
            "dedupe": dedupe_stats,
            "text_duplicates_across_splits": text_duplicates,
        },
        "dry_run": dry_run,
    }

    if not dry_run:
        write_manifest(deduped, output / MANIFEST_NAME)
        charset.save(output / "charset.json")
        (output / STATS_NAME).write_text(
            json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        emit(f"wrote {len(deduped)} records to {output / MANIFEST_NAME}")
    return stats


def read_label_table(
    path: Path, *, image_column: str, text_column: str
) -> dict[str, str]:
    """Read an image-to-text mapping from a CSV, TSV, JSON or JSONL file.

    Keys are the raw image references from the table. Callers resolve them
    against a directory because Arabic OCR exports disagree on whether the
    reference is a path, a basename or a bare stem.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in _LABEL_SUFFIXES:
        message = f"unsupported label file type {suffix!r} for {path}"
        raise ValueError(message)

    if suffix in {".jsonl", ".json"}:
        rows: list[dict[str, Any]] = []
        text = path.read_text(encoding="utf-8")
        if suffix == ".json":
            payload = json.loads(text)
            rows = payload if isinstance(payload, list) else payload.get("data", [])
        else:
            rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        table = {}
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            image, label = row.get(image_column), row.get(text_column)
            if isinstance(image, str) and isinstance(label, str):
                table[image] = label
        return table

    delimiter = "\t" if suffix == ".tsv" else ","
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        table = {}
        for row in reader:
            image, label = row.get(image_column), row.get(text_column)
            if isinstance(image, str) and isinstance(label, str):
                table[image] = label
        return table


def _index_label_table(table: Mapping[str, str]) -> dict[str, str]:
    """Index labels by full reference, basename and stem for tolerant lookup."""
    index: dict[str, str] = {}
    for key, value in table.items():
        index[key] = value
        name = Path(key).name
        index.setdefault(name, value)
        index.setdefault(Path(name).stem, value)
    return index


def import_local_lines(
    root: Path,
    *,
    labels_file: Path | None = None,
    image_column: str = "image",
    text_column: str = "text",
    image_dir: Path | None = None,
    filters: FilterConfig | None = None,
) -> list[tuple[Path, str]]:
    """Import line crops from a local directory with a label table.

    Supports the two layouts Arabic OCR exports use most: a label file next
    to an image folder, and one ``.txt`` transcript per image. The mapping is
    explicit; nothing about the dataset is guessed.
    """
    root = Path(root)
    config = filters or FilterConfig()
    directory = Path(image_dir) if image_dir is not None else root
    if not directory.is_dir():
        message = f"image directory not found: {directory}"
        raise FileNotFoundError(message)

    images = sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        message = f"no images found under {directory}"
        raise ValueError(message)

    pairs: list[tuple[Path, str]] = []
    if labels_file is not None:
        index = _index_label_table(
            read_label_table(
                Path(labels_file), image_column=image_column, text_column=text_column
            )
        )
        for image_path in images:
            key = str(image_path.relative_to(directory))
            label = (
                index.get(key)
                or index.get(image_path.name)
                or index.get(image_path.stem)
            )
            if label is None:
                continue
            pairs.append((image_path, label))
    else:
        for image_path in images:
            label_path = image_path.with_suffix(".txt")
            if not label_path.is_file():
                continue
            pairs.append((image_path, label_path.read_text(encoding="utf-8")))

    usable: list[tuple[Path, str]] = []
    for image_path, raw in pairs:
        text = normalize_text(raw)
        try:
            with Image.open(image_path) as image:
                width, height = image.size
        except (UnidentifiedImageError, OSError):
            continue
        if filter_reason(text, width, height, config) is not None:
            continue
        usable.append((image_path, text))
    return usable


def inspect_local_dataset(root: Path, *, max_files: int = 40) -> dict[str, Any]:
    """Summarize a local dataset tree without assuming its layout.

    Reports the directory structure, image counts per folder, and the header
    and first rows of every CSV/TSV/JSON file. Use this before writing an
    importer for a dataset whose schema is unknown.
    """
    root = Path(root)
    if not root.is_dir():
        message = f"dataset directory not found: {root}"
        raise FileNotFoundError(message)

    tree: list[str] = []
    image_counts: Counter[str] = Counter()
    tables: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if len(tree) < max_files:
            tree.append(str(relative) + ("/" if path.is_dir() else ""))
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            image_counts[str(relative.parent)] += 1
        if path.is_file() and path.suffix.lower() in _LABEL_SUFFIXES:
            sample: Any
            try:
                if path.suffix.lower() in {".csv", ".tsv"}:
                    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
                    with path.open(encoding="utf-8", newline="") as handle:
                        reader = csv.reader(handle, delimiter=delimiter)
                        sample = list(islice(reader, 3))
                else:
                    lines = path.read_text(encoding="utf-8").splitlines()[:3]
                    sample = [json.loads(line) for line in lines if line.strip()]
            except (OSError, json.JSONDecodeError, UnicodeDecodeError):
                sample = "unreadable"
            tables.append({"path": str(relative), "sample": sample})

    return {
        "root": str(root),
        "tree": tree,
        "image_counts": dict(sorted(image_counts.items())),
        "total_images": sum(image_counts.values()),
        "tables": tables,
    }
