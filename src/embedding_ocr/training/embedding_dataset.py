"""Build text-image and image-image contrastive datasets for fine-tuning."""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from embedding_ocr.training.fonts import FontFamily, iter_font_paths
from embedding_ocr.training.render import augment_image, render_text
from embedding_ocr.training.short_texts import TextSample, hard_negative

MODES = ("text_image", "image_image")


@dataclass(frozen=True)
class EmbeddingDatasetConfig:
    """Configuration for synthetic contrastive dataset generation."""

    output_dir: Path
    text_image_rows: int = 60_000
    image_image_rows: int = 20_000
    val_fraction: float = 0.05
    canvas_size: int = 128
    augment: bool = True
    seed: int = 42


@dataclass(frozen=True)
class BuiltDataset:
    """Paths and counts for one generated dataset split."""

    root: Path
    manifests: dict[str, Path]
    counts: dict[str, int]


def build_synthetic_dataset(
    samples: Sequence[TextSample],
    train_fonts: Sequence[FontFamily],
    heldout_fonts: Sequence[FontFamily],
    config: EmbeddingDatasetConfig,
) -> BuiltDataset:
    """Render synthetic triplets and write manifests + PNG files.

    Text-image rows use a text anchor and a rendered positive. Image-image rows
    use two different fonts for the anchor and positive. Validation rows use
    held-out font families only.
    """
    if len(samples) < 2:
        msg = "At least two text samples are required."
        raise ValueError(msg)
    train_paths = [str(path) for path in iter_font_paths(train_fonts)]
    heldout_paths = [str(path) for path in iter_font_paths(heldout_fonts)]
    if not train_paths or not heldout_paths:
        msg = "Both train and held-out font families must contain font files."
        raise ValueError(msg)
    if not 0 <= config.val_fraction < 1:
        msg = f"val_fraction must be in [0, 1), got {config.val_fraction}"
        raise ValueError(msg)

    rng = random.Random(config.seed)
    lexicon = [sample.text for sample in samples]
    rows: dict[str, list[dict[str, object]]] = {mode: [] for mode in MODES}
    val_counts = {
        "text_image": round(config.text_image_rows * config.val_fraction),
        "image_image": round(config.image_image_rows * config.val_fraction),
    }
    total_counts = {
        "text_image": config.text_image_rows,
        "image_image": config.image_image_rows,
    }

    root = Path(config.output_dir)
    train_root = root / "train"
    val_root = root / "val"
    _prepare_dirs(train_root, val_root)

    for mode in MODES:
        for index in range(total_counts[mode]):
            is_val = index < val_counts[mode]
            split = "val" if is_val else "train"
            fonts = heldout_paths if is_val else train_paths
            sample = rng.choice(samples)
            negative_text = hard_negative(sample.text, lexicon, rng=rng)
            row = _render_row(
                mode=mode,
                sample=sample,
                negative_text=negative_text,
                fonts=fonts,
                output_root=val_root if is_val else train_root,
                split=split,
                index=index,
                config=config,
                rng=rng,
            )
            rows[mode].append(row)

    manifests: dict[str, Path] = {}
    for mode in MODES:
        path = train_root / f"{mode}.jsonl"
        _write_manifest(path, [row for row in rows[mode] if row["split"] == "train"])
        manifests[f"train/{mode}"] = path
        val_path = val_root / f"{mode}.jsonl"
        _write_manifest(val_path, [row for row in rows[mode] if row["split"] == "val"])
        manifests[f"val/{mode}"] = val_path

    counts = {
        "train": sum(
            1
            for row in rows["text_image"] + rows["image_image"]
            if row["split"] == "train"
        ),
        "val": sum(
            1
            for row in rows["text_image"] + rows["image_image"]
            if row["split"] == "val"
        ),
    }
    summary = {
        "config": {
            "text_image_rows": config.text_image_rows,
            "image_image_rows": config.image_image_rows,
            "val_fraction": config.val_fraction,
            "canvas_size": config.canvas_size,
            "augment": config.augment,
            "seed": config.seed,
        },
        "train_families": [family.name for family in train_fonts],
        "heldout_families": [family.name for family in heldout_fonts],
        "counts": counts,
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2))
    return BuiltDataset(root=root, manifests=manifests, counts=counts)


def _render_row(
    *,
    mode: str,
    sample: TextSample,
    negative_text: str,
    fonts: Sequence[str],
    output_root: Path,
    split: str,
    index: int,
    config: EmbeddingDatasetConfig,
    rng: random.Random,
) -> dict[str, object]:
    image_dir = output_root / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    positive_font = str(rng.choice(fonts))
    row_id = hashlib.sha1(
        f"{split}|{mode}|{index}|{sample.text_id}".encode()
    ).hexdigest()[:16]

    positive = render_text(
        sample.text,
        positive_font,
        canvas_size=config.canvas_size,
        seed=rng.randrange(1_000_000),
    )
    if config.augment:
        positive = augment_image(
            positive, hard=True, rng=random.Random(rng.randrange(1_000_000))
        )
    positive_path = image_dir / f"{row_id}_positive.png"
    positive.save(positive_path)

    if mode == "text_image":
        anchor: object = sample.text
        anchor_path: str | None = None
    else:
        anchor_font = _different_font(
            [str(font) for font in fonts], positive_font, rng=rng
        )
        anchor_img = render_text(
            sample.text,
            anchor_font,
            canvas_size=config.canvas_size,
            seed=rng.randrange(1_000_000),
        )
        if config.augment:
            anchor_img = augment_image(
                anchor_img, hard=True, rng=random.Random(rng.randrange(1_000_000))
            )
        anchor_path_obj = image_dir / f"{row_id}_anchor.png"
        anchor_img.save(anchor_path_obj)
        anchor = str(anchor_path_obj)
        anchor_path = str(anchor_path_obj)

    negative_font = _different_font(
        [str(font) for font in fonts], positive_font, rng=rng
    )
    negative_img = render_text(
        negative_text,
        negative_font,
        canvas_size=config.canvas_size,
        seed=rng.randrange(1_000_000),
    )
    if config.augment:
        negative_img = augment_image(
            negative_img, hard=True, rng=random.Random(rng.randrange(1_000_000))
        )
    negative_path = image_dir / f"{row_id}_negative.png"
    negative_img.save(negative_path)

    return {
        "id": row_id,
        "anchor": anchor,
        "anchor_path": anchor_path,
        "positive": str(positive_path),
        "negative": str(negative_path),
        "text": sample.text,
        "negative_text": negative_text,
        "text_id": sample.text_id,
        "mode": mode,
        "split": split,
        "font": positive_font,
    }


def _different_font(fonts: Sequence[str], current: str, *, rng: random.Random) -> str:
    candidates = [font for font in fonts if font != current]
    return rng.choice(candidates) if candidates else current


def _prepare_dirs(train_root: Path, val_root: Path) -> None:
    for root in (train_root, val_root):
        (root / "images").mkdir(parents=True, exist_ok=True)


def _write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_manifest(path: str | Path) -> list[dict[str, object]]:
    """Read a JSONL manifest written by :func:`build_synthetic_dataset`."""
    rows: list[dict[str, object]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                loaded = json.loads(line)
                if isinstance(loaded, dict):
                    rows.append(loaded)
    return rows
