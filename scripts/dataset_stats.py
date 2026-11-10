#!/usr/bin/env python3
"""Compute dataset statistics for the merged YOLO word dataset.

Reads the merged layout written by ``scripts/build_real_word_dataset.py`` and
emits ``stats.json`` next to it, so the Hugging Face dataset card can quote
numbers that are reproducible instead of hand-copied.
"""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from PIL import Image

SPLITS = ("train", "val", "test")


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[index]


def _source_of(stem: str) -> str:
    return stem.split("__", 1)[0] if "__" in stem else "unknown"


def _read_labels(label: Path) -> list[tuple[float, float, float, float]]:
    boxes: list[tuple[float, float, float, float]] = []
    for line in label.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        _, cx, cy, w, h = parts
        boxes.append((float(cx), float(cy), float(w), float(h)))
    return boxes


def collect(root: Path) -> dict[str, Any]:
    per_split: dict[str, Any] = {}
    per_source: dict[str, dict[str, Any]] = {}
    widths: list[float] = []
    heights: list[float] = []
    box_w: list[float] = []
    box_h: list[float] = []
    aspect: list[float] = []
    boxes_per_image: list[float] = []
    extensions: dict[str, int] = {}

    for split in SPLITS:
        image_dir = root / "images" / split
        label_dir = root / "labels" / split
        images = (
            sorted(p for p in image_dir.iterdir() if p.is_file())
            if image_dir.is_dir()
            else []
        )
        boxes_total = 0
        empty = 0
        for image in images:
            extensions[image.suffix.lower()] = (
                extensions.get(image.suffix.lower(), 0) + 1
            )
            with Image.open(image) as handle:
                width, height = handle.size
            widths.append(width)
            heights.append(height)
            label = label_dir / f"{image.stem}.txt"
            boxes = _read_labels(label) if label.exists() else []
            if not boxes:
                empty += 1
            boxes_total += len(boxes)
            boxes_per_image.append(len(boxes))
            source = per_source.setdefault(
                _source_of(image.stem), {"images": 0, "boxes": 0, "splits": {}}
            )
            source["images"] += 1
            source["boxes"] += len(boxes)
            source["splits"][split] = source["splits"].get(split, 0) + 1
            for _, _, w, h in boxes:
                box_w.append(w)
                box_h.append(h)
                if h > 0:
                    aspect.append(w / h)
        per_split[split] = {
            "images": len(images),
            "boxes": boxes_total,
            "images_without_boxes": empty,
            "boxes_per_image": {
                "mean": round(statistics.fmean(boxes_per_image), 2)
                if boxes_per_image
                else 0,
                "median": statistics.median(boxes_per_image) if boxes_per_image else 0,
                "max": max(boxes_per_image) if boxes_per_image else 0,
            },
        }
        boxes_per_image = []

    return {
        "generated": datetime.now(UTC).isoformat(timespec="seconds"),
        "splits": per_split,
        "totals": {
            "images": sum(s["images"] for s in per_split.values()),
            "boxes": sum(s["boxes"] for s in per_split.values()),
        },
        "classes": ["word"],
        "image_size_px": {
            "width": {
                "min": int(min(widths)) if widths else 0,
                "median": int(statistics.median(widths)) if widths else 0,
                "max": int(max(widths)) if widths else 0,
            },
            "height": {
                "min": int(min(heights)) if heights else 0,
                "median": int(statistics.median(heights)) if heights else 0,
                "max": int(max(heights)) if heights else 0,
            },
        },
        "box_size_normalized": {
            "width": {
                "median": round(statistics.median(box_w), 4) if box_w else 0,
                "p95": round(_quantile(box_w, 0.95), 4),
                "max": round(max(box_w), 4) if box_w else 0,
            },
            "height": {
                "median": round(statistics.median(box_h), 4) if box_h else 0,
                "p95": round(_quantile(box_h, 0.95), 4),
                "max": round(max(box_h), 4) if box_h else 0,
            },
            "aspect_ratio": {
                "median": round(statistics.median(aspect), 3) if aspect else 0,
                "p95": round(_quantile(aspect, 0.95), 3),
            },
        },
        "file_extensions": dict(sorted(extensions.items(), key=lambda kv: -kv[1])),
        "sources": {
            name: {
                "images": info["images"],
                "boxes": info["boxes"],
                "images_per_split": info["splits"],
            }
            for name, info in sorted(per_source.items())
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("artifacts/yolo_words_real"))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    stats = collect(args.root)
    out = args.out or (args.root / "stats.json")
    out.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
