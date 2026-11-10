#!/usr/bin/env python3
"""Evaluate a trained YOLO word detector on a split of the merged dataset.

Reports overall box metrics and, with ``--per-source``, the same metrics for
each original dataset (matched by the ``<source>__`` filename prefix written by
``scripts/build_real_word_dataset.py``). Per-source runs reuse Ultralytics' own
mAP implementation on a temporary directory of symlinks, so the numbers stay
comparable to the overall run.

Ultralytics YOLO is AGPL-3.0 or commercially licensed. Review the license
before distributing trained weights or integrating this backend.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

_CONSOLE = Console()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a YOLO word detector and report box metrics."
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("artifacts/yolo_runs/words_960/weights/best.pt"),
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("artifacts/yolo_words_real/data.yaml"),
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument(
        "--conf",
        type=float,
        default=0.001,
        help="Low threshold so the mAP curve covers the full recall range.",
    )
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument(
        "--per-source",
        action="store_true",
        help="Also report metrics for each original dataset separately.",
    )
    parser.add_argument("--out", type=Path, default=None)
    return parser


def _box_metrics(result: Any) -> dict[str, float]:
    box = result.box
    return {
        "precision": float(box.mp),
        "recall": float(box.mr),
        "map50": float(box.map50),
        "map50_95": float(box.map),
    }


def _source_of(path: Path) -> str:
    stem = path.stem
    return stem.split("__", 1)[0] if "__" in stem else "unknown"


def _link_subset(image_dir: Path, label_dir: Path, prefix: str, target: Path) -> int:
    images_out = target / "images"
    labels_out = target / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)
    count = 0
    for image in sorted(image_dir.iterdir()):
        if not image.is_file() or not image.stem.startswith(f"{prefix}__"):
            continue
        label = label_dir / f"{image.stem}.txt"
        if not label.exists():
            continue
        (images_out / image.name).symlink_to(image.resolve())
        (labels_out / label.name).symlink_to(label.resolve())
        count += 1
    return count


def _write_subset_yaml(target: Path, image_dir: Path) -> Path:
    config = target / "data.yaml"
    resolved = str(image_dir.resolve())
    # Ultralytics requires both keys even when only validating.
    config.write_text(
        f"train: {resolved}\nval: {resolved}\nnc: 1\nnames:\n- word\n",
        encoding="utf-8",
    )
    return config


def _count_images(data: Path, split: str) -> int:
    """Resolve the split directory the same way Ultralytics does."""
    try:
        from ultralytics.data.utils import (
            check_det_dataset,  # type: ignore[import-untyped]
        )

        info = check_det_dataset(str(data))
        key = split if split in info else "val"
        resolved = info.get(key)
        if isinstance(resolved, str) and Path(resolved).is_dir():
            return len([p for p in Path(resolved).iterdir() if p.is_file()])
    except Exception:  # noqa: BLE001 - fall back to a plain relative path
        pass
    root = data.parent
    for candidate in (root / "images" / split, root / split / "images"):
        if candidate.is_dir():
            return len([p for p in candidate.iterdir() if p.is_file()])
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not args.weights.exists():
        print(f"error: weights not found: {args.weights}", file=sys.stderr)
        return 2
    if not args.data.exists():
        print(f"error: dataset config not found: {args.data}", file=sys.stderr)
        return 2

    try:
        from ultralytics import YOLO  # type: ignore[import-untyped]
    except ImportError:
        print(
            "error: ultralytics is not installed. "
            "Run `uv pip install --python .venv/bin/python -e '.[train]'`.",
            file=sys.stderr,
        )
        return 2

    model = YOLO(str(args.weights))
    val_args: dict[str, Any] = {
        "data": str(args.data),
        "split": args.split,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "conf": args.conf,
        "iou": args.iou,
        "plots": False,
        "verbose": False,
        "save_json": False,
    }

    _CONSOLE.print(
        f"evaluating [bold]{args.weights}[/] on split [bold]{args.split}[/] "
        f"at {args.imgsz}px"
    )
    overall = _box_metrics(model.val(**val_args))
    overall["images"] = float(_count_images(args.data, args.split))

    rows: list[tuple[str, dict[str, float]]] = [(args.split, overall)]

    if args.per_source:
        root = args.data.parent
        image_dir = root / "images" / args.split
        label_dir = root / "labels" / args.split
        prefixes = sorted({_source_of(p) for p in image_dir.iterdir() if p.is_file()})
        prefixes = [p for p in prefixes if p != "unknown"]
        if not prefixes:
            _CONSOLE.print(
                "[yellow]no '<source>__' filename prefixes found; "
                "skipping per-source breakdown[/]"
            )
        for prefix in prefixes:
            tmp = Path(tempfile.mkdtemp(prefix=f"eval_{prefix}_"))
            try:
                count = _link_subset(image_dir, label_dir, prefix, tmp)
                if count == 0:
                    continue
                config = _write_subset_yaml(tmp, tmp / "images")
                metrics = _box_metrics(
                    model.val(**{**val_args, "data": str(config), "split": "val"})
                )
                metrics["images"] = float(count)
                rows.append((prefix, metrics))
            finally:
                shutil.rmtree(tmp, ignore_errors=True)

    table = Table(title=f"YOLO word detector — {args.weights.name} on '{args.split}'")
    table.add_column("subset", style="bold")
    table.add_column("images", justify="right")
    table.add_column("precision", justify="right")
    table.add_column("recall", justify="right")
    table.add_column("mAP50", justify="right")
    table.add_column("mAP50-95", justify="right")
    for name, metrics in rows:
        table.add_row(
            name,
            f"{int(metrics['images'])}",
            f"{metrics['precision']:.4f}",
            f"{metrics['recall']:.4f}",
            f"{metrics['map50']:.4f}",
            f"{metrics['map50_95']:.4f}",
        )
    _CONSOLE.print(table)

    payload = {
        "weights": str(args.weights),
        "data": str(args.data),
        "split": args.split,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "results": {name: metrics for name, metrics in rows},
    }
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        _CONSOLE.print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
