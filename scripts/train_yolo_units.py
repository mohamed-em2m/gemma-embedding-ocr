"""Train a YOLO character/unit detector on the synthetic letter dataset.

Ultralytics YOLO is AGPL-3.0 or commercially licensed. Review the license
before distributing trained weights or integrating this backend.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a YOLO detector for Latin/Arabic character units."
    )
    parser.add_argument(
        "--data",
        default=Path("artifacts/yolo_letters/data.yaml"),
        type=Path,
    )
    parser.add_argument("--model", default="yolo26s.pt")
    parser.add_argument("--epochs", default=100, type=_positive_int)
    parser.add_argument("--imgsz", default=1280, type=_positive_int)
    parser.add_argument("--batch", default=-1, type=int)
    parser.add_argument("--device", default=None)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--patience", default=20, type=_positive_int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--project", default=Path("artifacts/yolo_runs"), type=Path)
    parser.add_argument("--name", default="yolo_units")
    parser.add_argument("--max-det", default=1000, type=_positive_int)
    parser.add_argument("--degrees", default=2.0, type=float)
    parser.add_argument("--mosaic", default=0.2, type=float)
    parser.add_argument("--cache", default="none", choices=("none", "ram", "disk"))
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    data_path = Path(args.data)
    if not data_path.exists():
        print(
            f"error: dataset config not found: {data_path}\n"
            "Run scripts/build_letter_dataset.py first.",
            file=sys.stderr,
        )
        return 2

    try:
        from ultralytics import YOLO  # type: ignore[import-untyped]
    except ImportError:
        print(
            "error: ultralytics is not installed. "
            "Run `uv pip install --python .venv/bin/python -e '.[train]'` or `bash scripts/setup_env.sh --train`.",
            file=sys.stderr,
        )
        return 2

    model = YOLO(args.model)
    cache = False if args.cache == "none" else args.cache
    train_args = {
        "data": str(data_path),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "workers": args.workers,
        "patience": args.patience,
        "seed": args.seed,
        "project": str(Path(args.project).resolve()),
        "name": args.name,
        "max_det": args.max_det,
        "degrees": args.degrees,
        "mosaic": args.mosaic,
        "close_mosaic": 10,
        "fliplr": 0.0,
        "flipud": 0.0,
        "translate": 0.1,
        "scale": 0.3,
        "hsv_h": 0.015,
        "hsv_s": 0.3,
        "hsv_v": 0.3,
        "erasing": 0.0,
        "single_cls": False,
        "rect": False,
        "cache": cache,
        "pretrained": True,
        "deterministic": True,
        "plots": True,
        "save": True,
        "exist_ok": False,
    }
    if args.resume:
        train_args["resume"] = True
    results = model.train(**train_args)
    save_dir = getattr(results, "save_dir", None)
    if save_dir is not None:
        print(f"Training results: {save_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
