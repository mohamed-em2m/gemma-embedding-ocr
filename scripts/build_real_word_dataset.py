"""Build one merged word-detection dataset from real Roboflow-style sources.

All classes (``word``, ``texto``, ...) are collapsed to the single class
``word``.  Classification exports are converted to full-image word boxes.

Examples:
    # Merge extracted Roboflow exports.
    uv run --no-sync python scripts/build_real_word_dataset.py \
        --source artifacts/real_sources/devanagari-word-detection \
        --source artifacts/real_sources/word-detection-using-yolo \
        --source artifacts/real_sources/words-5tolc \
        --source artifacts/real_sources/math-words \
        --out artifacts/yolo_words_real

    # Use a YAML description instead.
    uv run --no-sync python scripts/build_real_word_dataset.py \
        --config configs/real_word_sources.yaml \
        --out artifacts/yolo_words_real

    # Mix real data with the synthetic generator output.
    uv run --no-sync python scripts/build_real_word_dataset.py \
        --config configs/real_word_sources.yaml \
        --source artifacts/yolo_words_synthetic \
        --out artifacts/yolo_words_mixed
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from embedding_ocr.training.real_dataset import (
    SourceSpec,
    load_source_specs,
    merge_datasets,
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _fraction(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed < 1.0:
        raise argparse.ArgumentTypeError("must be in [0, 1)")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Merge Roboflow/YOLO word-detection datasets into one single-class "
            "YOLO dataset (class 0 = word). Classification exports become "
            "full-image word boxes."
        )
    )
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        type=Path,
        help="Dataset directory or .zip export. Repeat for multiple sources.",
    )
    parser.add_argument(
        "--config",
        default=None,
        type=Path,
        help="YAML file with a 'sources' list (see configs/real_word_sources.yaml).",
    )
    parser.add_argument("--out", default=Path("artifacts/yolo_words_real"), type=Path)
    parser.add_argument("--target-class", default="word")
    parser.add_argument("--val-frac", default=0.1, type=_fraction)
    parser.add_argument("--test-frac", default=0.0, type=_fraction)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument(
        "--no-preserve-splits",
        action="store_true",
        help="Re-split every source instead of keeping its train/valid/test folders.",
    )
    parser.add_argument(
        "--no-ensure-val",
        action="store_true",
        help="Do not carve a validation split out of train when it is missing.",
    )
    parser.add_argument(
        "--drop-empty",
        action="store_true",
        help="Drop images whose label files are empty or missing.",
    )
    parser.add_argument(
        "--no-dedupe",
        action="store_true",
        help="Keep duplicate images across sources (deduplication is on by default).",
    )
    parser.add_argument(
        "--copy-mode",
        default="copy",
        choices=("copy", "hardlink", "symlink"),
        help="How to place images in the merged dataset.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-per-source", default=None, type=_positive_int)
    parser.add_argument("--allow-empty-val", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    specs: list[SourceSpec] = []
    if args.config is not None:
        if not args.config.is_file():
            print(f"error: config not found: {args.config}", file=sys.stderr)
            return 2
        specs.extend(load_source_specs(args.config))
    specs.extend(SourceSpec(path=path) for path in args.source)

    if not specs:
        print(
            "error: no sources. Pass --source <dir|zip> or --config <yaml>.",
            file=sys.stderr,
        )
        return 2

    print(f"Merging {len(specs)} source(s) into {args.out} ...")
    try:
        manifest = merge_datasets(
            specs,
            args.out,
            target_class=args.target_class,
            val_fraction=args.val_frac,
            test_fraction=args.test_frac,
            seed=args.seed,
            preserve_splits=not args.no_preserve_splits,
            ensure_val=not args.no_ensure_val,
            keep_empty=not args.drop_empty,
            dedupe=not args.no_dedupe,
            copy_mode=args.copy_mode,
            overwrite=args.overwrite,
            max_per_source=args.max_per_source,
            allow_empty_val=args.allow_empty_val,
        )
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    for source in manifest["sources"]:
        print(
            f"  {source['name']}: {source['kind']}, "
            f"classes={source['class_names']}, "
            f"images={sum(source['image_counts'].values())}, "
            f"boxes={source['boxes_kept']}"
        )
    totals = manifest["totals"]
    print(f"Images: {totals['image_counts']}")
    print(f"Boxes:  {totals['box_counts']}")
    print(f"Duplicates removed: {totals['duplicates_removed']}")
    print(
        "Train with:\n"
        "  uv run --no-sync python scripts/train_yolo_words.py "
        f"--data {args.out / 'data.yaml'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
