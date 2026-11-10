"""Build a synthetic word-detection dataset for YOLO training.

Each image gets one YOLO box per rendered word (class 0: ``word``). Words are
rendered with real font shaping for Latin and Arabic, placed on procedural or
photo backgrounds, and split by font so validation/test fonts never leak into
training.

Example:
    uv run --no-sync python scripts/build_word_dataset.py \
        --fonts /usr/share/fonts --out artifacts/yolo_words \
        --n 30000 --size 1280 --layout mixed
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from embedding_ocr.training.letter_dataset import build_dataset, discover_fonts


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
        description="Generate a YOLO word-detection dataset from fonts."
    )
    parser.add_argument("--fonts", required=True, type=Path)
    parser.add_argument("--out", default=Path("artifacts/yolo_words"), type=Path)
    parser.add_argument("--n", default=30000, type=_positive_int)
    parser.add_argument("--size", default=1280, type=_positive_int)
    parser.add_argument(
        "--layout", default="mixed", choices=("line", "collage", "mixed")
    )
    parser.add_argument("--val-frac", default=0.1, type=_fraction)
    parser.add_argument("--test-frac", default=0.1, type=_fraction)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--bgs", default=None, type=Path)
    parser.add_argument("--words-latin", default=None, type=Path)
    parser.add_argument("--words-arabic", default=None, type=Path)
    parser.add_argument("--words", default=None, type=Path)
    parser.add_argument("--min-font-size", default=None, type=_positive_int)
    parser.add_argument("--max-font-size", default=None, type=_positive_int)
    parser.add_argument("--max-words", default=8, type=_positive_int)
    parser.add_argument("--max-lines", default=6, type=_positive_int)
    parser.add_argument("--empty-frac", default=0.02, type=_fraction)
    parser.add_argument("--preview-count", default=30, type=int)
    parser.add_argument("--max-fonts", default=None, type=_positive_int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        fonts = discover_fonts(args.fonts)
    except (ImportError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if not fonts:
        print(
            "error: no fonts cover the full Latin or Arabic charset",
            file=sys.stderr,
        )
        return 2

    latin_count = sum("latin" in font.scripts for font in fonts)
    arabic_count = sum("arabic" in font.scripts for font in fonts)
    print(f"Discovered {len(fonts)} fonts ({latin_count} Latin, {arabic_count} Arabic)")
    try:
        manifest = build_dataset(
            fonts_dir=args.fonts,
            output_dir=args.out,
            n=args.n,
            size=args.size,
            val_fraction=args.val_frac,
            test_fraction=args.test_frac,
            seed=args.seed,
            backgrounds_dir=args.bgs,
            latin_words=args.words_latin,
            arabic_words=args.words_arabic,
            mixed_words=args.words,
            layout=args.layout,
            min_font_size=args.min_font_size,
            max_font_size=args.max_font_size,
            max_words=args.max_words,
            max_lines=args.max_lines,
            empty_fraction=args.empty_frac,
            preview_count=args.preview_count,
            max_fonts=args.max_fonts,
            label_level="word",
        )
    except (ImportError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"Wrote word dataset to {args.out}")
    print(f"Images: {manifest['image_counts']}")
    print(f"Labels: {manifest['label_counts']}")
    print(f"Classes: {manifest['classes']}")
    print(
        "Train with:\n"
        "  uv run --no-sync python scripts/train_yolo_words.py "
        f"--data {args.out / 'data.yaml'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
