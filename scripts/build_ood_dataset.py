"""Build the held-out Latin/Arabic OOD OCR dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from embedding_ocr.data.ood_dataset import (
    DEFAULT_CONDITIONS,
    OOD_CONDITIONS,
    build_ood_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/ood_latin_arabic_v1")
    parser.add_argument("--seed", type=int, default=20261108)
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=list(DEFAULT_CONDITIONS),
        choices=list(OOD_CONDITIONS),
    )
    parser.add_argument("--font-size", type=int, default=44)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_ood_dataset(
        Path(args.out),
        seed=args.seed,
        conditions=args.conditions,
        font_size=args.font_size,
        overwrite=args.overwrite,
    )
    print(f"Wrote OOD manifest: {manifest}")
    print(f"Dataset root: {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
