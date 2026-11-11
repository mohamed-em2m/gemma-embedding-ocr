#!/usr/bin/env python
"""Import real word crops from a Hugging Face dataset into a JSONL manifest.

Dataset IDs and column names differ between OCR corpora, so they are explicit
arguments rather than guesses. Example:

  uv run --no-sync python scripts/import_real_crops.py \
    --dataset <verified-id> --config <verified-config> --split train \
    --image-column image --text-column text --limit 50000 \
    --output data/real_crops
"""

from __future__ import annotations

import argparse
import io
import logging
from pathlib import Path

from PIL import Image

from embedding_ocr.training.real_text_crops import save_real_crops

LOGGER = logging.getLogger("import_real_crops")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--image-column", default="image")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--limit", type=int, default=10_000)
    parser.add_argument("--max-documents", type=int, default=500_000)
    parser.add_argument("--source", default=None)
    parser.add_argument("--output", type=Path, default=Path("data/real_crops"))
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    from datasets import load_dataset

    dataset = load_dataset(
        args.dataset,
        args.config,
        split=args.split,
        streaming=True,
    )
    crops: list[tuple[Image.Image, str]] = []
    for index, row in enumerate(dataset):
        if index >= args.max_documents or len(crops) >= args.limit:
            break
        text = row.get(args.text_column)
        if not isinstance(text, str):
            continue
        image = _to_pil(row.get(args.image_column))
        if image is None:
            continue
        crops.append((image, text))
    if not crops:
        raise SystemExit("No valid rows imported; check column names and split.")
    manifest = save_real_crops(
        crops,
        args.output,
        source=args.source or args.dataset,
        prefix="real",
    )
    LOGGER.info("Wrote %d crops to %s", len(crops), manifest)


def _to_pil(value: object) -> Image.Image | None:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, dict):
        raw = value.get("bytes")
        path = value.get("path")
        if isinstance(raw, bytes):
            return Image.open(io.BytesIO(raw)).convert("RGB")
        if isinstance(path, str):
            return Image.open(path).convert("RGB")
    if isinstance(value, str):
        return Image.open(value).convert("RGB")
    return None


if __name__ == "__main__":
    main()
