#!/usr/bin/env python
"""Build synthetic contrastive data and optionally mix sampled text corpora."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from embedding_ocr.training.embedding_dataset import (
    EmbeddingDatasetConfig,
    build_synthetic_dataset,
)
from embedding_ocr.training.fonts import (
    discover_font_families,
    split_font_families,
)
from embedding_ocr.training.real_text_crops import load_real_crops
from embedding_ocr.training.short_texts import (
    TextSample,
    dedupe_texts,
    make_synthetic_texts,
)

LOGGER = logging.getLogger("build_embedding_dataset")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fonts-dir", type=Path, default=Path("data/google_fonts/ofl"))
    parser.add_argument("--texts", type=Path, default=Path("data/short_texts.jsonl"))
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/embedding_dataset")
    )
    parser.add_argument("--text-image-rows", type=int, default=60_000)
    parser.add_argument("--image-image-rows", type=int, default=20_000)
    parser.add_argument("--synthetic-count", type=int, default=20_000)
    parser.add_argument("--real-crops", type=Path, default=None)
    parser.add_argument("--real-rows", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--holdout-fraction", type=float, default=0.15)
    parser.add_argument("--canvas-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-augment", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    families = discover_font_families(args.fonts_dir)
    train_fonts, heldout_fonts = split_font_families(
        families,
        holdout_fraction=args.holdout_fraction,
        seed=args.seed,
    )
    LOGGER.info(
        "Fonts: %d train families, %d held-out families",
        len(train_fonts),
        len(heldout_fonts),
    )

    source_texts = _load_source_texts(args.texts)
    synthetic = make_synthetic_texts(
        count=args.synthetic_count,
        seed=args.seed,
        lexicon=[sample.text for sample in source_texts] or None,
    )
    samples = dedupe_texts(
        [sample.text for sample in [*source_texts, *synthetic]],
        source="mixed",
    )
    if len(samples) < 10:
        raise SystemExit(
            "Not enough unique texts; provide --texts or increase --synthetic-count"
        )
    LOGGER.info(
        "Texts: %d unique (%d source, %d synthetic)",
        len(samples),
        len(source_texts),
        len(synthetic),
    )

    real_crops = load_real_crops(args.real_crops) if args.real_crops else []
    if args.real_rows and not real_crops:
        raise SystemExit("--real-rows requires --real-crops")
    LOGGER.info("Real crops: %d", len(real_crops))

    built = build_synthetic_dataset(
        samples,
        train_fonts,
        heldout_fonts,
        EmbeddingDatasetConfig(
            output_dir=args.output,
            text_image_rows=args.text_image_rows,
            image_image_rows=args.image_image_rows,
            real_rows=args.real_rows,
            val_fraction=args.val_fraction,
            canvas_size=args.canvas_size,
            augment=not args.no_augment,
            seed=args.seed,
        ),
        real_crops=real_crops,
    )
    LOGGER.info("Built %s rows at %s", built.counts, built.root)
    summary = built.root / "summary.json"
    print(json.dumps(json.loads(summary.read_text()), indent=2))


def _load_source_texts(path: Path) -> list[TextSample]:
    if not path.exists():
        LOGGER.warning("Text file not found: %s; using synthetic texts only", path)
        return []
    samples: list[TextSample] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = payload.get("text") if isinstance(payload, dict) else None
            if not isinstance(text, str):
                continue
            sample = TextSample.from_text(
                text, source=str(payload.get("source", "corpus"))
            )
            if sample is not None:
                samples.append(sample)
    return dedupe_texts((sample.text for sample in samples), source="corpus")


if __name__ == "__main__":
    main()
