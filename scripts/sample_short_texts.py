#!/usr/bin/env python
"""Sample short Latin strings from public Hugging Face text corpora."""

from __future__ import annotations

import argparse
import json
import logging
import random
from collections.abc import Iterable
from pathlib import Path

from embedding_ocr.training.short_texts import (
    TextSample,
    dedupe_texts,
    words_from_document,
)

LOGGER = logging.getLogger("sample_short_texts")

# Public dataset IDs verified against their Hugging Face dataset cards at
# implementation time. Re-check before a long run if a dataset is renamed.
SOURCES: dict[str, tuple[str, str | None, str, str]] = {
    "wikipedia": ("wikimedia/wikipedia", "20231101.en", "train", "text"),
    "fineweb-edu": ("HuggingFaceFW/fineweb-edu", "sample-10BT", "train", "text"),
    "c4": ("allenai/c4", "en", "train", "text"),
    "fineweb": ("HuggingFaceFW/fineweb", "sample-10BT", "train", "text"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        action="append",
        choices=sorted(SOURCES),
        default=None,
        help="Corpus to sample. Repeat for multiple sources.",
    )
    parser.add_argument("--per-source", type=int, default=10_000)
    parser.add_argument("--max-documents", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/short_texts.jsonl"),
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    if args.per_source < 1:
        raise SystemExit("--per-source must be positive")
    sources = args.source or ["wikipedia", "fineweb-edu", "c4", "fineweb"]
    rng = random.Random(args.seed)
    samples: list[TextSample] = []
    seen: set[str] = set()

    for source in sources:
        try:
            new_samples = list(
                _sample_source(source, args.per_source, rng, args.max_documents)
            )
        except Exception as exc:  # noqa: BLE001 - report the source and continue
            LOGGER.warning("Skipping %s: %s", source, exc)
            continue
        for sample in new_samples:
            if sample.text_id in seen:
                continue
            seen.add(sample.text_id)
            samples.append(sample)
        LOGGER.info("%s: %d samples (total %d)", source, len(new_samples), len(samples))

    if not samples:
        raise SystemExit("No texts sampled; check network access and dataset IDs.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(
                json.dumps(
                    {
                        "id": sample.text_id,
                        "text": sample.text,
                        "source": sample.source,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    LOGGER.info("Wrote %d texts to %s", len(samples), args.output)


def _sample_source(
    source: str,
    count: int,
    rng: random.Random,
    max_documents: int,
) -> Iterable[TextSample]:
    dataset_id, config, split, text_key = SOURCES[source]
    from datasets import load_dataset

    dataset = load_dataset(dataset_id, config, split=split, streaming=True)
    collected: list[TextSample] = []
    for index, row in enumerate(dataset):
        if index >= max_documents or len(collected) >= count:
            break
        text = row.get(text_key)
        if not isinstance(text, str):
            continue
        candidate = words_from_document(text, rng=rng)
        if candidate is None:
            continue
        sample = TextSample.from_text(candidate, source=source)
        if sample is not None:
            collected.append(sample)
    return dedupe_texts(
        (sample.text for sample in collected),
        limit=count,
        source=source,
    )


if __name__ == "__main__":
    main()
