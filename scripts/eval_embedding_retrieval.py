#!/usr/bin/env python
"""Evaluate text-image and image-image retrieval on held-out fonts."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from PIL import Image

from embedding_ocr.eval.retrieval import (
    case_pair_report,
    predictions_from_retrieval,
    retrieval_metrics,
)
from embedding_ocr.recognition.embedding.embedder import EmbeddingGemma2
from embedding_ocr.training.embedding_dataset import load_manifest

LOGGER = logging.getLogger("eval_embedding_retrieval")
MODES = ("text_image", "image_image")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="google/embeddinggemma-2")
    parser.add_argument(
        "--dataset", type=Path, default=Path("artifacts/embedding_dataset")
    )
    parser.add_argument("--split", default="val")
    parser.add_argument("--dim", type=int, default=768)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--prompt-name", default="Document")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/retrieval_metrics.json")
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    embedder = EmbeddingGemma2(
        dim=args.dim,
        device=args.device,
        dtype=args.dtype,
        model_id=args.model_id,
        prompt_name=args.prompt_name,
    )
    report: dict[str, object] = {"model_id": args.model_id, "dim": args.dim}
    for mode in MODES:
        path = args.dataset / args.split / f"{mode}.jsonl"
        if not path.exists():
            LOGGER.warning("Skipping missing manifest %s", path)
            continue
        rows = _unique_rows(load_manifest(path), limit=args.limit)
        if len(rows) < 2:
            LOGGER.warning("Not enough rows in %s", path)
            continue
        if mode == "text_image":
            queries = embedder.embed_texts([str(row["text"]) for row in rows])
            targets = embedder.embed_images(
                [
                    np.asarray(
                        Image.open(str(row["positive"])).convert("RGB"), dtype=np.uint8
                    )
                    for row in rows
                ]
            )
        else:
            queries = embedder.embed_images(
                [
                    np.asarray(
                        Image.open(str(row["anchor_path"])).convert("RGB"),
                        dtype=np.uint8,
                    )
                    for row in rows
                ]
            )
            targets = embedder.embed_images(
                [
                    np.asarray(
                        Image.open(str(row["positive"])).convert("RGB"), dtype=np.uint8
                    )
                    for row in rows
                ]
            )
        labels = [str(row["text"]) for row in rows]
        metrics = retrieval_metrics(queries, targets, labels, labels)
        predictions = predictions_from_retrieval(
            queries, np.asarray(targets, dtype=np.float32), labels
        )
        report[mode] = {
            "metrics": metrics,
            "case_pairs": case_pair_report(labels, predictions),
        }
        LOGGER.info("%s: %s", mode, metrics)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    LOGGER.info("Wrote %s", args.output)


def _unique_rows(
    rows: list[dict[str, object]], *, limit: int
) -> list[dict[str, object]]:
    seen: set[str] = set()
    unique: list[dict[str, object]] = []
    for row in rows:
        text_id = str(row.get("text_id", ""))
        if text_id in seen:
            continue
        seen.add(text_id)
        unique.append(row)
        if len(unique) >= limit:
            break
    return unique


if __name__ == "__main__":
    main()
