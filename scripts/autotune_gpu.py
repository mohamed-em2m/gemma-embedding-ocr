#!/usr/bin/env python
"""Benchmark the best GPU and write a safe batch-size recommendation.

Example:
    uv run --no-sync python scripts/autotune_gpu.py --device auto

The result is cached in ``artifacts/gpu_tuning.json``.  ``EmbeddingGemma2``
reads that cache when its ``batch_size`` is ``0`` (the default).
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from embedding_ocr.config import load_config
from embedding_ocr.recognition.embedding.embedder import EmbeddingGemma2
from embedding_ocr.utils.gpu import (
    DEFAULT_TUNING_PATH,
    autotune_batch_size,
    describe_gpus,
    query_gpus,
    resolve_device,
    save_tuning_cache,
    select_best_gpu,
)


def _parse_candidates(value: str) -> tuple[int, ...]:
    candidates = tuple(int(part) for part in value.split(",") if part.strip())
    if not candidates:
        msg = "--candidates must contain at least one positive integer"
        raise argparse.ArgumentTypeError(msg)
    return candidates


def _dummy_images(count: int) -> list[np.ndarray]:
    images: list[np.ndarray] = []
    for index in range(count):
        canvas = Image.new("RGB", (64, 64), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((16, 12), chr(65 + index % 26), fill="black")
        images.append(np.asarray(canvas, dtype=np.uint8))
    return images


def _dummy_texts(count: int) -> list[str]:
    return [f"sample word {index}" for index in range(count)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--device", default="auto", help="auto | cuda | cuda:N | cpu")
    parser.add_argument("--dim", type=int, default=None, help="Embedding dimension")
    parser.add_argument("--max-batch", type=int, default=64)
    parser.add_argument(
        "--candidates",
        type=_parse_candidates,
        default=_parse_candidates("1,2,4,8,12,16,24,32,48,64"),
    )
    parser.add_argument("--min-free-mb", type=float, default=1024)
    parser.add_argument("--output", type=Path, default=DEFAULT_TUNING_PATH)
    parser.add_argument("--skip-images", action="store_true")
    parser.add_argument("--skip-text", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    dim = args.dim or config.embedder.dim

    gpus = query_gpus()
    print(f"Visible GPUs: {describe_gpus(gpus)}")
    device = resolve_device(args.device, min_free_mb=args.min_free_mb)
    print(f"Selected device: {device}")
    if not device.startswith("cuda"):
        print("No CUDA device selected; batch tuning is only needed on GPU.")
        return 0

    selected_gpu = next(
        (gpu for gpu in gpus if device == f"cuda:{gpu.index}"),
        select_best_gpu(gpus, min_free_mb=args.min_free_mb),
    )
    if selected_gpu is not None:
        print(
            f"GPU {selected_gpu.index}: {selected_gpu.name}, "
            f"{selected_gpu.free_mb:.0f} MiB free, "
            f"util={selected_gpu.utilization if selected_gpu.utilization is not None else 'n/a'}"
        )

    print(f"Loading {config.embedder.model_id or 'google/embeddinggemma-2'}...")
    embedder = EmbeddingGemma2(
        dim=dim,
        device=device,
        batch_size=1,
        auto_oom_fallback=False,
    )

    candidates = tuple(size for size in args.candidates if size <= args.max_batch)
    if not candidates:
        print("ERROR: no candidate batch sizes are <= --max-batch.")
        return 2
    image_result = None
    text_result = None

    if not args.skip_images:
        images = _dummy_images(max(candidates))

        def encode_images(size: int) -> None:
            embedder.batch_size = size
            embedder.embed_images(images[:size], use_cache=False)

        print("Tuning image batch size...")
        image_result = autotune_batch_size(
            encode_images,
            device=device,
            candidates=candidates,
            max_batch=args.max_batch,
        )
        print(
            f"  image batch={image_result.batch_size} "
            f"peak={image_result.peak_mb:.0f}MiB "
            f"latency={image_result.latency_ms:.0f}ms"
        )

    if not args.skip_text:
        texts = _dummy_texts(max(candidates))

        def encode_texts(size: int) -> None:
            embedder.batch_size = size
            embedder.embed_texts(texts[:size])

        print("Tuning text batch size...")
        text_result = autotune_batch_size(
            encode_texts,
            device=device,
            candidates=candidates,
            max_batch=args.max_batch,
        )
        print(
            f"  text batch={text_result.batch_size} "
            f"peak={text_result.peak_mb:.0f}MiB "
            f"latency={text_result.latency_ms:.0f}ms"
        )

    sizes = [
        result.batch_size
        for result in (image_result, text_result)
        if result is not None
    ]
    if not sizes:
        print("Nothing tuned; pass without --skip-images/--skip-text.")
        return 1

    payload = {
        "version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "model_id": embedder.name,
        "dim": dim,
        "device": device,
        "gpu_count": len(gpus),
        "gpu": selected_gpu.as_dict() if selected_gpu is not None else None,
        "batch_size": min(sizes),
        "candidates": list(candidates),
        "image": image_result.as_dict() if image_result is not None else None,
        "text": text_result.as_dict() if text_result is not None else None,
    }
    save_tuning_cache(args.output, payload)
    print(f"Saved recommendation: batch_size={payload['batch_size']} -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
