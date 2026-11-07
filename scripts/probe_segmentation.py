"""Probe word detectors and embedders for multilingual segmentation.

Modes:
  detectors  Run optional PaddleOCR / Surya / CRAFT / EasyOCR detectors on
             Arabic line images and report box counts and latency.
  embedders  Compare EmbeddingGemma 2, SigLIP and Jina v4 on Arabic glyph
             retrieval (image->text and image->image across fonts).
  all        Run both.

The script never hard-fails when an optional dependency is missing; the error
is recorded in the JSON report so the probe can run incrementally.

Usage:
  uv run python scripts/probe_segmentation.py --mode all --dry-run
  uv run python scripts/probe_segmentation.py --mode embedders \
      --embedders gemma siglip jina --out docs/segmentation_probe.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from embedding_ocr.data.glyph_renderer import ARABIC_CLASSES
from embedding_ocr.recognition.embedding.hierarchical.vocabulary import (
    DEFAULT_ARABIC_FONTS,
    render_prototype_image,
)
from embedding_ocr.recognition.embedding.index import NumpyIndex


def _existing_arabic_fonts() -> list[str]:
    return [font for font in DEFAULT_ARABIC_FONTS if Path(font).exists()]


def _load_images(images_dir: Path | None) -> list[tuple[str, np.ndarray]]:
    if images_dir is None:
        fonts = _existing_arabic_fonts()
        if not fonts:
            return []
        image = render_prototype_image(
            "مرحبا بالعالم",
            font_path=fonts[0],
            font_size=36,
            script="arabic",
        )
        return [("synthetic_arabic_line", image)]

    images: list[tuple[str, np.ndarray]] = []
    for pattern in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"):
        for path in sorted(images_dir.glob(pattern)):
            from PIL import Image

            images.append((path.name, np.array(Image.open(path).convert("RGB"))))
    return images


def _run_craft(image: np.ndarray) -> dict[str, Any]:
    from embedding_ocr.recognition.embedding.hierarchical.craft_detector import (
        CraftWordDetector,
    )

    detector = CraftWordDetector(cuda=False)
    boxes = detector.detect_word_boxes(image, rtl=True)
    return {"boxes": boxes}


def _run_easyocr(image: np.ndarray) -> dict[str, Any]:
    import torch
    from easyocr import Reader  # type: ignore[import-untyped]

    reader = Reader(["ar", "en"], gpu=torch.cuda.is_available())
    results = reader.readtext(image, detail=1, paragraph=False)
    boxes = [result[0] for result in results]
    return {"boxes": boxes}


def _run_surya(image: np.ndarray) -> dict[str, Any]:
    from PIL import Image
    from surya.detection import DetectionPredictor  # type: ignore[import-untyped]

    predictor = DetectionPredictor()
    predictions = predictor([Image.fromarray(image)])
    boxes = [
        [list(map(float, bbox)) for bbox in getattr(pred, "bboxes", [])]
        for pred in predictions
    ]
    return {"boxes": boxes}


def _run_paddle(image: np.ndarray) -> dict[str, Any]:
    from paddleocr import PaddleOCR  # type: ignore[import-untyped]

    ocr = PaddleOCR(
        lang="ar",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )
    results = ocr.predict(image) if hasattr(ocr, "predict") else [ocr.ocr(image)]
    boxes: list[Any] = []
    for result in results:
        if isinstance(result, dict):
            boxes.extend(result.get("dt_polys", result.get("rec_polys", [])))
    return {"boxes": boxes}


DETECTORS: dict[str, Any] = {
    "paddle": _run_paddle,
    "surya": _run_surya,
    "craft": _run_craft,
    "easyocr": _run_easyocr,
}


def probe_detectors(
    images: list[tuple[str, np.ndarray]], detector_names: list[str]
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for name in detector_names:
        fn = DETECTORS[name]
        rows: list[dict[str, Any]] = []
        for image_name, image in images:
            started = time.perf_counter()
            try:
                result = fn(image)
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                rows.append(
                    {
                        "image": image_name,
                        "num_boxes": len(result.get("boxes", [])),
                        "latency_ms": elapsed_ms,
                    }
                )
            except Exception as exc:  # noqa: BLE001 - probe reports all failures
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                rows.append(
                    {
                        "image": image_name,
                        "error": f"{type(exc).__name__}: {exc}",
                        "latency_ms": elapsed_ms,
                    }
                )
        report[name] = rows
    return report


def _build_embedder(name: str, dim: int | None, device: str) -> Any:
    if name == "gemma":
        from embedding_ocr.recognition.embedding.embedder import EmbeddingGemma2

        return EmbeddingGemma2(dim=dim or 768, device=device)
    if name == "siglip":
        from embedding_ocr.recognition.embedding.embedder import SiglipEmbedder

        return SiglipEmbedder(dim=dim or 1152, device=device)
    if name == "jina":
        from embedding_ocr.recognition.embedding.embedder import JinaEmbeddingsV4

        return JinaEmbeddingsV4(dim=dim or 2048, device=device)
    msg = f"unknown embedder {name!r}"
    raise ValueError(msg)


def _retrieval_accuracy(
    index: NumpyIndex,
    labels: list[str],
    query_vectors: np.ndarray,
    query_labels: list[str],
    top_k: int,
) -> dict[str, float]:
    sims, ids = index.search(query_vectors, k=min(top_k, len(labels)))
    top1 = 0
    top5 = 0
    for row_ids, expected in zip(ids, query_labels, strict=False):
        retrieved = [labels[int(idx)] for idx in row_ids]
        if retrieved and retrieved[0] == expected:
            top1 += 1
        if expected in retrieved:
            top5 += 1
    total = max(1, len(query_labels))
    return {"top1": top1 / total, "top5": top5 / total}


def _render_classes(
    classes: list[str], font: str, font_size: int = 36
) -> list[np.ndarray]:
    return [
        render_prototype_image(
            char, font_path=font, font_size=font_size, script="arabic"
        )
        for char in classes
    ]


def probe_embedders(
    embedder_names: list[str],
    dim: int | None,
    device: str,
    fonts: list[str],
    top_k: int,
) -> dict[str, Any]:
    classes = list(ARABIC_CLASSES)
    report: dict[str, Any] = {}
    if not fonts:
        return {"error": "no Arabic fonts installed"}

    for name in embedder_names:
        started = time.perf_counter()
        try:
            embedder = _build_embedder(name, dim, device)
            text_vectors = embedder.embed_texts(classes)

            text_index = NumpyIndex(dim=embedder.dim)
            text_index.build(text_vectors, np.arange(len(classes)))

            query_images = _render_classes(classes, fonts[0])
            query_vectors = embedder.embed_images(query_images)
            image_to_text = _retrieval_accuracy(
                text_index, classes, query_vectors, classes, top_k
            )

            image_to_image: dict[str, float] | None = None
            if len(fonts) >= 2:
                index_images = _render_classes(classes, fonts[0])
                query_images_b = _render_classes(classes, fonts[1])
                index_vectors = embedder.embed_images(index_images)
                query_vectors_b = embedder.embed_images(query_images_b)
                image_index = NumpyIndex(dim=embedder.dim)
                image_index.build(index_vectors, np.arange(len(classes)))
                image_to_image = _retrieval_accuracy(
                    image_index, classes, query_vectors_b, classes, top_k
                )

            report[name] = {
                "model": embedder.name,
                "dim": embedder.dim,
                "image_to_text": image_to_text,
                "image_to_image": image_to_image,
                "latency_ms": (time.perf_counter() - started) * 1000.0,
            }
        except Exception as exc:  # noqa: BLE001 - probe reports all failures
            report[name] = {
                "error": f"{type(exc).__name__}: {exc}",
                "latency_ms": (time.perf_counter() - started) * 1000.0,
            }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=["detectors", "embedders", "all"], default="all"
    )
    parser.add_argument(
        "--detectors",
        nargs="+",
        default=["paddle", "surya", "craft", "easyocr"],
        choices=sorted(DETECTORS),
    )
    parser.add_argument(
        "--embedders",
        nargs="+",
        default=["gemma", "siglip", "jina"],
        choices=["gemma", "siglip", "jina"],
    )
    parser.add_argument("--images-dir", type=Path, default=None)
    parser.add_argument("--fonts", nargs="+", default=None)
    parser.add_argument("--dim", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--out", type=Path, default=Path("docs/segmentation_probe.json")
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    fonts = args.fonts or _existing_arabic_fonts()
    report: dict[str, Any] = {
        "mode": args.mode,
        "fonts": fonts,
        "dry_run": args.dry_run,
    }

    if args.dry_run:
        report["plan"] = {
            "detectors": args.detectors if args.mode in {"detectors", "all"} else [],
            "embedders": args.embedders if args.mode in {"embedders", "all"} else [],
            "images_dir": str(args.images_dir) if args.images_dir else "synthetic",
        }
    else:
        if args.mode in {"detectors", "all"}:
            images = _load_images(args.images_dir)
            report["detectors"] = probe_detectors(images, args.detectors)
        if args.mode in {"embedders", "all"}:
            report["embedders"] = probe_embedders(
                args.embedders, args.dim, args.device, fonts, args.top_k
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
