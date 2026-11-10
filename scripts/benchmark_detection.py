#!/usr/bin/env python3
"""Benchmark text detection boxes on annotated or synthetic images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from rich.console import Console
from rich.table import Table

from embedding_ocr.config import AppConfig, DetectorSettings, load_config
from embedding_ocr.eval.detection import (
    DetectionMetrics,
    DetectionSample,
    aggregate_metrics,
    detection_metrics,
    load_detection_annotations,
    metrics_to_dict,
)
from embedding_ocr.pipeline.detect import build_detector

_CONSOLE = Console()
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _find_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        "/usr/share/fonts/liberation-sans-fonts/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def make_synthetic_sample(output_dir: Path) -> DetectionSample:
    """Render a two-line page and return line-level reference boxes."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "synthetic_two_lines.png"
    image = Image.new("RGB", (1000, 360), "white")
    draw = ImageDraw.Draw(image)
    font = _find_font(36)
    lines = ("First line alpha beta", "Second line gamma delta")
    boxes: list[tuple[int, int, int, int]] = []
    for index, text in enumerate(lines):
        x, y = 30, 30 + index * 120
        draw.text((x, y), text, font=font, fill="black")
        left, top, right, bottom = draw.textbbox((x, y), text, font=font)
        boxes.append((left, top, right - left, bottom - top))
    image.save(path)
    return DetectionSample(
        name=path.name,
        image_path=path,
        boxes=tuple(boxes),
        text=" ".join(lines),
    )


def load_samples(images_dir: Path, annotations: Path | None) -> list[DetectionSample]:
    if annotations is not None and annotations.exists():
        return load_detection_annotations(annotations)
    samples: list[DetectionSample] = []
    for path in sorted(images_dir.iterdir()):
        if path.suffix.lower() not in _IMAGE_SUFFIXES:
            continue
        samples.append(DetectionSample(name=path.name, image_path=path, boxes=()))
    return samples


def draw_overlay(
    image: np.ndarray,
    predictions: list[tuple[int, int, int, int]],
    references: list[tuple[int, int, int, int]],
    output_path: Path,
) -> None:
    canvas = Image.fromarray(image).convert("RGB")
    draw = ImageDraw.Draw(canvas)
    for x, y, width, height in references:
        draw.rectangle((x, y, x + width, y + height), outline=(0, 180, 0), width=3)
    for x, y, width, height in predictions:
        draw.rectangle((x, y, x + width, y + height), outline=(220, 0, 0), width=3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def _enabled_backends(settings: DetectorSettings, all_backends: bool) -> list[str]:
    if all_backends:
        return ["auto", "paddle", "craft", "lines", "visual", "full"]
    return [settings.backend]


def evaluate_backend(
    backend: str,
    samples: list[DetectionSample],
    base_settings: DetectorSettings,
    device: str,
    model_name: str | None,
    iou_threshold: float,
    output_dir: Path,
) -> tuple[dict[str, object], list[DetectionMetrics]]:
    overrides: dict[str, object] = {"backend": backend, "device": device}
    if model_name is not None:
        overrides["paddle_model"] = model_name
    settings = base_settings.model_copy(update=overrides)
    detector = build_detector(settings)
    resolved = getattr(detector, "resolved_name", backend)
    per_image: list[DetectionMetrics] = []
    details: list[dict[str, object]] = []
    for sample in samples:
        image = np.array(Image.open(sample.image_path).convert("RGB"))
        predictions = [region.bbox for region in detector.detect(image)]
        current_resolved = getattr(detector, "resolved_name", resolved)
        metrics = detection_metrics(
            predictions, sample.boxes, iou_threshold=iou_threshold
        )
        if sample.boxes:
            per_image.append(metrics)
        details.append(
            {
                "name": sample.name,
                "predictions": [list(box) for box in predictions],
                "references": [list(box) for box in sample.boxes],
                "metrics": metrics_to_dict(metrics),
            }
        )
        draw_overlay(
            image,
            predictions,
            list(sample.boxes),
            output_dir
            / "overlays"
            / f"{Path(sample.name).stem}_{current_resolved}.png",
        )
    aggregate = aggregate_metrics(per_image)
    resolved = getattr(detector, "resolved_name", resolved)
    return {
        "backend": backend,
        "resolved_backend": resolved,
        "aggregate": metrics_to_dict(aggregate),
        "images": details,
    }, per_image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--images", type=Path, default=Path("tests/images"))
    parser.add_argument("--annotations", type=Path, default=None)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--backend", default=None)
    parser.add_argument("--all", action="store_true", help="Compare every backend.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--model", default=None)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/detection_bench")
    )
    args = parser.parse_args()

    config: AppConfig = load_config(args.config)
    settings = config.detector
    if args.backend is not None:
        settings = settings.model_copy(update={"backend": args.backend})

    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = load_samples(args.images, args.annotations)
    if args.synthetic:
        samples = [make_synthetic_sample(output_dir / "synthetic"), *samples]
    if not samples:
        _CONSOLE.print(f"[red]No images found in {args.images}[/red]")
        return

    table = Table(title="Detection benchmark")
    table.add_column("backend")
    table.add_column("resolved")
    table.add_column("pred", justify="right")
    table.add_column("ref", justify="right")
    table.add_column("matched", justify="right")
    table.add_column("precision", justify="right")
    table.add_column("recall", justify="right")
    table.add_column("f1", justify="right")

    report: dict[str, object] = {"samples": len(samples), "backends": []}
    backends = _enabled_backends(settings, args.all)
    backend_reports: list[dict[str, object]] = []
    for backend in backends:
        try:
            payload, per_image = evaluate_backend(
                backend,
                samples,
                settings,
                args.device,
                args.model,
                args.iou,
                output_dir,
            )
        except Exception as exc:
            _CONSOLE.print(f"[yellow]{backend} unavailable:[/yellow] {exc}")
            continue
        backend_reports.append(payload)
        aggregate = aggregate_metrics(per_image)
        table.add_row(
            backend,
            str(payload["resolved_backend"]),
            str(aggregate.num_predictions),
            str(aggregate.num_references),
            str(aggregate.matched),
            f"{aggregate.precision:.3f}",
            f"{aggregate.recall:.3f}",
            f"{aggregate.f1:.3f}",
        )
    report["backends"] = backend_reports
    (output_dir / "detection_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _CONSOLE.print(table)
    _CONSOLE.print(f"Report: {output_dir / 'detection_report.json'}")
    _CONSOLE.print(f"Overlays: {output_dir / 'overlays'}")


if __name__ == "__main__":
    main()
