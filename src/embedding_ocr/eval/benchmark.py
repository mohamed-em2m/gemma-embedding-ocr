"""Simple manifest-driven OCR benchmark runner."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from embedding_ocr.eval.metrics import evaluate
from embedding_ocr.pipeline.pipeline import OCRPipeline


@dataclass(frozen=True)
class BenchmarkSample:
    """One image/reference pair."""

    image_path: Path
    reference: str


def read_manifest(path: str | Path) -> list[BenchmarkSample]:
    """Read a TSV/CSV manifest of ``image_path<TAB>reference`` lines.

    Blank lines and ``#`` comments are ignored. Relative image paths are
    resolved against the manifest directory.
    """
    manifest = Path(path)
    base = manifest.parent
    samples: list[BenchmarkSample] = []
    for line in manifest.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "\t" in stripped:
            image_part, reference = stripped.split("\t", 1)
        elif "," in stripped:
            image_part, reference = stripped.split(",", 1)
        else:
            msg = f"manifest line must contain a tab or comma: {line!r}"
            raise ValueError(msg)
        image_path = Path(image_part.strip())
        if not image_path.is_absolute():
            image_path = base / image_path
        samples.append(
            BenchmarkSample(image_path=image_path, reference=reference.strip())
        )
    return samples


def _load_image(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def run_benchmark(
    pipeline: OCRPipeline,
    samples: list[BenchmarkSample],
) -> dict[str, object]:
    """Run a pipeline over samples and return metrics + per-sample results."""
    references: list[str] = []
    hypotheses: list[str] = []
    rows: list[dict[str, object]] = []
    latencies: list[float] = []

    for sample in samples:
        image = _load_image(sample.image_path)
        started = time.perf_counter()
        result = pipeline.recognize(image)
        latency_ms = (time.perf_counter() - started) * 1000.0
        latencies.append(latency_ms)
        references.append(sample.reference)
        hypotheses.append(result.text)
        rows.append(
            {
                "image": str(sample.image_path),
                "reference": sample.reference,
                "prediction": result.text,
                "confidence": result.confidence,
                "latency_ms": latency_ms,
            }
        )

    metrics = evaluate(references, hypotheses)
    metrics["latency_ms_mean"] = float(np.mean(latencies)) if latencies else 0.0
    metrics["latency_ms_p95"] = (
        float(np.percentile(latencies, 95)) if latencies else 0.0
    )
    return {"metrics": metrics, "samples": rows}


def write_report(report: dict[str, object], path: str | Path) -> None:
    """Write a benchmark report as JSON."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False))
