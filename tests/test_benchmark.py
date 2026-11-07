"""Tests for the manifest-driven benchmark runner."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from embedding_ocr.eval.benchmark import read_manifest, run_benchmark, write_report
from embedding_ocr.pipeline.types import OCRResult


class DummyPipeline:
    def recognize(self, image: np.ndarray) -> OCRResult:
        return OCRResult(text="hello", confidence=0.9)


def test_manifest_round_trip_and_benchmark(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.png"
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(image_path)
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text("sample.png\thello\n")

    samples = read_manifest(manifest)
    assert len(samples) == 1
    assert samples[0].image_path == image_path
    assert samples[0].reference == "hello"

    report = run_benchmark(DummyPipeline(), samples)  # type: ignore[arg-type]
    assert report["metrics"]["cer"] == 0.0
    assert report["metrics"]["exact_match"] == 1.0
    assert report["samples"][0]["prediction"] == "hello"

    output = tmp_path / "report.json"
    write_report(report, output)
    saved = json.loads(output.read_text())
    assert saved["metrics"]["wer"] == 0.0
