"""Evaluation metrics and benchmark helpers."""

from embedding_ocr.eval.benchmark import (
    BenchmarkSample,
    read_manifest,
    run_benchmark,
    write_report,
)
from embedding_ocr.eval.metrics import (
    character_error_rate,
    evaluate,
    exact_match,
    normalize_text,
    word_error_rate,
)

__all__ = [
    "BenchmarkSample",
    "character_error_rate",
    "evaluate",
    "exact_match",
    "normalize_text",
    "read_manifest",
    "run_benchmark",
    "word_error_rate",
    "write_report",
]
