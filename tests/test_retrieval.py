"""Tests for retrieval metrics and case-pair reporting."""

from __future__ import annotations

import numpy as np
import pytest

from embedding_ocr.eval.retrieval import (
    case_pair_report,
    predictions_from_retrieval,
    retrieval_metrics,
)


def test_retrieval_metrics_perfect_match() -> None:
    vectors = np.eye(4, dtype=np.float32)
    metrics = retrieval_metrics(
        vectors,
        vectors,
        ["a", "b", "c", "d"],
        ["a", "b", "c", "d"],
    )
    assert metrics["top_1"] == 1.0
    assert metrics["top_5"] == 1.0
    assert metrics["mrr"] == 1.0
    assert metrics["num_queries"] == 4


def test_retrieval_metrics_handles_duplicate_labels() -> None:
    queries = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    targets = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    metrics = retrieval_metrics(queries, targets, ["a", "b"], ["a", "a"])
    assert metrics["top_1"] == 0.5


def test_predictions_from_retrieval() -> None:
    queries = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    targets = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    assert predictions_from_retrieval(queries, targets, ["a", "b"]) == ["a", "b"]


def test_retrieval_metrics_validates_shapes() -> None:
    with pytest.raises(ValueError, match="dimensions"):
        retrieval_metrics(
            np.ones((1, 2)),
            np.ones((1, 3)),
            ["a"],
            ["a"],
        )
    with pytest.raises(ValueError, match="Label counts"):
        retrieval_metrics(np.ones((1, 2)), np.ones((1, 2)), [], ["a"])


def test_case_pair_report_counts_confusions() -> None:
    report = case_pair_report(["c", "C", "o", "O"], ["C", "C", "O", "O"])
    assert report["c/C"]["accuracy"] == 0.5
    assert report["c/C"]["case_confusions"] == 1
    assert report["o/O"]["accuracy"] == 0.5
