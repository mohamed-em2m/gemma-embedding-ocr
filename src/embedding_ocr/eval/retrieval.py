"""Retrieval accuracy and case-pair diagnostics for embedding evaluation."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from embedding_ocr.data.glyph_renderer import CASE_PAIRS


def retrieval_metrics(
    query_vectors: np.ndarray,
    target_vectors: np.ndarray,
    query_labels: Sequence[str],
    target_labels: Sequence[str],
    *,
    ks: Sequence[int] = (1, 5),
) -> dict[str, float | int | list[float]]:
    """Compute top-k retrieval accuracy and mean reciprocal rank.

    Vectors are L2-normalized internally; cosine similarity is used.
    """
    queries = _normalize(query_vectors)
    targets = _normalize(target_vectors)
    if queries.shape[1] != targets.shape[1]:
        msg = "Query and target dimensions must match."
        raise ValueError(msg)
    if len(query_labels) != len(queries) or len(target_labels) != len(targets):
        msg = "Label counts must match vector counts."
        raise ValueError(msg)
    if not len(targets):
        msg = "At least one target vector is required."
        raise ValueError(msg)

    similarities = queries @ targets.T
    order = np.argsort(-similarities, axis=1)
    metrics: dict[str, float | int | list[float]] = {}
    for k in ks:
        top = order[:, : min(k, order.shape[1])]
        hits = [
            any(target_labels[index] == label for index in row)
            for row, label in zip(top, query_labels, strict=True)
        ]
        metrics[f"top_{k}"] = float(np.mean(hits)) if hits else 0.0

    reciprocal_ranks: list[float] = []
    for row, label in zip(order, query_labels, strict=True):
        for rank, index in enumerate(row, start=1):
            if target_labels[index] == label:
                reciprocal_ranks.append(1.0 / rank)
                break
        else:
            reciprocal_ranks.append(0.0)
    metrics["mrr"] = float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0
    metrics["num_queries"] = len(queries)
    return metrics


def predictions_from_retrieval(
    query_vectors: np.ndarray,
    target_vectors: np.ndarray,
    target_labels: Sequence[str],
) -> list[str]:
    """Return the top-1 target label for each query."""
    queries = _normalize(query_vectors)
    targets = _normalize(target_vectors)
    similarities = queries @ targets.T
    indices = np.argmax(similarities, axis=1)
    return [target_labels[int(index)] for index in indices]


def case_pair_report(
    labels: Sequence[str],
    predictions: Sequence[str],
) -> dict[str, dict[str, float | int]]:
    """Report accuracy and case confusion for shape-identical letter pairs."""
    if len(labels) != len(predictions):
        msg = "Labels and predictions must have equal length."
        raise ValueError(msg)
    report: dict[str, dict[str, float | int]] = {}
    for lower, upper in CASE_PAIRS:
        pair = {lower, upper}
        indices = [index for index, label in enumerate(labels) if label in pair]
        if not indices:
            continue
        correct = sum(predictions[index] == labels[index] for index in indices)
        confused = sum(
            predictions[index] in pair and predictions[index] != labels[index]
            for index in indices
        )
        report[f"{lower}/{upper}"] = {
            "count": len(indices),
            "accuracy": correct / len(indices),
            "case_confusions": confused,
        }
    return report


def _normalize(vectors: np.ndarray) -> np.ndarray:
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim == 1:
        array = array[None, :]
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.maximum(norms, 1e-12)
