"""Text-detection box metrics and annotation helpers."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

Box = tuple[int, int, int, int]


@dataclass(frozen=True)
class DetectionSample:
    """One annotated detection image."""

    name: str
    image_path: Path
    boxes: tuple[Box, ...]
    text: str = ""


@dataclass(frozen=True)
class DetectionMetrics:
    """Precision/recall statistics for one image or a whole set."""

    num_predictions: int
    num_references: int
    matched: int
    precision: float
    recall: float
    f1: float
    missed: int
    false_positives: int


def _clamp_box(box: Sequence[float]) -> Box:
    x, y, w, h = (float(value) for value in box)
    x = max(0.0, x)
    y = max(0.0, y)
    return int(round(x)), int(round(y)), max(0, int(round(w))), max(0, int(round(h)))


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    """Return intersection-over-union for two ``(x, y, w, h)`` boxes."""
    ax, ay, aw, ah = (float(value) for value in a)
    bx, by, bw, bh = (float(value) for value in b)
    left = max(ax, bx)
    top = max(ay, by)
    right = min(ax + aw, bx + bw)
    bottom = min(ay + ah, by + bh)
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0 else 0.0


def match_boxes(
    predictions: Sequence[Box],
    references: Sequence[Box],
    iou_threshold: float = 0.5,
) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    """Greedily match predicted boxes to reference boxes by descending IoU."""
    candidates: list[tuple[float, int, int]] = []
    for pred_index, prediction in enumerate(predictions):
        for ref_index, reference in enumerate(references):
            iou = box_iou(prediction, reference)
            if iou >= iou_threshold:
                candidates.append((iou, pred_index, ref_index))
    candidates.sort(reverse=True)

    used_predictions: set[int] = set()
    used_references: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for iou, pred_index, ref_index in candidates:
        if pred_index in used_predictions or ref_index in used_references:
            continue
        used_predictions.add(pred_index)
        used_references.add(ref_index)
        matches.append((pred_index, ref_index, iou))

    unmatched_predictions = [
        index for index in range(len(predictions)) if index not in used_predictions
    ]
    unmatched_references = [
        index for index in range(len(references)) if index not in used_references
    ]
    return matches, unmatched_predictions, unmatched_references


def detection_metrics(
    predictions: Sequence[Box],
    references: Sequence[Box],
    iou_threshold: float = 0.5,
) -> DetectionMetrics:
    """Compute precision, recall, F1, missed boxes, and false positives."""
    matches, unmatched_predictions, unmatched_references = match_boxes(
        predictions, references, iou_threshold=iou_threshold
    )
    matched = len(matches)
    precision = matched / len(predictions) if predictions else 0.0
    recall = matched / len(references) if references else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )
    return DetectionMetrics(
        num_predictions=len(predictions),
        num_references=len(references),
        matched=matched,
        precision=precision,
        recall=recall,
        f1=f1,
        missed=len(unmatched_references),
        false_positives=len(unmatched_predictions),
    )


def aggregate_metrics(metrics: Sequence[DetectionMetrics]) -> DetectionMetrics:
    """Aggregate per-image metrics by summing counts before dividing."""
    predictions = sum(item.num_predictions for item in metrics)
    references = sum(item.num_references for item in metrics)
    matched = sum(item.matched for item in metrics)
    precision = matched / predictions if predictions else 0.0
    recall = matched / references if references else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )
    return DetectionMetrics(
        num_predictions=predictions,
        num_references=references,
        matched=matched,
        precision=precision,
        recall=recall,
        f1=f1,
        missed=sum(item.missed for item in metrics),
        false_positives=sum(item.false_positives for item in metrics),
    )


def load_detection_annotations(path: Path) -> list[DetectionSample]:
    """Load a JSON annotation file.

    Expected shape::

        {
          "image.jpg": {
            "boxes": [[x, y, w, h], ...],
            "text": "optional transcription"
          }
        }
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        msg = "detection annotations must be a JSON object"
        raise ValueError(msg)

    base_dir = Path(path).parent
    samples: list[DetectionSample] = []
    for name, raw in payload.items():
        if not isinstance(raw, dict):
            continue
        raw_boxes = raw.get("boxes", [])
        if not isinstance(raw_boxes, list):
            continue
        boxes: list[Box] = []
        for raw_box in raw_boxes:
            if isinstance(raw_box, (list, tuple)) and len(raw_box) == 4:
                boxes.append(_clamp_box(raw_box))
        samples.append(
            DetectionSample(
                name=str(name),
                image_path=base_dir / str(name),
                boxes=tuple(boxes),
                text=str(raw.get("text", "")),
            )
        )
    return samples


def metrics_to_dict(metrics: DetectionMetrics) -> dict[str, Any]:
    """Return a JSON-serializable metrics dictionary."""
    return {
        "num_predictions": metrics.num_predictions,
        "num_references": metrics.num_references,
        "matched": metrics.matched,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "f1": metrics.f1,
        "missed": metrics.missed,
        "false_positives": metrics.false_positives,
    }
