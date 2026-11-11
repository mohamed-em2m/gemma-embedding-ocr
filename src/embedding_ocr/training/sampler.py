"""Batch sampling and collation helpers for OCR contrastive training."""

from __future__ import annotations

import random
from collections.abc import Iterator
from typing import Any

from sentence_transformers.base.data_collator import BaseDataCollator
from sentence_transformers.base.sampler import DefaultBatchSampler
from sentence_transformers.sentence_transformer.data_collator import (
    SentenceTransformerDataCollator,
)

METADATA_COLUMNS = frozenset(
    {
        "id",
        "text",
        "text_id",
        "negative_text",
        "mode",
        "font",
        "anchor_path",
        "source",
    }
)


class UniqueTextBatchSampler(DefaultBatchSampler):
    """Batch sampler that allows each text label at most once per batch.

    Sentence Transformers' duplicate samplers hash all columns, so the same
    text rendered in different fonts is not recognized as a duplicate. This
    sampler keys on ``text_id`` to avoid false negatives in MNRL.
    """

    def __init__(
        self,
        dataset: Any,
        batch_size: int,
        drop_last: bool,
        valid_label_columns: list[str] | None = None,
        generator: Any = None,
        seed: int = 0,
    ) -> None:
        super().__init__(
            dataset,
            batch_size=batch_size,
            drop_last=drop_last,
            valid_label_columns=valid_label_columns,
            generator=generator,
            seed=seed,
        )
        self._text_ids = list(dataset["text_id"])

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        indices = list(range(len(self._text_ids)))
        rng.shuffle(indices)
        batch: list[int] = []
        seen: set[str] = set()
        for index in indices:
            text_id = str(self._text_ids[index])
            if text_id in seen:
                continue
            batch.append(index)
            seen.add(text_id)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
                seen = set()
        if batch and not self.drop_last:
            yield batch

    def __len__(self) -> int:
        return max(1, (len(self._text_ids) + self.batch_size - 1) // self.batch_size)


class OcrDataCollator(SentenceTransformerDataCollator):
    """Collator that drops manifest metadata before tokenization."""

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        cleaned = [
            {
                key: value
                for key, value in feature.items()
                if key not in METADATA_COLUMNS
            }
            for feature in features
        ]
        return super().__call__(cleaned)


def split_sentence_features(
    batch: dict[str, Any],
    columns: tuple[str, ...] = ("anchor", "positive", "negative"),
) -> list[dict[str, Any]]:
    """Split a collated batch into the per-column dicts expected by losses."""
    features: list[dict[str, Any]] = []
    for column in columns:
        prefix = f"{column}_"
        feature = {
            key[len(prefix) :]: value
            for key, value in batch.items()
            if key.startswith(prefix)
        }
        if not feature:
            continue
        task_key = f"{column}_task"
        if task_key in batch:
            feature["task"] = batch[task_key]
        features.append(feature)
    return features


__all__ = [
    "BaseDataCollator",
    "OcrDataCollator",
    "UniqueTextBatchSampler",
    "split_sentence_features",
]
