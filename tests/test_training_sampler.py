"""Tests for text-unique batch sampling and collator metadata filtering."""

from __future__ import annotations

import torch
from datasets import Dataset

from embedding_ocr.training.sampler import (
    OcrDataCollator,
    UniqueTextBatchSampler,
    split_sentence_features,
)


def test_unique_text_sampler_has_no_duplicate_texts() -> None:
    dataset = Dataset.from_dict(
        {
            "text_id": ["a", "a", "b", "c", "c", "d"],
            "value": [1, 2, 3, 4, 5, 6],
        }
    )
    sampler = UniqueTextBatchSampler(dataset, batch_size=4, drop_last=False, seed=0)
    batches = list(sampler)
    assert batches
    for batch in batches:
        ids = [dataset[index]["text_id"] for index in batch]
        assert len(ids) == len(set(ids))
    assert len(sampler) >= 1


def test_unique_text_sampler_respects_drop_last() -> None:
    dataset = Dataset.from_dict({"text_id": ["a", "a", "a"], "value": [1, 2, 3]})
    sampler = UniqueTextBatchSampler(dataset, batch_size=4, drop_last=True)
    assert list(sampler) == []


def test_ocr_collator_removes_metadata() -> None:
    collator = OcrDataCollator(preprocess_fn=_fake_preprocess)
    features = [
        {
            "anchor": "hello",
            "positive": "hello",
            "text_id": "abc",
            "mode": "text_image",
            "text": "hello",
        }
    ]
    batch = collator(features)
    assert "text_id" not in batch
    assert "anchor_input_ids" in batch


def test_split_sentence_features() -> None:
    batch = {
        "anchor_input_ids": torch.tensor([[1, 2]]),
        "positive_input_ids": torch.tensor([[1, 2]]),
        "negative_input_ids": torch.tensor([[3, 4]]),
        "anchor_task": "document",
    }
    features = split_sentence_features(batch, columns=("anchor", "positive"))
    assert len(features) == 2
    assert features[0]["input_ids"].shape == (1, 2)
    assert features[0]["task"] == "document"


def _fake_preprocess(
    inputs: list[object],
    *,
    prompt: str | None = None,
    task: str | None = None,
    max_length: int | None = None,
) -> dict[str, object]:
    del prompt, task, max_length
    count = len(inputs)
    return {
        "input_ids": torch.ones((count, 4), dtype=torch.long),
        "attention_mask": torch.ones((count, 4), dtype=torch.long),
    }
