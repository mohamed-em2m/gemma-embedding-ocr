"""Tests for training batch-size probing without real CUDA."""

from __future__ import annotations

from pathlib import Path

import pytest

from embedding_ocr.training.batch_probe import (
    aggregate_probe_results,
    probe_training_batch,
    to_batch_tuning_result,
)


def test_probe_stops_on_oom_and_caches(tmp_path: Path) -> None:
    cache = tmp_path / "batch_tuning.json"
    tried: list[int] = []

    def step(size: int) -> None:
        tried.append(size)
        if size >= 8:
            raise RuntimeError("CUDA out of memory")

    key = {"model_id": "google/embeddinggemma-2", "precision": "float32"}
    result = probe_training_batch(
        step,
        key=key,
        device="cpu",
        candidates=(1, 2, 4, 8),
        cache_path=cache,
    )

    assert result.batch_size == 4
    assert tried == [1, 2, 4, 8]
    assert result.attempts[-1]["status"] == "oom"
    assert cache.exists()


def test_probe_reuses_matching_cache(tmp_path: Path) -> None:
    cache = tmp_path / "batch_tuning.json"
    key = {"model_id": "google/embeddinggemma-2", "precision": "float32"}
    first = probe_training_batch(
        lambda _size: None,
        key=key,
        device="cpu",
        candidates=(1, 2),
        cache_path=cache,
    )
    assert first.batch_size == 2

    def fail(_size: int) -> None:
        raise AssertionError("step should not run when the cache matches")

    second = probe_training_batch(
        fail,
        key=key,
        device="cpu",
        candidates=(1, 2),
        cache_path=cache,
    )
    assert second.cached
    assert second.batch_size == 2


def test_probe_retunes_when_key_changes(tmp_path: Path) -> None:
    cache = tmp_path / "batch_tuning.json"
    probe_training_batch(
        lambda _size: None,
        key={"model_id": "base", "precision": "float32"},
        device="cpu",
        candidates=(1, 2),
        cache_path=cache,
    )
    called: list[int] = []

    def step(size: int) -> None:
        called.append(size)

    result = probe_training_batch(
        step,
        key={"model_id": "finetuned", "precision": "float32"},
        device="cpu",
        candidates=(1, 2),
        cache_path=cache,
    )
    assert not result.cached
    assert called == [1, 2]


def test_probe_retune_flag_forces_rerun(tmp_path: Path) -> None:
    cache = tmp_path / "batch_tuning.json"
    key = {"model_id": "model"}
    probe_training_batch(
        lambda _size: None,
        key=key,
        device="cpu",
        candidates=(1,),
        cache_path=cache,
    )
    called: list[int] = []

    def step(size: int) -> None:
        called.append(size)

    result = probe_training_batch(
        step,
        key=key,
        device="cpu",
        candidates=(1,),
        cache_path=cache,
        retune=True,
    )
    assert not result.cached
    assert called == [1]


def test_aggregate_uses_minimum_batch_and_maximum_memory(tmp_path: Path) -> None:
    cache = tmp_path / "batch_tuning.json"
    first = probe_training_batch(
        lambda _size: None,
        key={"model_id": "model"},
        device="cpu",
        candidates=(1, 2, 4),
        cache_path=tmp_path / "a.json",
        world_size=1,
    )
    second = probe_training_batch(
        lambda _size: None,
        key={"model_id": "model"},
        device="cpu",
        candidates=(1, 2),
        cache_path=tmp_path / "b.json",
        world_size=1,
    )
    combined = aggregate_probe_results([first, second], cache_path=cache)
    assert combined.batch_size == 2
    assert combined.world_size == 2
    assert cache.exists()


def test_probe_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="headroom"):
        probe_training_batch(
            lambda _size: None,
            key={},
            device="cpu",
            headroom=1.0,
        )
    with pytest.raises(ValueError, match="no candidate"):
        probe_training_batch(
            lambda _size: None,
            key={},
            device="cpu",
            candidates=(),
        )


def test_to_batch_tuning_result(tmp_path: Path) -> None:
    result = probe_training_batch(
        lambda _size: None,
        key={"model_id": "model"},
        device="cpu",
        candidates=(1, 2),
        cache_path=tmp_path / "batch_tuning.json",
    )
    shared = to_batch_tuning_result(result)
    assert shared.batch_size == 2
