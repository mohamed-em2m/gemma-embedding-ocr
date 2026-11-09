"""Tests for GPU selection and batch-size autotuning without real CUDA."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from embedding_ocr.recognition.embedding.embedder import EmbeddingGemma2
from embedding_ocr.utils.gpu import (
    BatchTuningResult,
    GPUInfo,
    autotune_batch_size,
    is_oom_error,
    recommend_batch_size,
    resolve_device,
    save_tuning_cache,
    select_best_gpu,
)


def _gpu(index: int, free_mb: float, utilization: float | None) -> GPUInfo:
    return GPUInfo(
        index=index,
        name=f"T4-{index}",
        total_mb=15360,
        free_mb=free_mb,
        utilization=utilization,
    )


def test_select_best_gpu_prefers_idle_device() -> None:
    gpus = [_gpu(0, 13000, 43), _gpu(1, 15000, 0)]
    best = select_best_gpu(gpus)
    assert best is not None
    assert best.index == 1


def test_select_best_gpu_prefers_free_memory() -> None:
    gpus = [_gpu(0, 12000, 0), _gpu(1, 15000, 80)]
    best = select_best_gpu(gpus)
    assert best is not None
    assert best.index == 1


def test_select_best_gpu_skips_low_memory() -> None:
    gpus = [_gpu(0, 512, 0), _gpu(1, 4096, 10)]
    best = select_best_gpu(gpus, min_free_mb=1024)
    assert best is not None
    assert best.index == 1


def test_select_best_gpu_none_when_all_low() -> None:
    assert select_best_gpu([_gpu(0, 256, 0)], min_free_mb=1024) is None


def test_resolve_device_auto_without_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("embedding_ocr.utils.gpu.query_gpus", lambda: [])
    assert resolve_device("auto") == "cpu"
    assert resolve_device("cuda") == "cuda"


def test_recommend_batch_size_uses_matching_cache(tmp_path: Path) -> None:
    path = tmp_path / "gpu_tuning.json"
    save_tuning_cache(
        path,
        {
            "device": "cuda:1",
            "model_id": "google/embeddinggemma-2",
            "dim": 768,
            "batch_size": 12,
        },
    )
    assert (
        recommend_batch_size(
            "cuda:1",
            model_id="google/embeddinggemma-2",
            dim=768,
            cache_path=path,
        )
        == 12
    )


def test_recommend_batch_size_heuristic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "embedding_ocr.utils.gpu.query_gpus",
        lambda: [_gpu(0, 8500, 0)],
    )
    assert recommend_batch_size("cuda:0", cache_path="/nonexistent") == 12


def test_is_oom_error() -> None:
    assert is_oom_error(RuntimeError("CUDA out of memory"))
    assert not is_oom_error(RuntimeError("shape mismatch"))


def test_autotune_batch_size_stops_on_oom() -> None:
    tried: list[int] = []

    def encode(size: int) -> None:
        tried.append(size)
        if size >= 8:
            raise RuntimeError("CUDA out of memory")

    result = autotune_batch_size(encode, device="cpu", candidates=(1, 2, 4, 8))
    assert isinstance(result, BatchTuningResult)
    assert result.batch_size == 4
    assert tried == [1, 2, 4, 8]
    assert result.attempts[-1]["status"] == "oom"


def test_embeddinggemma_halves_batch_on_oom(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeModel:
        def __init__(self) -> None:
            self.calls: list[int] = []

        def encode(self, inputs: list[str], **kwargs: object) -> np.ndarray:
            batch_size = int(kwargs["batch_size"])
            self.calls.append(batch_size)
            if len(self.calls) == 1:
                raise RuntimeError("CUDA out of memory")
            return np.ones((len(inputs), 8), dtype=np.float32)

    monkeypatch.setattr(
        "embedding_ocr.recognition.embedding.embedder.torch.cuda.empty_cache",
        lambda: None,
    )
    embedder = object.__new__(EmbeddingGemma2)
    embedder.dim = 8
    embedder.batch_size = 4
    embedder._device = "cuda:0"
    embedder._auto_oom_fallback = True
    embedder._model = FakeModel()

    result = embedder._encode(["a", "b", "c", "d"])

    assert result.shape == (4, 8)
    assert embedder.batch_size == 2
    assert embedder._model.calls == [4, 2]
