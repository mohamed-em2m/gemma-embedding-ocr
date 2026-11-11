"""Probe safe training batch sizes before a long fine-tuning run.

The probe runs real forward/backward passes without stepping the optimizer.
It stops at the first CUDA OOM and keeps memory headroom for the optimizer,
gradient buffers, and allocator fragmentation.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from embedding_ocr.utils.gpu import (
    MAX_BATCH_SIZE,
    BatchTuningResult,
    is_oom_error,
    load_tuning_cache,
)

DEFAULT_BATCH_TUNING_PATH = Path("artifacts/batch_tuning.json")
DEFAULT_CANDIDATES = (1, 2, 4, 8, 16, 32, 64)


@dataclass(frozen=True)
class TrainingBatchProbeResult:
    """Result of probing one process' largest safe training batch."""

    batch_size: int
    peak_mb: float
    latency_ms: float
    throughput: float
    world_size: int = 1
    attempts: tuple[dict[str, Any], ...] = ()
    cached: bool = False
    key: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "batch_size": self.batch_size,
            "peak_mb": self.peak_mb,
            "latency_ms": self.latency_ms,
            "throughput": self.throughput,
            "world_size": self.world_size,
            "attempts": list(self.attempts),
            "cached": self.cached,
            "key": dict(self.key),
        }


def probe_training_batch(
    step: Callable[[int], None],
    *,
    key: Mapping[str, Any],
    device: str,
    candidates: Sequence[int] = DEFAULT_CANDIDATES,
    max_batch: int = MAX_BATCH_SIZE,
    headroom: float = 0.15,
    cache_path: Path = DEFAULT_BATCH_TUNING_PATH,
    retune: bool = False,
    world_size: int = 1,
) -> TrainingBatchProbeResult:
    """Find the largest batch whose forward/backward pass fits.

    Args:
        step: Callable that performs one forward/backward pass for a batch
            size. It must not call ``optimizer.step()``.
        key: Settings that determine memory use. A cached result is reused only
            when every key/value matches.
        device: ``cpu`` or a CUDA device string.
        candidates: Batch sizes to try in ascending order.
        max_batch: Hard upper bound.
        headroom: Fraction of device memory reserved for optimizer state and
            allocator fragmentation.
        cache_path: JSON cache written after a successful probe.
        retune: Ignore an existing matching cache entry.
        world_size: Number of DDP processes used by the caller. The final run
            should use the minimum probe result across ranks.

    Returns:
        The largest safe batch size and measured peak memory/throughput.
    """
    if not 0 <= headroom < 1:
        msg = f"headroom must be in [0, 1), got {headroom}"
        raise ValueError(msg)
    if max_batch < 1:
        msg = f"max_batch must be positive, got {max_batch}"
        raise ValueError(msg)

    cache_key = dict(key)
    if device.startswith("cuda"):
        cache_key["device"] = _device_name(device)
        cache_key["gpu_count"] = max(1, world_size)
        cache_key["precision"] = cache_key.get("precision", "float32")

    if not retune:
        cached = _load_cached_result(cache_path, cache_key)
        if cached is not None:
            return cached

    sizes = sorted({size for size in candidates if 1 <= size <= max_batch})
    if not sizes:
        msg = f"no candidate batch sizes in [1, {max_batch}]"
        raise ValueError(msg)

    total_mb = _total_memory_mb(device)
    limit_mb = total_mb * (1.0 - headroom) if total_mb else 0.0
    attempts: list[dict[str, Any]] = []
    best = 0
    best_peak = 0.0
    best_latency = 0.0
    best_throughput = 0.0

    for size in sizes:
        _reset_peak(device)
        started = time.perf_counter()
        try:
            step(size)
        except Exception as exc:
            if not is_oom_error(exc):
                raise
            attempts.append({"batch_size": size, "status": "oom"})
            _empty_cache(device)
            break
        _synchronize(device)
        elapsed = time.perf_counter() - started
        peak_mb = _peak_memory_mb(device)
        latency_ms = elapsed * 1000
        throughput = size / elapsed if elapsed > 0 else 0.0
        attempts.append(
            {
                "batch_size": size,
                "status": "ok",
                "peak_mb": round(peak_mb, 2),
                "latency_ms": round(latency_ms, 2),
                "throughput": round(throughput, 2),
            }
        )
        if limit_mb and peak_mb > limit_mb:
            attempts[-1]["status"] = "over_headroom"
            attempts[-1]["limit_mb"] = round(limit_mb, 2)
            _empty_cache(device)
            if best == 0:
                best = size
                best_peak = peak_mb
                best_latency = latency_ms
                best_throughput = throughput
            break
        best = size
        best_peak = peak_mb
        best_latency = latency_ms
        best_throughput = throughput

    if best == 0:
        msg = "batch size 1 failed to run; cannot determine a safe batch size"
        raise RuntimeError(msg)

    result = TrainingBatchProbeResult(
        batch_size=best,
        peak_mb=round(best_peak, 2),
        latency_ms=round(best_latency, 2),
        throughput=round(best_throughput, 2),
        world_size=max(1, world_size),
        attempts=tuple(attempts),
        key=cache_key,
    )
    _save_cached_result(cache_path, result)
    return result


def aggregate_probe_results(
    results: Sequence[TrainingBatchProbeResult],
    *,
    cache_path: Path = DEFAULT_BATCH_TUNING_PATH,
) -> TrainingBatchProbeResult:
    """Combine DDP rank probes into one conservative result.

    The final batch size is the minimum across ranks; peak memory is the
    maximum. The aggregate is written to the cache for later runs.
    """
    if not results:
        msg = "at least one probe result is required"
        raise ValueError(msg)
    first = results[0]
    combined = TrainingBatchProbeResult(
        batch_size=min(result.batch_size for result in results),
        peak_mb=max(result.peak_mb for result in results),
        latency_ms=max(result.latency_ms for result in results),
        throughput=sum(result.throughput for result in results),
        world_size=sum(max(1, result.world_size) for result in results),
        attempts=tuple(attempt for result in results for attempt in result.attempts),
        key=first.key,
    )
    _save_cached_result(cache_path, combined)
    return combined


def _load_cached_result(
    path: Path, key: Mapping[str, Any]
) -> TrainingBatchProbeResult | None:
    payload = load_tuning_cache(path)
    if not payload or payload.get("key") != dict(key):
        return None
    try:
        return TrainingBatchProbeResult(
            batch_size=max(1, int(payload["batch_size"])),
            peak_mb=float(payload["peak_mb"]),
            latency_ms=float(payload["latency_ms"]),
            throughput=float(payload["throughput"]),
            world_size=max(1, int(payload.get("world_size", 1))),
            attempts=tuple(payload.get("attempts", [])),
            cached=True,
            key=key,
        )
    except (KeyError, TypeError, ValueError):
        return None


def _save_cached_result(path: Path, result: TrainingBatchProbeResult) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = result.as_dict()
    payload.pop("cached", None)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _device_name(device: str) -> str:
    try:
        import torch

        index = int(device.split(":", 1)[1]) if ":" in device else 0
        if torch.cuda.is_available():
            return str(torch.cuda.get_device_properties(index).name)
    except (ImportError, RuntimeError, ValueError):
        pass
    return device


def _total_memory_mb(device: str) -> float:
    if not device.startswith("cuda"):
        return 0.0
    try:
        import torch

        index = int(device.split(":", 1)[1]) if ":" in device else 0
        return torch.cuda.get_device_properties(index).total_memory / (1024 * 1024)
    except (ImportError, RuntimeError, ValueError):
        return 0.0


def _reset_peak(device: str) -> None:
    if not device.startswith("cuda"):
        return
    import torch

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def _peak_memory_mb(device: str) -> float:
    if not device.startswith("cuda"):
        return 0.0
    import torch

    return torch.cuda.max_memory_allocated() / (1024 * 1024)


def _empty_cache(device: str) -> None:
    if not device.startswith("cuda"):
        return
    import torch

    torch.cuda.empty_cache()


def _synchronize(device: str) -> None:
    if not device.startswith("cuda"):
        return
    import torch

    torch.cuda.synchronize()


def to_batch_tuning_result(result: TrainingBatchProbeResult) -> BatchTuningResult:
    """Convert a training probe into the shared batch-tuning dataclass."""
    return BatchTuningResult(
        batch_size=result.batch_size,
        peak_mb=result.peak_mb,
        latency_ms=result.latency_ms,
        attempts=result.attempts,
    )
