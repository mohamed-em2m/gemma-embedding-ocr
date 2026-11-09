"""GPU discovery, best-device selection, and batch-size tuning helpers."""

from __future__ import annotations

import json
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

DEFAULT_TUNING_PATH = Path("artifacts/gpu_tuning.json")
AUTO_BATCH_SIZE = 0
DEFAULT_BATCH_SIZE = 16
MAX_BATCH_SIZE = 64


@dataclass(frozen=True)
class GPUInfo:
    """One visible CUDA device and its current memory/utilisation state."""

    index: int
    name: str
    total_mb: float
    free_mb: float
    utilization: float | None = None

    @property
    def free_fraction(self) -> float:
        """Fraction of device memory currently available."""
        return self.free_mb / self.total_mb if self.total_mb else 0.0

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return asdict(self)


@dataclass(frozen=True)
class BatchTuningResult:
    """Result of increasing batch size until an OOM or the limit is reached."""

    batch_size: int
    peak_mb: float
    latency_ms: float
    attempts: tuple[dict[str, Any], ...]

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "batch_size": self.batch_size,
            "peak_mb": self.peak_mb,
            "latency_ms": self.latency_ms,
            "attempts": list(self.attempts),
        }


def _query_utilization() -> dict[int, float]:
    """Read GPU utilisation percentages from nvidia-smi when available."""
    command = [
        "nvidia-smi",
        "--query-gpu=index,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return {}
    if completed.returncode != 0:
        return {}
    utilization: dict[int, float] = {}
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 2:
            continue
        try:
            utilization[int(parts[0])] = float(parts[1])
        except ValueError:
            continue
    return utilization


def query_gpus() -> list[GPUInfo]:
    """Return visible CUDA devices with free memory and utilisation."""
    try:
        import torch
    except ImportError:
        return []
    if not torch.cuda.is_available():
        return []

    utilization = _query_utilization()
    gpus: list[GPUInfo] = []
    for index in range(torch.cuda.device_count()):
        try:
            properties = torch.cuda.get_device_properties(index)
            free_bytes, total_bytes = torch.cuda.mem_get_info(index)
        except (AssertionError, RuntimeError):
            continue
        gpus.append(
            GPUInfo(
                index=index,
                name=str(properties.name),
                total_mb=total_bytes / (1024 * 1024),
                free_mb=free_bytes / (1024 * 1024),
                utilization=utilization.get(index),
            )
        )
    return gpus


def select_best_gpu(
    gpus: Sequence[GPUInfo] | None = None,
    *,
    min_free_mb: float = 1024,
) -> GPUInfo | None:
    """Select the least-used GPU with enough free memory.

    The most free memory wins; ties are broken by utilisation and index.
    """
    candidates = [
        gpu
        for gpu in (query_gpus() if gpus is None else gpus)
        if gpu.free_mb >= min_free_mb
    ]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda gpu: (
            -gpu.free_mb,
            gpu.utilization if gpu.utilization is not None else 0.0,
            gpu.index,
        ),
    )


def resolve_device(device: str = "auto", *, min_free_mb: float = 1024) -> str:
    """Resolve ``auto`` or ``cuda`` to the best available CUDA device.

    Explicit values such as ``cpu`` or ``cuda:1`` are returned unchanged.
    """
    if device not in {"auto", "cuda"}:
        return device
    gpu = select_best_gpu(min_free_mb=min_free_mb)
    if gpu is None:
        return "cpu" if device == "auto" else "cuda"
    return f"cuda:{gpu.index}"


def describe_gpus(gpus: Sequence[GPUInfo] | None = None) -> str:
    """Return a compact human-readable GPU inventory."""
    inventory = list(query_gpus() if gpus is None else gpus)
    if not inventory:
        return "no CUDA devices"
    return "; ".join(
        f"cuda:{gpu.index} {gpu.name} free={gpu.free_mb:.0f}MB "
        f"util={gpu.utilization if gpu.utilization is not None else 'n/a'}"
        for gpu in inventory
    )


def _batch_size_from_free_memory(free_mb: float, *, max_batch: int) -> int:
    """Conservative fallback batch size when no tuned cache is available."""
    for threshold, batch in (
        (12000, 16),
        (8000, 12),
        (5000, 8),
        (3000, 4),
        (1500, 2),
    ):
        if free_mb >= threshold:
            return min(batch, max_batch)
    return 1


def recommend_batch_size(
    device: str,
    *,
    model_id: str | None = None,
    dim: int | None = None,
    cache_path: Path = DEFAULT_TUNING_PATH,
    fallback: int = DEFAULT_BATCH_SIZE,
    max_batch: int = MAX_BATCH_SIZE,
) -> int:
    """Choose a batch size from a tuned cache, then free-memory heuristics."""
    if device == "cpu":
        return max(1, min(fallback, max_batch))

    cached = get_cached_batch_size(
        device,
        model_id=model_id,
        dim=dim,
        cache_path=cache_path,
        max_batch=max_batch,
    )
    if cached is not None:
        return cached

    gpus = query_gpus()
    gpu = _find_gpu(gpus, device)
    if gpu is None:
        return max(1, min(fallback, max_batch))
    return _batch_size_from_free_memory(gpu.free_mb, max_batch=max_batch)


def get_cached_batch_size(
    device: str,
    *,
    model_id: str | None = None,
    dim: int | None = None,
    cache_path: Path = DEFAULT_TUNING_PATH,
    max_batch: int = MAX_BATCH_SIZE,
) -> int | None:
    """Return a matching cached batch size, or ``None`` when unavailable."""
    cached = load_tuning_cache(cache_path)
    if not cached or not _cache_matches(cached, device, model_id, dim):
        return None
    try:
        return max(1, min(int(cached["batch_size"]), max_batch))
    except (KeyError, TypeError, ValueError):
        return None


def _find_gpu(gpus: Sequence[GPUInfo], device: str) -> GPUInfo | None:
    if not gpus:
        return None
    if device.startswith("cuda:"):
        try:
            index = int(device.split(":", 1)[1])
        except ValueError:
            return None
        return next((gpu for gpu in gpus if gpu.index == index), None)
    return select_best_gpu(gpus)


def _cache_matches(
    cached: dict[str, Any],
    device: str,
    model_id: str | None,
    dim: int | None,
) -> bool:
    cached_device = str(cached.get("device", ""))
    if (
        cached_device
        and cached_device != device
        and not (cached_device.startswith("cuda") and device.startswith("cuda"))
    ):
        return False
    if model_id is not None and cached.get("model_id") not in (None, model_id):
        return False
    return not (dim is not None and cached.get("dim") not in (None, dim))


def load_tuning_cache(path: Path = DEFAULT_TUNING_PATH) -> dict[str, Any] | None:
    """Load a tuning result, returning ``None`` for missing/invalid files."""
    try:
        payload = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def save_tuning_cache(path: Path, payload: dict[str, Any]) -> None:
    """Write a tuning result as JSON, creating parent directories."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")


def is_oom_error(exc: BaseException) -> bool:
    """Return true when an exception is a CUDA out-of-memory error."""
    if exc.__class__.__name__ == "OutOfMemoryError":
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def autotune_batch_size(
    encode: Callable[[int], Any],
    *,
    device: str,
    candidates: Sequence[int] = (1, 2, 4, 8, 12, 16, 24, 32, 48, 64),
    max_batch: int = MAX_BATCH_SIZE,
) -> BatchTuningResult:
    """Run ``encode(batch_size)`` until OOM, returning the largest safe batch."""
    sizes = sorted({size for size in candidates if 1 <= size <= max_batch})
    if not sizes:
        msg = f"no candidate batch sizes in [1, {max_batch}]"
        raise ValueError(msg)

    peak_mb = 0.0
    latency_ms = 0.0
    attempts: list[dict[str, Any]] = []
    best = 0
    best_peak = 0.0
    best_latency = 0.0

    for size in sizes:
        if device.startswith("cuda"):
            import torch

            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        try:
            encode(size)
        except Exception as exc:
            if not is_oom_error(exc):
                raise
            attempts.append({"batch_size": size, "status": "oom"})
            if device.startswith("cuda"):
                import torch

                torch.cuda.empty_cache()
            break
        elapsed_ms = (time.perf_counter() - started) * 1000
        if device.startswith("cuda"):
            import torch

            torch.cuda.synchronize()
            peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        latency_ms = elapsed_ms
        best = size
        best_peak = peak_mb
        best_latency = latency_ms
        attempts.append(
            {
                "batch_size": size,
                "status": "ok",
                "peak_mb": round(peak_mb, 2),
                "latency_ms": round(latency_ms, 2),
            }
        )

    if best == 0:
        msg = "batch size 1 failed to run; cannot determine a safe batch size"
        raise RuntimeError(msg)
    return BatchTuningResult(
        batch_size=best,
        peak_mb=round(best_peak, 2),
        latency_ms=round(best_latency, 2),
        attempts=tuple(attempts),
    )
