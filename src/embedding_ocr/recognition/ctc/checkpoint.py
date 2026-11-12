"""Checkpoint persistence for the CTC line recognizer.

A checkpoint carries the architecture config, the charset, the training state
and the normalization policy. Loading validates that the charset and the
model's output size agree, so a checkpoint can never be paired with a
mismatched vocabulary.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from embedding_ocr.recognition.ctc.charset import CharSet
from embedding_ocr.recognition.ctc.model import CTCSeq2Seq, ModelConfig
from embedding_ocr.utils.io import git_hash

CHECKPOINT_VERSION = 1


def save_checkpoint(
    path: Path,
    model: CTCSeq2Seq,
    *,
    charset: CharSet,
    optimizer: torch.optim.Optimizer | None = None,
    epoch: int = 0,
    step: int = 0,
    metrics: dict[str, float] | None = None,
    normalization: dict[str, Any] | None = None,
) -> None:
    """Write a self-describing checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": CHECKPOINT_VERSION,
        "model_config": asdict(model.config),
        "charset": list(charset.chars),
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "epoch": epoch,
        "step": step,
        "metrics": metrics or {},
        "normalization": normalization or {},
        "git_hash": git_hash(),
    }
    torch.save(payload, path)


def load_checkpoint(path: Path, map_location: str = "cpu") -> dict[str, Any]:
    """Read a checkpoint payload."""
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    if not isinstance(payload, dict) or "model_config" not in payload:
        message = f"not a CTC checkpoint: {path}"
        raise ValueError(message)
    if payload.get("version") != CHECKPOINT_VERSION:
        message = (
            f"unsupported checkpoint version {payload.get('version')!r} in {path}; "
            f"expected {CHECKPOINT_VERSION}"
        )
        raise ValueError(message)
    return payload


def build_model(
    payload: dict[str, Any], device: str | torch.device = "cpu"
) -> tuple[CTCSeq2Seq, CharSet]:
    """Instantiate the model and charset recorded in a checkpoint payload."""
    charset = CharSet(tuple(payload["charset"]))
    config = ModelConfig(**payload["model_config"])
    if config.vocab_size != charset.size:
        message = (
            f"checkpoint charset has {charset.size} classes but the model was "
            f"built with vocab_size={config.vocab_size}"
        )
        raise ValueError(message)
    model = CTCSeq2Seq(config)
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model, charset


def load_model(
    path: Path, device: str | torch.device = "cpu"
) -> tuple[CTCSeq2Seq, CharSet]:
    """Load a checkpoint and return an eval-mode model with its charset."""
    return build_model(load_checkpoint(path), device=device)


def load_optimizer_state(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Return the optimizer state if the checkpoint has one."""
    state = payload.get("optimizer")
    return state if isinstance(state, dict) else None


def describe_checkpoint(path: Path) -> dict[str, Any]:
    """Return checkpoint metadata without building the model."""
    payload = load_checkpoint(path)
    return {
        "version": payload.get("version"),
        "epoch": payload.get("epoch"),
        "step": payload.get("step"),
        "metrics": payload.get("metrics", {}),
        "normalization": payload.get("normalization", {}),
        "git_hash": payload.get("git_hash"),
        "charset_size": len(payload.get("charset", [])),
        "model_config": payload.get("model_config"),
    }
