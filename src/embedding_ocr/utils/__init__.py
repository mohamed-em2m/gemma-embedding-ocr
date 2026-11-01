"""Utility helpers: device selection, seeding, and logging setup."""

from __future__ import annotations

import logging
import random

import numpy as np


def get_device(device: str = "auto") -> str:
    """Resolve device string.

    Args:
        device: "auto", "cpu", or "cuda".

    Returns:
        "cuda" if CUDA is available and device is "auto" or "cuda"; else "cpu".
    """
    if device == "auto":
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    return device


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Create a logger with a standard format.

    Args:
        name: Logger name (usually __name__).
        level: Logging level.

    Returns:
        Configured Logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger
