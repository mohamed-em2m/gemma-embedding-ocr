"""Surya OCR 2.5 recognition adapter implementing RecognitionEngine.

Model card: https://huggingface.co/datalab-to/surya-ocr-2.5

Surya 2.5 is a single ~689M-parameter document VLM (Qwen3.5-style) that
handles layout analysis, OCR and table recognition.  The official
``surya-ocr`` package wraps the model and an inference backend (vLLM on
NVIDIA GPUs, llama.cpp otherwise)::

    pip install surya-ocr

The package is an **optional** dependency.  Install it with ``uv sync
--extra surya`` (or ``uv pip install surya-ocr``) before using this adapter.
The adapter feeds the crop to ``RecognitionPredictor`` and converts the
returned HTML blocks into plain text.
"""

from __future__ import annotations

import html
import os
import re
from typing import Any

import numpy as np
from PIL import Image

from embedding_ocr.recognition.base import RecognitionEngine, RecognitionResult

_TAG_RE = re.compile(r"<[^>]+>")


def _html_to_text(value: str) -> str:
    """Convert Surya's HTML/plain block output into plain text."""
    text = _TAG_RE.sub(" ", value)
    text = html.unescape(text)
    return " ".join(text.split()).strip()


def _block_field(block: Any, name: str, default: Any = None) -> Any:
    if isinstance(block, dict):
        return block.get(name, default)
    return getattr(block, name, default)


class SuryaOCRRecognizer(RecognitionEngine):
    """Adapter for datalab-to/surya-ocr-2.5 through the ``surya-ocr`` package."""

    def __init__(
        self,
        model_id: str = "datalab-to/surya-ocr-2.5",
        device: str = "auto",
        backend: str | None = None,
        max_new_tokens: int = 2048,
    ) -> None:
        """Create the inference manager and recognition predictor.

        Args:
            model_id: Surya checkpoint (Hugging Face id or local directory).
            device: ``auto`` | ``cpu`` | ``cuda``.  Used to pick a backend when
                ``backend`` is not given explicitly.
            backend: ``vllm``, ``llamacpp``, ``mlx`` or ``None`` for auto.
            max_new_tokens: Output budget passed to the predictor when the
                installed surya version supports it.
        """
        try:
            from surya.inference import SuryaInferenceManager
            from surya.recognition import RecognitionPredictor
        except ImportError as exc:  # pragma: no cover - depends on optional dep
            msg = (
                "surya-ocr is not installed. Install it with "
                "`uv sync --extra surya` or `uv pip install surya-ocr`, then "
                "ensure a vLLM (CUDA) or llama.cpp (CPU) backend is available."
            )
            raise ImportError(msg) from exc

        os.environ.setdefault("SURYA_MODEL_CHECKPOINT", model_id)
        if backend is not None:
            os.environ["SURYA_INFERENCE_BACKEND"] = backend
        elif device == "cuda":
            os.environ.setdefault("SURYA_INFERENCE_BACKEND", "vllm")
        elif device == "cpu":
            os.environ.setdefault("SURYA_INFERENCE_BACKEND", "llamacpp")

        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self._manager = SuryaInferenceManager()
        self._predictor = RecognitionPredictor(self._manager)

    def recognize(self, image: np.ndarray) -> RecognitionResult:
        """Recognize text in a single text-region crop."""
        pil_image = Image.fromarray(image).convert("RGB")
        try:
            predictions = self._predictor(
                [pil_image], max_new_tokens=self.max_new_tokens
            )
        except TypeError:
            predictions = self._predictor([pil_image])

        if not predictions:
            return RecognitionResult(text="", confidence=0.0)

        page = predictions[0]
        blocks = _block_field(page, "blocks", []) or []
        chunks: list[str] = []
        confidences: list[float] = []
        for block in blocks:
            if bool(_block_field(block, "skipped", False)):
                continue
            raw = _block_field(block, "html")
            if raw is None:
                raw = _block_field(block, "content", "")
            text = _html_to_text(str(raw))
            if text:
                chunks.append(text)
            confidence = _block_field(block, "confidence")
            if isinstance(confidence, (int, float)):
                confidences.append(float(confidence))

        text = "\n".join(chunks).strip()
        if text:
            confidence = sum(confidences) / len(confidences) if confidences else 0.9
        else:
            confidence = 0.0
        return RecognitionResult(text=text, confidence=confidence)
