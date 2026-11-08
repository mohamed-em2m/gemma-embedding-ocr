"""Optional DTLR line-recognition baseline.

DTLR (Detection-based Text Line Recognition) is an external research
repository, not a pip package. This adapter loads an official DTLR checkout
from ``DTLR_ROOT`` and an inference checkpoint, then exposes the project's
``RecognitionEngine`` interface. The adapter is intentionally line-level:
DTLR is trained on text lines and is not a drop-in replacement for the
character/word embedding pipeline.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from embedding_ocr.recognition.base import RecognitionEngine, RecognitionResult

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]
_DEFAULT_IMAGE_SIZE = 800
_DEFAULT_MAX_SIZE = 1333
_DEFAULT_CONFIG = Path("config") / "Latin_CTC.py"
_DEFAULT_CHARSET = Path("datasets") / "default_charset.json"


def _optional_path(value: str | Path | None, env_name: str) -> Path | None:
    raw = value if value is not None else os.environ.get(env_name)
    return None if raw is None else Path(raw).expanduser()


def _output_field(outputs: Any, name: str) -> Any:
    if isinstance(outputs, dict):
        return outputs[name]
    return getattr(outputs, name)


def _decode_detections(
    outputs: Any, charset: list[str] | tuple[str, ...]
) -> tuple[str, float]:
    """Decode DTLR outputs using the repository's non-NMS convention.

    DTLR predicts a character for every query. Background is added at class
    index 0; positive classes are offset by one. Line order is recovered by
    sorting query boxes by horizontal centre.
    """
    pred_logits = _output_field(outputs, "pred_logits")
    pred_boxes = _output_field(outputs, "pred_boxes")

    if pred_logits.ndim == 3:
        pred_logits = pred_logits[0]
        pred_boxes = pred_boxes[0]
    if pred_logits.numel() == 0:
        return "", 0.0

    order = torch.argsort(pred_boxes[:, 0])
    logits = pred_logits[order].sigmoid()
    num_classes = logits.shape[-1]
    epsilon = 0.03 / num_classes

    total = logits.sum(dim=-1)
    scores = torch.zeros(
        (logits.shape[0], num_classes + 1), dtype=logits.dtype, device=logits.device
    )
    scores[:, 1:] = logits

    low = total < 1.0 - epsilon
    scores[low, 0] = 1.0 - total[low]
    scores[~low, 0] = epsilon
    scores[~low, 1:] = (1.0 - epsilon) * logits[~low] / total[~low].unsqueeze(-1)

    prediction = scores.argmax(dim=-1)
    selected = prediction != 0
    labels = (prediction[selected] - 1).tolist()

    text_chars: list[str] = []
    for label in labels:
        if 0 <= label < len(charset):
            text_chars.append(charset[label])

    if selected.any():
        confidence = float(
            scores[selected].gather(1, prediction[selected].unsqueeze(-1)).mean()
        )
    else:
        confidence = 0.0
    return "".join(text_chars), max(0.0, min(1.0, confidence))


class DTLRRecognizer(RecognitionEngine):
    """Line-level DTLR recognizer backed by an external checkout.

    Args:
        root: Path to the cloned DTLR repository. Falls back to ``DTLR_ROOT``.
        config_path: DTLR config. Defaults to ``config/Latin_CTC.py`` under
            ``root``.
        weights: Inference checkpoint path. Falls back to ``DTLR_WEIGHTS``.
        charset_path: JSON charset matching the checkpoint. Defaults to
            ``datasets/default_charset.json`` under ``root``.
        device: Torch device. The official DTLR code pins CUDA device 0 during
            import, so ``cuda`` is the supported setting.
    """

    name = "dtlr"

    def __init__(
        self,
        root: str | Path | None = None,
        config_path: str | Path | None = None,
        weights: str | Path | None = None,
        charset_path: str | Path | None = None,
        device: str = "cuda",
    ) -> None:
        resolved_root = _optional_path(root, "DTLR_ROOT")
        if resolved_root is None:
            msg = (
                "DTLR requires a cloned repository; pass root=... or set "
                "DTLR_ROOT to the checkout directory."
            )
            raise ValueError(msg)
        if not resolved_root.is_dir():
            msg = f"DTLR root is not a directory: {resolved_root}"
            raise FileNotFoundError(msg)
        if not (resolved_root / "models" / "dino" / "dino.py").exists():
            msg = f"{resolved_root} does not look like a DTLR checkout"
            raise FileNotFoundError(msg)

        resolved_weights = _optional_path(weights, "DTLR_WEIGHTS")
        if resolved_weights is None:
            msg = "DTLR requires weights=... or DTLR_WEIGHTS pointing to a checkpoint"
            raise ValueError(msg)
        if not resolved_weights.is_file():
            msg = f"DTLR checkpoint not found: {resolved_weights}"
            raise FileNotFoundError(msg)

        resolved_config = (
            _optional_path(config_path, "DTLR_CONFIG")
            or resolved_root / _DEFAULT_CONFIG
        )
        if not resolved_config.is_file():
            msg = f"DTLR config not found: {resolved_config}"
            raise FileNotFoundError(msg)

        resolved_charset = (
            _optional_path(charset_path, "DTLR_CHARSET")
            or resolved_root / _DEFAULT_CHARSET
        )
        if not resolved_charset.is_file():
            msg = f"DTLR charset not found: {resolved_charset}"
            raise FileNotFoundError(msg)

        self.root = resolved_root
        self.weights = resolved_weights
        self.config_path = resolved_config
        self.charset_path = resolved_charset
        self.charset = self._load_charset(resolved_charset)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self._model = self._load_model(device)
        self._image_size = _DEFAULT_IMAGE_SIZE
        self._max_size = _DEFAULT_MAX_SIZE

    @staticmethod
    def _load_charset(path: Path) -> tuple[str, ...]:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
            msg = f"DTLR charset must be a JSON list of strings: {path}"
            raise ValueError(msg)
        return tuple(raw)

    def _load_model(self, device: str) -> torch.nn.Module:
        if self.device.type != "cuda" or not torch.cuda.is_available():
            msg = (
                "DTLR's official code requires CUDA and calls "
                "torch.cuda.set_device(0) at import; install a CUDA torch build."
            )
            raise RuntimeError(msg)

        root_str = str(self.root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)

        try:
            from models import dino as dino_module  # noqa: F401
            from models.registry import MODULE_BUILD_FUNCS
            from util.slconfig import SLConfig
        except ImportError as exc:
            msg = (
                "Could not import DTLR. Install its inference dependencies "
                "(addict, yapf, timm, scipy, torchvision) and compile the "
                "multi-scale deformable-attention extension."
            )
            raise ImportError(msg) from exc

        config = SLConfig.fromfile(str(self.config_path))
        config.device = device
        config.num_classes = len(self.charset)
        config.dn_labelbook_size = len(self.charset) + 1

        build_func = MODULE_BUILD_FUNCS.get(config.modelname)
        if build_func is None:
            msg = f"Unsupported DTLR model name: {config.modelname!r}"
            raise ValueError(msg)

        model, _criterion, _postprocessors = build_func(config)
        checkpoint = self._load_checkpoint(self.weights)
        state = checkpoint.get("model", checkpoint)
        model.load_state_dict(state)
        return model.to(self.device).eval()

    @staticmethod
    def _load_checkpoint(path: Path) -> dict[str, Any]:
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            checkpoint = torch.load(path, map_location="cpu")
        if not isinstance(checkpoint, dict):
            msg = f"DTLR checkpoint must contain a state dict: {path}"
            raise ValueError(msg)
        return checkpoint

    def _preprocess(self, image: np.ndarray) -> torch.Tensor | None:
        from torchvision.transforms import functional as tv_functional

        array = np.asarray(image)
        if array.size == 0:
            return None
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)

        pil_image = Image.fromarray(array).convert("RGB")
        width, height = pil_image.size
        if width <= 0 or height <= 0:
            return None

        size = self._image_size
        longest = max(width, height)
        shortest = min(width, height)
        if longest / shortest * size > self._max_size:
            size = int(round(self._max_size * shortest / longest))

        if (width <= height and width == size) or (height <= width and height == size):
            resized_height, resized_width = height, width
        elif width < height:
            resized_width = size
            resized_height = int(size * height / width)
        else:
            resized_height = size
            resized_width = int(size * width / height)

        resized = tv_functional.resize(
            pil_image, (max(1, resized_height), max(1, resized_width))
        )
        tensor = tv_functional.to_tensor(resized)
        tensor = tv_functional.normalize(tensor, _IMAGENET_MEAN, _IMAGENET_STD)
        return tensor.unsqueeze(0).to(self.device)

    def recognize(self, image: np.ndarray) -> RecognitionResult:
        tensor = self._preprocess(image)
        if tensor is None:
            return RecognitionResult(text="", confidence=0.0)

        with torch.inference_mode():
            outputs = self._model(tensor)
        text, confidence = _decode_detections(outputs, self.charset)
        return RecognitionResult(text=text, confidence=confidence)
