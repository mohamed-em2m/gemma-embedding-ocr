"""LightOnOCR-3 recognition adapter implementing RecognitionEngine.

Model card: https://huggingface.co/lightonai/LightOnOCR-3-0.8B

The card documents the Qwen3.5 vision-language architecture (Transformers
>= 5.5.4) and says that only two prompts are in-distribution: an empty
string for transcription-only mode and ``"grounding"``.  The adapter
therefore defaults to ``prompt=""`` (full transcription).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from PIL import Image

from embedding_ocr.recognition.base import RecognitionEngine, RecognitionResult


def _load_model_class() -> type[Any]:
    """Return the documented model class, with a fallback for newer hubs."""
    import transformers

    for attribute in (
        "Qwen3_5ForConditionalGeneration",
        "AutoModelForMultimodalLM",
    ):
        model_cls = getattr(transformers, attribute, None)
        if model_cls is not None:
            return model_cls

    msg = (
        "This transformers build exposes neither "
        "Qwen3_5ForConditionalGeneration nor AutoModelForMultimodalLM; "
        "upgrade to transformers>=5.5.4."
    )
    raise ImportError(msg)


class LightOnOCRRecognizer(RecognitionEngine):
    """Adapter for lightonai/LightOnOCR-3-0.8B."""

    def __init__(
        self,
        model_id: str = "lightonai/LightOnOCR-3-0.8B",
        device: str = "auto",
        prompt: str = "",
        max_new_tokens: int = 256,
    ) -> None:
        """Load the model and processor.

        Args:
            model_id: Hugging Face model id.
            device: ``auto`` selects CUDA when available, else CPU.
            prompt: ``""`` for transcription-only, ``"grounding"`` for layout.
            max_new_tokens: Generation budget per crop.
        """
        from transformers import AutoProcessor

        self.prompt = prompt
        self.max_new_tokens = max_new_tokens
        self._device = (
            "cuda" if device == "auto" and torch.cuda.is_available() else device
        )
        if self._device == "auto":
            self._device = "cpu"

        dtype = torch.bfloat16 if self._device == "cuda" else torch.float32
        model_cls = _load_model_class()
        self._processor = AutoProcessor.from_pretrained(model_id)
        self._model = model_cls.from_pretrained(
            model_id,
            dtype=dtype,
            device_map=self._device,
        )
        self._model.eval()

    def _prepare_inputs(self, pil_image: Image.Image) -> Any:
        content: list[dict[str, Any]] = [{"type": "image", "image": pil_image}]
        if self.prompt:
            content.append({"type": "text", "text": self.prompt})
        conversation = [{"role": "user", "content": content}]
        return self._processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        ).to(self._device)

    def recognize(self, image: np.ndarray) -> RecognitionResult:
        """Recognize text in a single text-region crop."""
        pil_image = Image.fromarray(image)
        inputs = self._prepare_inputs(pil_image)

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        prompt_len = inputs["input_ids"].shape[1]
        generated_ids = output_ids[0, prompt_len:]
        text = self._processor.decode(generated_ids, skip_special_tokens=True).strip()
        return RecognitionResult(text=text, confidence=0.9 if text else 0.0)
