"""dots.mocr recognition adapter implementing RecognitionEngine.

Model card: https://huggingface.co/dots-studio/dots.mocr

The card documents loading through ``AutoModelForCausalLM`` and
``AutoProcessor`` with ``trust_remote_code=True``.  The exact prompt that
maximises single-glyph transcription quality is *unverified*; the default
here is a plain transcription instruction.  Pass ``prompt`` to override it.

``qwen_vl_utils`` is used when available (the card's reference path).  If it
is not installed, the adapter falls back to ``apply_chat_template`` with
tokenization, which works with processors that accept multimodal content
directly.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from PIL import Image

from embedding_ocr.recognition.base import RecognitionEngine, RecognitionResult

_DEFAULT_PROMPT = "Extract all text from this image. Output only the text."


class DotsMOCRRecognizer(RecognitionEngine):
    """Adapter for dots-studio/dots.mocr (3B document OCR VLM)."""

    def __init__(
        self,
        model_id: str = "dots-studio/dots.mocr",
        device: str = "auto",
        prompt: str = _DEFAULT_PROMPT,
        max_new_tokens: int = 128,
    ) -> None:
        """Load the model and processor.

        Args:
            model_id: Hugging Face model id.
            device: ``auto`` selects CUDA when available, else CPU.
            prompt: Instruction sent alongside the image.
            max_new_tokens: Generation budget per crop.
        """
        from transformers import AutoModelForCausalLM, AutoProcessor

        self.prompt = prompt
        self.max_new_tokens = max_new_tokens
        self._device = (
            "cuda" if device == "auto" and torch.cuda.is_available() else device
        )
        if self._device == "auto":
            self._device = "cpu"

        dtype = torch.bfloat16 if self._device == "cuda" else torch.float32
        self._processor = AutoProcessor.from_pretrained(
            model_id, trust_remote_code=True
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map=self._device,
            trust_remote_code=True,
        )
        self._model.eval()

    def _prepare_inputs(self, pil_image: Image.Image) -> Any:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": self.prompt},
                ],
            }
        ]
        try:
            from qwen_vl_utils import process_vision_info

            text = self._processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            image_inputs, video_inputs = process_vision_info(messages)
            return self._processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self._device)
        except ImportError:
            return self._processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self._device)

    def recognize(self, image: np.ndarray) -> RecognitionResult:
        """Recognize text in a single text-region crop."""
        pil_image = Image.fromarray(image)
        inputs = self._prepare_inputs(pil_image)

        with torch.no_grad():
            output = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        prompt_len = inputs["input_ids"].shape[1]
        generated = output[0][prompt_len:]
        text = self._processor.decode(generated, skip_special_tokens=True).strip()
        return RecognitionResult(text=text, confidence=0.9 if text else 0.0)
