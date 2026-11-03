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

from typing import Any, cast

import numpy as np
import torch
from PIL import Image

from embedding_ocr.recognition.base import RecognitionEngine, RecognitionResult

_DEFAULT_PROMPT = "Output the single character in this image."


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
        from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor

        self.prompt = prompt
        self.max_new_tokens = max_new_tokens
        self._device = (
            "cuda" if device == "auto" and torch.cuda.is_available() else device
        )
        if self._device == "auto":
            self._device = "cpu"

        dtype = torch.bfloat16 if self._device == "cuda" else torch.float32
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        if hasattr(config, "vision_config"):
            if isinstance(config.vision_config, dict):
                config.vision_config["attn_implementation"] = "sdpa"
            else:
                config.vision_config.attn_implementation = "sdpa"
        if hasattr(config, "_attn_implementation"):
            config._attn_implementation = "sdpa"

        self._processor = AutoProcessor.from_pretrained(
            model_id, trust_remote_code=True
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            model_id,
            config=config,
            dtype=dtype,
            device_map=self._device,
            trust_remote_code=True,
            attn_implementation="sdpa",
        )
        self._model.eval()

        # Ensure class method handles None cache_position safely in modern transformers
        cls = type(self._model)
        orig_prepare = cls.prepare_inputs_for_generation

        def _safe_prepare(
            model_self: Any,
            input_ids: torch.Tensor,
            cache_position: torch.Tensor | None = None,
            **kwargs: Any,
        ) -> dict[str, Any]:
            input_ids_long = cast(torch.LongTensor, input_ids.long())
            try:
                return orig_prepare(
                    model_self,
                    input_ids_long,
                    cache_position=cache_position,
                    **kwargs,
                )
            except TypeError as exc:
                if "NoneType" in str(exc) and "subscriptable" in str(exc):
                    # Fallback when cache_position was None
                    seq_len = input_ids.shape[1]
                    fallback_pos = torch.arange(
                        seq_len, device=input_ids.device, dtype=torch.long
                    )
                    return orig_prepare(
                        model_self,
                        input_ids_long,
                        cache_position=fallback_pos,
                        **kwargs,
                    )
                raise

        cls.prepare_inputs_for_generation = _safe_prepare  # type: ignore[assignment]

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
        if isinstance(inputs, dict):
            inputs.pop("mm_token_type_ids", None)
            if "input_ids" in inputs and "cache_position" not in inputs:
                inputs["cache_position"] = torch.arange(
                    inputs["input_ids"].shape[1], device=inputs["input_ids"].device
                )
        elif hasattr(inputs, "pop"):
            inputs.pop("mm_token_type_ids", None)

        with torch.no_grad():
            output = cast(Any, self._model).generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                eos_token_id=[151643, 151673],
                do_sample=False,
            )

        prompt_len = inputs["input_ids"].shape[1]
        generated = output[0][prompt_len:]
        text = self._processor.decode(generated, skip_special_tokens=True).strip()
        return RecognitionResult(text=text, confidence=0.9 if text else 0.0)
