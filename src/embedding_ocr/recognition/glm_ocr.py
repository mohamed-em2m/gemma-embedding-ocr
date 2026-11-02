"""GLM-OCR recognition adapter implementing RecognitionEngine."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from embedding_ocr.recognition.base import RecognitionEngine, RecognitionResult


class GLMOCRRecognizer(RecognitionEngine):
    """Adapter for GLM-OCR (zai-org/GLM-OCR)."""

    def __init__(
        self,
        model_id: str = "zai-org/GLM-OCR",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype: torch.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    ) -> None:
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map=device,
            trust_remote_code=True,
        )

    def recognize(self, image: np.ndarray) -> RecognitionResult:
        """Recognize text in an image crop.

        Args:
            image: Text-region image as uint8 numpy array (H, W, C).

        Returns:
            RecognitionResult with recognized text and confidence.
        """
        pil_image = Image.fromarray(image)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": "Text Recognition:"},
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=128)

        text = self.processor.decode(output[0], skip_special_tokens=True).strip()
        # Clean potential prompt echoes
        if "Text Recognition:" in text:
            text = text.split("Text Recognition:", 1)[-1].strip()

        return RecognitionResult(text=text, confidence=0.9 if text else 0.0)
