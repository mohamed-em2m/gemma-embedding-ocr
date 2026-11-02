"""OXR-1.0 recognition adapter implementing RecognitionEngine.

Model card: https://huggingface.co/SoMarkAI/OXR-1.0

OXR is not documented for direct ``transformers`` inference: the card ships
an out-of-tree vLLM plugin and a three-service stack.  This adapter therefore
talks to any OpenAI-compatible ``/chat/completions`` endpoint (for example a
local vLLM server launched with the OXR plugin).  Start the server separately
and pass its base URL, e.g.::

    OXRRecognizer(base_url="http://127.0.0.1:8080/v1")

No extra Python dependency is required (``urllib`` is used).
"""

from __future__ import annotations

import base64
import io
import json
import urllib.error
import urllib.request
from typing import Any

import numpy as np
from PIL import Image

from embedding_ocr.recognition.base import RecognitionEngine, RecognitionResult

_DEFAULT_PROMPT = "Extract all text from this image. Output only the text."


class OXRRecognizer(RecognitionEngine):
    """HTTP adapter for a local OpenAI-compatible OXR-1.0 server."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080/v1",
        model: str = "SoMarkAI/OXR-1.0",
        api_key: str = "dummy",
        prompt: str = _DEFAULT_PROMPT,
        max_tokens: int = 128,
        timeout: float = 120.0,
    ) -> None:
        """Configure the client.

        Args:
            base_url: OpenAI-compatible base URL, including ``/v1``.
            model: Served model name.
            api_key: Bearer token; most local servers accept any value.
            prompt: Instruction sent alongside the image.
            max_tokens: Generation budget per crop.
            timeout: Per-request timeout in seconds.
        """
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.prompt = prompt
        self.max_tokens = max_tokens
        self.timeout = timeout

    def _encode_image(self, image: np.ndarray) -> str:
        pil_image = Image.fromarray(image)
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"

    def recognize(self, image: np.ndarray) -> RecognitionResult:
        """Recognize text in a single text-region crop."""
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": self._encode_image(image)},
                        },
                        {"type": "text", "text": self.prompt},
                    ],
                }
            ],
            "max_tokens": self.max_tokens,
            "temperature": 0.0,
        }
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body: dict[str, Any] = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            msg = (
                f"OXR request failed at {self.base_url}. Start an "
                f"OpenAI-compatible OXR/vLLM server first: {exc}"
            )
            raise RuntimeError(msg) from exc

        text = body["choices"][0]["message"]["content"].strip()
        return RecognitionResult(text=text, confidence=0.9 if text else 0.0)
