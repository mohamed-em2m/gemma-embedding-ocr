"""OXR-1.0 recognition adapter implementing RecognitionEngine.

Model card: https://huggingface.co/SoMarkAI/OXR-1.0

OXR ships two different, documented serving interfaces:

1. **SoMark-compatible parsing API** (recommended).  The model card's local
   stack exposes ``POST /v1/parse/sync`` on port 8000, and the public free
   endpoint is ``https://free-oxr.somark.ai/api/v1``.  This adapter uploads
   the crop as a file and reads ``data.outputs.markdown`` / ``data.outputs.json``.
2. **OpenAI-compatible chat API** exposed by the out-of-tree vLLM plugin on
   port 8080 (``POST /v1/chat/completions``).

``backend="auto"`` (the default) chooses the parsing API when the URL looks
like a SoMark endpoint (port 8000, ``/api/`` in the path, or a ``somark``
host) and the chat API otherwise.  Set ``backend="parse"`` or ``"chat"`` to
force one.

Examples::

    # Public free API (no API key required)
    OXRRecognizer(base_url="https://free-oxr.somark.ai/api/v1")

    # Local document-parsing API (model card default stack)
    OXRRecognizer(base_url="http://127.0.0.1:8000/v1")

    # Local vLLM model service
    OXRRecognizer(base_url="http://127.0.0.1:8080/v1", backend="chat")

Only the Python standard library is used; no extra dependency is required.
"""

from __future__ import annotations

import base64
import html
import io
import json
import re
import secrets
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

import numpy as np
from PIL import Image

from embedding_ocr.recognition.base import RecognitionEngine, RecognitionResult

_DEFAULT_PROMPT = "Extract all text from this image. Output only the text."
_FREE_API_URL = "https://free-oxr.somark.ai/api/v1"
_TAG_RE = re.compile(r"<[^>]+>")


def _strip_markup(text: str) -> str:
    """Turn small markdown/HTML fragments into plain text."""
    without_tags = _TAG_RE.sub(" ", text)
    unescaped = html.unescape(without_tags)
    return " ".join(unescaped.split()).strip()


class OXRRecognizer(RecognitionEngine):
    """HTTP adapter for OXR-1.0 (parsing API or OpenAI-compatible chat API)."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000/v1",
        model: str = "SoMarkAI/OXR-1.0",
        api_key: str = "dummy",
        prompt: str = _DEFAULT_PROMPT,
        backend: str = "auto",
        max_tokens: int = 128,
        timeout: float = 120.0,
    ) -> None:
        """Configure the client.

        Args:
            base_url: Parsing API or OpenAI-compatible base URL, including
                ``/v1`` (for example ``http://127.0.0.1:8000/v1`` or
                ``https://free-oxr.somark.ai/api/v1``).
            model: Served model name (chat backend only).
            api_key: SoMark API key / bearer token.  Local servers and the
                free API accept ``dummy``.
            prompt: Instruction sent alongside the image (chat backend only).
            backend: ``auto``, ``parse`` (SoMark ``/parse/sync``) or ``chat``
                (OpenAI ``/chat/completions``).
            max_tokens: Generation budget per crop (chat backend only).
            timeout: Per-request timeout in seconds.
        """
        if backend not in {"auto", "parse", "chat"}:
            msg = f"backend must be 'auto', 'parse' or 'chat'; got {backend!r}"
            raise ValueError(msg)

        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.prompt = prompt
        self.backend = backend
        self.max_tokens = max_tokens
        self.timeout = timeout

    # ------------------------------------------------------------------
    # Backend selection
    # ------------------------------------------------------------------
    def _resolved_backend(self) -> str:
        if self.backend != "auto":
            return self.backend
        parsed = urllib.parse.urlparse(self.base_url)
        host = parsed.netloc.lower()
        path = parsed.path.rstrip("/").lower()
        if (
            "somark" in host
            or "/api/" in path
            or parsed.port == 8000
            or path.endswith("/api/v1")
        ):
            return "parse"
        return "chat"

    # ------------------------------------------------------------------
    # Image encoding
    # ------------------------------------------------------------------
    @staticmethod
    def _encode_image(image: np.ndarray) -> tuple[bytes, str]:
        pil_image = Image.fromarray(image).convert("RGB")
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        return buffer.getvalue(), "crop.png"

    @staticmethod
    def _encode_data_url(image_bytes: bytes) -> str:
        encoded = base64.b64encode(image_bytes).decode("ascii")
        return f"data:image/png;base64,{encoded}"

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------
    def _post_chat(self, image: np.ndarray) -> str:
        image_bytes, _ = self._encode_image(image)
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": self._encode_data_url(image_bytes)},
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
        body, _ = self._open(request)
        choices = body.get("choices") or []
        if not choices:
            return ""
        message = choices[0].get("message") or {}
        content = message.get("content", "")
        return str(content).strip()

    def _post_parse(self, image: np.ndarray) -> str:
        image_bytes, filename = self._encode_image(image)
        fields: list[tuple[str, str]] = [
            ("api_key", self.api_key),
            ("output_formats", "markdown"),
            ("output_formats", "json"),
            ("element_formats", json.dumps({"image": "none", "table": "markdown"})),
            (
                "feature_config",
                json.dumps(
                    {
                        "enable_image_understanding": False,
                        "enable_table_image": False,
                    }
                ),
            ),
        ]
        body_bytes, content_type = _encode_multipart(
            fields=fields,
            file_field="file",
            filename=filename,
            file_bytes=image_bytes,
            file_content_type="image/png",
        )
        request = urllib.request.Request(
            f"{self.base_url}/parse/sync",
            data=body_bytes,
            headers={
                "Content-Type": content_type,
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        body, _ = self._open(request)
        code = body.get("code")
        if code not in (None, 0):
            msg = f"OXR parse failed: code={code} message={body.get('message')!r}"
            raise RuntimeError(msg)
        return _extract_parse_text(body)

    def _open(
        self, request: urllib.request.Request
    ) -> tuple[dict[str, Any], dict[str, str]]:
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8")
                headers = dict(response.headers.items())
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", errors="replace")[:500]
            except Exception:  # pragma: no cover - best-effort diagnostics
                detail = ""
            msg = (
                f"OXR request failed at {self.base_url} (HTTP {exc.code}). "
                f"{_server_hint()} {detail}".strip()
            )
            raise RuntimeError(msg) from exc
        except urllib.error.URLError as exc:
            msg = (
                f"OXR request failed at {self.base_url}. {_server_hint()} {exc}".strip()
            )
            raise RuntimeError(msg) from exc

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            msg = f"OXR returned non-JSON response from {self.base_url}: {raw[:200]!r}"
            raise RuntimeError(msg) from exc
        if not isinstance(parsed, dict):
            msg = f"OXR returned an unexpected JSON payload: {type(parsed).__name__}"
            raise RuntimeError(msg)
        return parsed, headers

    # ------------------------------------------------------------------
    # Recognition
    # ------------------------------------------------------------------
    def recognize(self, image: np.ndarray) -> RecognitionResult:
        """Recognize text in a single text-region crop."""
        backend = self._resolved_backend()
        text = self._post_parse(image) if backend == "parse" else self._post_chat(image)
        text = _strip_markup(text)
        return RecognitionResult(text=text, confidence=0.9 if text else 0.0)


def _server_hint() -> str:
    return (
        "Start an OXR document-parsing API (model card default: port 8000, "
        "endpoint /v1/parse/sync), a vLLM chat server (port 8080), or use the "
        f"free API at {_FREE_API_URL}."
    )


def _encode_multipart(
    *,
    fields: list[tuple[str, str]],
    file_field: str,
    filename: str,
    file_bytes: bytes,
    file_content_type: str,
) -> tuple[bytes, str]:
    """Encode a form with a single file part using the standard library."""
    boundary = f"----EmbeddingOCRBoundary{secrets.token_hex(12)}"
    buffer = io.BytesIO()

    def write(text: str) -> None:
        buffer.write(text.encode("utf-8"))

    for name, value in fields:
        write(f"--{boundary}\r\n")
        write(f'Content-Disposition: form-data; name="{name}"\r\n\r\n')
        write(f"{value}\r\n")

    write(f"--{boundary}\r\n")
    write(
        f'Content-Disposition: form-data; name="{file_field}"; '
        f'filename="{filename}"\r\n'
    )
    write(f"Content-Type: {file_content_type}\r\n\r\n")
    buffer.write(file_bytes)
    write("\r\n")
    write(f"--{boundary}--\r\n")

    return buffer.getvalue(), f"multipart/form-data; boundary={boundary}"


def _extract_parse_text(body: dict[str, Any]) -> str:
    """Pull text out of a SoMark/OXR ``/parse/sync`` response."""
    data = body.get("data") or {}
    outputs = data.get("outputs") or {}
    if not isinstance(outputs, dict):
        return ""

    markdown = outputs.get("markdown")
    if isinstance(markdown, str) and markdown.strip():
        return markdown.strip()

    structured = outputs.get("json")
    if isinstance(structured, dict):
        return _extract_json_text(structured)

    return ""


def _extract_json_text(payload: dict[str, Any]) -> str:
    pages = payload.get("pages")
    if not isinstance(pages, list):
        return ""
    chunks: list[str] = []
    for page in pages:
        if not isinstance(page, dict):
            continue
        blocks = page.get("blocks")
        if not isinstance(blocks, list):
            continue
        ordered = sorted(
            (block for block in blocks if isinstance(block, dict)),
            key=lambda block: block.get("idx", 0),
        )
        for block in ordered:
            content = block.get("content")
            if isinstance(content, str) and content.strip():
                chunks.append(content.strip())
    return "\n".join(chunks).strip()
