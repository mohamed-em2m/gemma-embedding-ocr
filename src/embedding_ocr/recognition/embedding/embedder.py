"""Embedder protocol and EmbeddingGemma2 backend.

Provides a standard interface for embedding images and texts into a
shared latent space, following the Embedder protocol defined in PLAN.md.

EmbeddingGemma2 facts (verified Phase 0):
  - Model ID: google/embeddinggemma-2
  - Architecture: ~740M params (270M text + 170M vision + 300M audio)
  - Native dim: 768; Matryoshka dims: 768, 512, 256, 128
  - License: Apache 2.0
  - Library: sentence-transformers >= 6.1.0
  - Images passed without prompt prefix; texts use prompt_name="Document"
  - Re-normalization required after Matryoshka truncation
  - GPU memory: ~1432 MB load footprint (bfloat16); ~13 GB peak (T4, batch=16)
  - Max safe batch size on T4 (bfloat16): 16 (batch=32 OOMs)
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

import numpy as np
import torch

from embedding_ocr.utils.gpu import (
    AUTO_BATCH_SIZE,
    get_cached_batch_size,
    is_oom_error,
    recommend_batch_size,
    resolve_device,
)

_LOGGER = logging.getLogger(__name__)


@runtime_checkable
class Embedder(Protocol):
    """Protocol for all embedding backends.

    Both embed_images and embed_texts must return float32 arrays of shape
    (n, dim), already L2-normalized and truncated to `dim`.
    """

    name: str
    dim: int  # active Matryoshka dimension

    def embed_images(self, images: list[np.ndarray]) -> np.ndarray:
        """Embed a list of images.

        Args:
            images: List of uint8 numpy arrays with shape (H, W, 3).

        Returns:
            Float32 array of shape (n, dim), L2-normalized.
        """
        ...

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Embed a list of text strings.

        Args:
            texts: List of text strings to embed.

        Returns:
            Float32 array of shape (n, dim), L2-normalized.
        """
        ...


class EmbeddingGemma2:
    """EmbeddingGemma 2 backend via sentence-transformers.

    Verified loading code (Phase 0):
        model = SentenceTransformer(
            "google/embeddinggemma-2",
            model_kwargs={"torch_dtype": torch.bfloat16},  # GPU only
            device="cuda",  # or "cpu"
        )

    Requires sentence-transformers >= 6.1.0 (6.2.0 confirmed working).
    sentence-transformers 5.7.0 is broken (missing normalize module).
    """

    name: str = "google/embeddinggemma-2"

    def __init__(
        self,
        dim: int = 768,
        device: str = "auto",
        batch_size: int = AUTO_BATCH_SIZE,
        dtype: str = "float32",
        auto_oom_fallback: bool = True,
    ) -> None:
        """Initialise the EmbeddingGemma2 backend.

        Args:
            dim: Active Matryoshka dimension. One of 768, 512, 256, 128.
            device: Device to use. ``auto``/``cuda`` selects the least-used GPU
                with enough free memory; explicit values such as ``cuda:1``
                are respected.
            batch_size: Images/texts per forward pass. ``0`` auto-selects from
                the tuned cache or free-memory heuristics, with OOM fallback.
            dtype: Weight dtype. ``bfloat16`` is used on GPU automatically.
            auto_oom_fallback: Halve the batch and retry on CUDA OOM.
        """
        if dim not in (768, 512, 256, 128):
            msg = f"dim must be one of 768, 512, 256, 128; got {dim}"
            raise ValueError(msg)

        self.dim = dim
        self._cache: dict[bytes, np.ndarray] = {}
        self._device = resolve_device(device)
        self._auto_oom_fallback = auto_oom_fallback

        # Lazy import to avoid requiring ST at module import time
        from sentence_transformers import (
            SentenceTransformer,  # type: ignore[import-untyped]
        )

        model_kwargs: dict[str, object] = {}
        if self._device.startswith("cuda"):
            model_kwargs["torch_dtype"] = torch.bfloat16
            torch.cuda.set_device(torch.device(self._device))

        self._model = SentenceTransformer(
            self.name,
            model_kwargs=model_kwargs,
            device=self._device,
        )
        if batch_size <= AUTO_BATCH_SIZE:
            cached = get_cached_batch_size(
                self._device,
                model_id=self.name,
                dim=self.dim,
            )
            self.batch_size = (
                cached
                if cached is not None
                else recommend_batch_size(
                    self._device,
                    model_id=self.name,
                    dim=self.dim,
                )
            )
        else:
            self.batch_size = max(1, batch_size)
        _LOGGER.info(
            "EmbeddingGemma2 device=%s batch_size=%d",
            self._device,
            self.batch_size,
        )

    def embed_images(
        self, images: list[np.ndarray], use_cache: bool = True
    ) -> np.ndarray:
        """Embed a list of numpy image arrays (H, W, 3) uint8.

        Converts to PIL.Image internally; model handles preprocessing.
        Utilizes fast memory caching and batched tensor encoding.
        """
        if not images:
            return np.empty((0, self.dim), dtype=np.float32)

        import hashlib

        from PIL import Image  # type: ignore[import-untyped]

        results: list[np.ndarray | None] = [None] * len(images)
        to_encode_indices: list[int] = []
        to_encode_pil: list[Image.Image] = []
        keys: list[bytes] = []

        if use_cache:
            for idx, img in enumerate(images):
                # Fast MD5 hash on byte buffer
                key = hashlib.md5(img.tobytes()).digest()
                keys.append(key)
                if key in self._cache:
                    results[idx] = self._cache[key]
                else:
                    to_encode_indices.append(idx)
                    to_encode_pil.append(Image.fromarray(img))
        else:
            to_encode_indices = list(range(len(images)))
            to_encode_pil = [Image.fromarray(img) for img in images]

        if to_encode_pil:
            encoded = self._encode(to_encode_pil)  # type: ignore[arg-type]

            for local_i, global_i in enumerate(to_encode_indices):
                vec = encoded[local_i]
                results[global_i] = vec
                if use_cache:
                    self._cache[keys[global_i]] = vec

        return np.array(results, dtype=np.float32)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Embed a list of text strings using the 'Document' task prefix."""
        return self._encode(texts, prompt_name="Document")

    def _encode(self, inputs: list[Any], **kwargs: Any) -> np.ndarray:
        """Encode a batch, halving the batch size on CUDA OOM."""
        batch_size = max(1, self.batch_size)
        while True:
            try:
                encoded = self._model.encode(
                    inputs,
                    batch_size=batch_size,
                    truncate_dim=self.dim,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    **kwargs,
                )
            except Exception as exc:
                if (
                    not is_oom_error(exc)
                    or not self._device.startswith("cuda")
                    or batch_size == 1
                    or not self._auto_oom_fallback
                ):
                    raise
                batch_size = max(1, batch_size // 2)
                torch.cuda.empty_cache()
                _LOGGER.warning(
                    "CUDA OOM while encoding; retrying with batch_size=%d",
                    batch_size,
                )
                continue
            self.batch_size = batch_size
            return np.asarray(encoded, dtype=np.float32)


def _feature_tensor(value: object) -> object:
    """Extract the embedding tensor from a transformers model output.

    Transformers 5.x returns ``BaseModelOutputWithPooling`` from SigLIP's
    ``get_image_features`` / ``get_text_features`` instead of a plain tensor.
    Prefer the projected embedding fields, then the pooler output.
    """
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)) and value:
        return _feature_tensor(value[0])
    if isinstance(value, dict):
        for key in (
            "image_embeds",
            "text_embeds",
            "embeddings",
            "image_features",
            "text_features",
            "single_vec_emb",
            "pooler_output",
            "last_hidden_state",
        ):
            candidate = value.get(key)
            if candidate is not None:
                return candidate
    for attr in (
        "image_embeds",
        "text_embeds",
        "embeddings",
        "image_features",
        "text_features",
        "single_vec_emb",
        "pooler_output",
        "last_hidden_state",
    ):
        candidate = getattr(value, attr, None)
        if candidate is not None:
            return candidate
    msg = (
        "Could not extract an embedding tensor from "
        f"{type(value).__name__}; expected a tensor or a model output with "
        "image_embeds/text_embeds/embeddings/pooler_output/last_hidden_state."
    )
    raise TypeError(msg)


def _to_numpy(value: object) -> np.ndarray:
    """Convert a torch tensor / model output / array-like to float32 numpy."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().float().numpy()
    if isinstance(value, np.ndarray):
        return np.asarray(value, dtype=np.float32)
    if isinstance(value, (tuple, list)):
        if not value:
            return np.empty((0, 0), dtype=np.float32)
        return np.stack([_to_numpy(item) for item in value])
    if isinstance(value, dict) or hasattr(value, "pooler_output"):
        return _to_numpy(_feature_tensor(value))
    return np.asarray(value, dtype=np.float32)


def _default_rope_parameters(
    config: object,
    device: torch.device | str | None = None,
    seq_len: int | None = None,
    layer_type: str | None = None,
    **kwargs: object,
) -> tuple[torch.Tensor, float]:
    """Compute ordinary RoPE inverse frequencies.

    Transformers 5.x removed the ``"default"`` entry from
    ``ROPE_INIT_FUNCTIONS``. Jina v4's bundled Qwen2.5-VL code still looks it
    up by name, which raises ``KeyError: 'default'`` on model construction.
    """
    del seq_len, layer_type, kwargs
    rope_parameters = getattr(config, "rope_parameters", None)
    if not isinstance(rope_parameters, dict):
        rope_parameters = getattr(config, "rope_scaling", None)
    if not isinstance(rope_parameters, dict):
        rope_parameters = {}

    base_value = (
        rope_parameters.get("rope_theta")
        or getattr(config, "rope_theta", None)
        or 10000.0
    )
    base = float(base_value)
    partial = rope_parameters.get(
        "partial_rotary_factor",
        getattr(config, "partial_rotary_factor", 1.0),
    )
    partial = float(partial or 1.0)

    head_dim = getattr(config, "head_dim", None)
    if head_dim is None:
        config_any: Any = config
        head_dim = int(config_any.hidden_size) // int(config_any.num_attention_heads)
    dim = int(head_dim * partial)
    inv_freq = 1.0 / (
        base
        ** (torch.arange(0, dim, 2, dtype=torch.int64, device=device).float() / dim)
    )
    return inv_freq, 1.0


def _ensure_default_rope_init() -> None:
    """Restore the legacy ``"default"`` RoPE entry when transformers drops it."""
    try:
        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
    except ImportError:  # pragma: no cover - transformers is an optional runtime dep
        return
    ROPE_INIT_FUNCTIONS.setdefault("default", _default_rope_parameters)


def _patch_jina_remote_code(model_id: str) -> None:
    """Adapt Jina v4's bundled Qwen code to transformers 5.x.

    Two compatibility issues are handled before ``from_pretrained``:

    1. The remote model declares ``_tied_weights_keys`` as a list, while
       transformers 5 expects a mapping during ``post_init``.
    2. The remote processor overrides ``__init__`` as ``(*args, **kwargs)``.
       transformers 5 infers sub-processors from ``cls.__dict__`` when the
       signature is opaque, so inherited ``*_class`` attributes are invisible
       and the processor is built with ``tokenizer=None``.  Copy those class
       attributes onto the remote processor so its tokenizer/image processor
       load correctly.
    """
    try:
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
    except ImportError:  # pragma: no cover - transformers is optional
        return

    try:
        model_class = get_class_from_dynamic_module(
            "modeling_jina_embeddings_v4.JinaEmbeddingsV4Model", model_id
        )
    except Exception:  # pragma: no cover - offline or remote-code failure
        model_class = None
    if model_class is not None and isinstance(
        getattr(model_class, "_tied_weights_keys", None), list
    ):
        model_class_any: Any = model_class
        model_class_any._tied_weights_keys = {}

    try:
        processor_class = get_class_from_dynamic_module(
            "modeling_jina_embeddings_v4.JinaEmbeddingsV4Processor", model_id
        )
    except Exception:  # pragma: no cover - offline or remote-code failure
        return
    for attribute in (
        "image_processor_class",
        "tokenizer_class",
        "video_processor_class",
    ):
        value = getattr(processor_class, attribute, None)
        if value is not None and attribute not in processor_class.__dict__:
            setattr(processor_class, attribute, value)


def _truncate_and_normalize(vectors: np.ndarray, dim: int) -> np.ndarray:
    """Apply Matryoshka truncation and L2 re-normalization.

    Args:
        vectors: Array of shape (n, native_dim).
        dim: Active dimension to keep.

    Returns:
        Float32 array of shape (n, dim) with unit L2 norm.
    """
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim == 1:
        array = array[None, :]
    if dim < array.shape[1]:
        array = array[:, :dim]
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.maximum(norms, 1e-12)


class SiglipEmbedder:
    """SigLIP SO400M image/text embedder via transformers.

    Model: ``google/siglip-so400m-patch14-384``.  SigLIP natively produces
    1152-d embeddings; Matryoshka truncation is not part of its training, but
    slicing + re-normalizing is exposed for size comparisons.

    Note:
        SigLIP uses separate image and text towers trained with a sigmoid
        contrastive loss.  Cross-modal retrieval works, but it is not trained
        with a Matryoshka objective, so truncated dimensions are approximate.
    """

    name = "google/siglip-so400m-patch14-384"
    native_dim = 1152

    def __init__(
        self,
        dim: int = 1152,
        device: str = "auto",
        batch_size: int = 16,
        dtype: str = "float32",
        model_id: str | None = None,
    ) -> None:
        """Load the processor and model.

        Args:
            dim: Active embedding dimension (<= 1152).
            device: ``auto`` selects CUDA when available, else CPU.
            batch_size: Images/texts per forward pass.
            dtype: ``float32`` or ``bfloat16`` (GPU recommended).
            model_id: Override the Hugging Face model id.
        """
        if dim <= 0 or dim > self.native_dim:
            msg = f"dim must be in [1, {self.native_dim}]; got {dim}"
            raise ValueError(msg)

        import torch
        from transformers import AutoModel, AutoProcessor

        if model_id is not None:
            self.name = model_id
        self.dim = dim
        self.batch_size = batch_size
        self._device = (
            "cuda" if device == "auto" and torch.cuda.is_available() else device
        )
        if self._device == "auto":
            self._device = "cpu"
        torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float32

        self._processor = AutoProcessor.from_pretrained(self.name)
        self._model = (
            AutoModel.from_pretrained(self.name, torch_dtype=torch_dtype)
            .to(self._device)
            .eval()
        )

    def _features_to_numpy(self, features: object) -> np.ndarray:
        array = _to_numpy(features)
        if array.ndim == 3:
            array = array.mean(axis=1)
        return _truncate_and_normalize(array, self.dim)

    def embed_images(self, images: list[np.ndarray]) -> np.ndarray:
        """Embed uint8 (H, W, 3) images into (n, dim) unit vectors."""
        import torch
        from PIL import Image

        if not images:
            return np.empty((0, self.dim), dtype=np.float32)

        batches: list[np.ndarray] = []
        for start in range(0, len(images), self.batch_size):
            batch = [
                Image.fromarray(img).convert("RGB")
                for img in images[start : start + self.batch_size]
            ]
            inputs = self._processor(images=batch, return_tensors="pt").to(self._device)
            with torch.no_grad():
                features = self._model.get_image_features(**inputs)
            batches.append(self._features_to_numpy(features))
        return np.concatenate(batches, axis=0).astype(np.float32)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Embed text strings into (n, dim) unit vectors."""
        import torch

        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)

        batches: list[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            inputs = self._processor(
                text=batch,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).to(self._device)
            with torch.no_grad():
                features = self._model.get_text_features(**inputs)
            batches.append(self._features_to_numpy(features))
        return np.concatenate(batches, axis=0).astype(np.float32)


class JinaEmbeddingsV4:
    """Jina Embeddings v4 multimodal embedder via transformers.

    Model: ``jinaai/jina-embeddings-v4`` (Qwen2.5-VL-3B base).  Dense
    embeddings are 2048-d, Matryoshka-truncatable to 1024/512/256/128.

    The model card requires ``transformers>=4.52`` and ``peft>=0.15.2``; the
    remote code exposes ``encode_text`` and ``encode_image``.  ``flash_attn``
    is optional and never forced here.
    """

    name = "jinaai/jina-embeddings-v4"
    native_dim = 2048
    matryoshka_dims = (128, 256, 512, 1024, 2048)

    def __init__(
        self,
        dim: int = 2048,
        device: str = "auto",
        batch_size: int = 8,
        dtype: str = "bfloat16",
        task: str = "retrieval",
        text_prompt: str = "passage",
        model_id: str | None = None,
    ) -> None:
        """Load the model.

        Args:
            dim: Active embedding dimension; one of 128, 256, 512, 1024, 2048.
            device: ``auto`` selects CUDA when available, else CPU.
            batch_size: Images/texts per forward pass.
            dtype: ``bfloat16`` (default) or ``float32``.
            task: ``retrieval``, ``text-matching`` or ``code``.
            text_prompt: Prompt name for text encoding (``passage``/``query``).
            model_id: Override the Hugging Face model id.
        """
        if dim not in self.matryoshka_dims:
            msg = f"dim must be one of {self.matryoshka_dims} for Jina v4; got {dim}"
            raise ValueError(msg)

        import torch
        from transformers import AutoModel

        if model_id is not None:
            self.name = model_id
        self.dim = dim
        self.batch_size = batch_size
        self.task = task
        self.text_prompt = text_prompt
        self._device = (
            "cuda" if device == "auto" and torch.cuda.is_available() else device
        )
        if self._device == "auto":
            self._device = "cpu"
        torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float32
        if self._device == "cpu" and dtype == "bfloat16":
            torch_dtype = torch.float32

        _ensure_default_rope_init()
        _patch_jina_remote_code(self.name)
        try:
            self._model = AutoModel.from_pretrained(
                self.name,
                trust_remote_code=True,
                torch_dtype=torch_dtype,
                attn_implementation="sdpa",
            )
        except ImportError as exc:
            msg = (
                "Jina v4 remote code requires the optional `peft` and "
                "`torchvision` dependencies. Install them with "
                "`uv pip install --python .venv/bin/python -e '.[jina]'` (or `uv pip install peft torchvision`)."
            )
            raise ImportError(msg) from exc
        self._model.to(self._device)
        self._model.eval()

    def _encode(self, method: str, **kwargs: object) -> np.ndarray:
        """Call a remote ``encode_*`` method with the supported kwargs."""
        import inspect

        encoder = getattr(self._model, method)
        try:
            parameters: Mapping[str, inspect.Parameter] = inspect.signature(
                encoder
            ).parameters
        except (TypeError, ValueError):  # pragma: no cover - exotic callables
            parameters = {}
        accepts_extra = not parameters or any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )
        call_kwargs = {
            key: value
            for key, value in kwargs.items()
            if accepts_extra or key in parameters
        }
        if accepts_extra or "return_numpy" in parameters:
            call_kwargs["return_numpy"] = True
        if accepts_extra or "truncate_dim" in parameters:
            call_kwargs["truncate_dim"] = self.dim
        return _truncate_and_normalize(_to_numpy(encoder(**call_kwargs)), self.dim)

    def embed_images(self, images: list[np.ndarray]) -> np.ndarray:
        """Embed uint8 (H, W, 3) images into (n, dim) unit vectors."""
        from PIL import Image

        if not images:
            return np.empty((0, self.dim), dtype=np.float32)
        pil_images = [Image.fromarray(img).convert("RGB") for img in images]
        return self._encode(
            "encode_image",
            images=pil_images,
            task=self.task,
            batch_size=self.batch_size,
        )

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Embed text strings into (n, dim) unit vectors."""
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        return self._encode(
            "encode_text",
            texts=texts,
            task=self.task,
            prompt_name=self.text_prompt,
            batch_size=self.batch_size,
        )
