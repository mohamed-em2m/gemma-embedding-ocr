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

from typing import Protocol, runtime_checkable

import numpy as np
import torch


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
        batch_size: int = 16,
        dtype: str = "float32",
    ) -> None:
        """Initialise the EmbeddingGemma2 backend.

        Args:
            dim: Active Matryoshka dimension. One of 768, 512, 256, 128.
            device: Device to use. "auto" selects CUDA if available else CPU.
            batch_size: Batch size for encoding. Use <= 16 on T4/15GB (bfloat16).
            dtype: Weight dtype. "bfloat16" recommended on GPU; "float32" on CPU.
        """
        if dim not in (768, 512, 256, 128):
            msg = f"dim must be one of 768, 512, 256, 128; got {dim}"
            raise ValueError(msg)

        self.dim = dim
        self.batch_size = batch_size
        self._cache: dict[bytes, np.ndarray] = {}

        # Resolve device
        if device == "auto":
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = device

        # Lazy import to avoid requiring ST at module import time
        from sentence_transformers import (
            SentenceTransformer,  # type: ignore[import-untyped]
        )

        model_kwargs: dict[str, object] = {}
        if self._device == "cuda":
            model_kwargs["torch_dtype"] = torch.bfloat16

        self._model = SentenceTransformer(
            self.name,
            model_kwargs=model_kwargs,
            device=self._device,
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
            encoded: np.ndarray = self._model.encode(
                to_encode_pil,  # type: ignore[arg-type]
                batch_size=self.batch_size,
                truncate_dim=self.dim,
                normalize_embeddings=True,
                show_progress_bar=False,
            ).astype(np.float32)

            for local_i, global_i in enumerate(to_encode_indices):
                vec = encoded[local_i]
                results[global_i] = vec
                if use_cache:
                    self._cache[keys[global_i]] = vec

        return np.array(results, dtype=np.float32)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Embed a list of text strings using the 'Document' task prefix."""
        result: np.ndarray = self._model.encode(
            texts,
            batch_size=self.batch_size,
            truncate_dim=self.dim,
            normalize_embeddings=True,
            prompt_name="Document",
            show_progress_bar=False,
        )
        return result.astype(np.float32)


def _to_numpy(value: object) -> np.ndarray:
    """Convert a torch tensor / array-like to a float32 numpy array."""
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().float().numpy()
    return np.asarray(value, dtype=np.float32)


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

        self._model = AutoModel.from_pretrained(
            self.name,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
        )
        self._model.to(self._device)
        self._model.eval()

    def _encode(self, method: str, **kwargs: object) -> np.ndarray:
        """Call a remote ``encode_*`` method with truncate_dim when supported."""
        encoder = getattr(self._model, method)
        try:
            result = encoder(truncate_dim=self.dim, return_numpy=True, **kwargs)
        except TypeError:
            result = encoder(**kwargs)
        return _truncate_and_normalize(_to_numpy(result), self.dim)

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
