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
        from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]

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
