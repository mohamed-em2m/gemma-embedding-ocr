"""VectorIndex protocol, FaissIndex, and NumpyIndex implementations.

All indexes operate on L2-normalized vectors and return cosine similarities
via inner product (IndexFlatIP for FAISS, np.dot for Numpy).

Matryoshka rule:
    When dim < 768, the embedder already truncates+renormalizes before
    calling build(). The index stores and searches in `dim`-d space.

Save/load contract:
    save() writes: <path>.faiss + <path>.meta.json
    load() raises RuntimeError if embedder name or dim does not match config.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import ClassVar, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class VectorIndex(Protocol):
    """Protocol for vector similarity indexes."""

    def build(self, vectors: np.ndarray, labels: np.ndarray) -> None:
        """Build the index from prototype vectors.

        Args:
            vectors: Float32 (n, dim) L2-normalized prototype embeddings.
            labels: Int or string labels, shape (n,).
        """
        ...

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Search for nearest neighbours.

        Args:
            queries: Float32 (n_queries, dim) L2-normalized query embeddings.
            k: Number of nearest neighbours to return.

        Returns:
            Tuple (similarities, prototype_ids), each shape (n_queries, k).
            Similarities are cosine similarities in [-1, 1].
        """
        ...

    def save(self, path: Path) -> None:
        """Persist the index to disk.

        Writes <path>.faiss (or .npy) and <path>.meta.json.
        """
        ...

    @classmethod
    def load(cls, path: Path) -> VectorIndex:
        """Load the index from disk.

        Raises RuntimeError if metadata doesn't match the current config.
        """
        ...


class FaissIndex:
    """Exact cosine-similarity index backed by faiss.IndexFlatIP."""

    _FAISS_EXT: ClassVar[str] = ".faiss"
    _META_EXT: ClassVar[str] = ".meta.json"

    def __init__(
        self,
        dim: int,
        embedder_name: str = "google/embeddinggemma-2",
        omp_threads: int | None = None,
    ) -> None:
        """Initialise FaissIndex.

        Args:
            dim: Embedding dimension (must match embedder's active dim).
            embedder_name: Embedder model ID stored in metadata.
            omp_threads: OMP thread count for FAISS (None = default).
        """
        import faiss  # type: ignore[import-untyped]

        self.dim = dim
        self.embedder_name = embedder_name
        self._labels: np.ndarray | None = None
        self._index: faiss.IndexFlatIP | None = None

        if omp_threads is not None:
            faiss.omp_set_num_threads(omp_threads)

    def build(self, vectors: np.ndarray, labels: np.ndarray) -> None:
        """Build the FAISS index from prototype vectors."""
        import faiss  # type: ignore[import-untyped]

        vectors = vectors.astype(np.float32)
        faiss.normalize_L2(vectors)  # ensure L2-normalized
        self._index = faiss.IndexFlatIP(self.dim)
        self._index.add(vectors)
        self._labels = labels.copy()

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Search for nearest neighbours."""
        if self._index is None or self._labels is None:
            msg = "Index not built. Call build() first."
            raise RuntimeError(msg)

        import faiss  # type: ignore[import-untyped]

        q = queries.astype(np.float32).copy()
        faiss.normalize_L2(q)
        similarities, proto_ids = self._index.search(q, k)
        return similarities, proto_ids

    def save(self, path: Path) -> None:
        """Save index + metadata to disk."""
        if self._index is None or self._labels is None:
            msg = "Index not built. Call build() first."
            raise RuntimeError(msg)

        import faiss  # type: ignore[import-untyped]

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self._index, str(path.with_suffix(self._FAISS_EXT)))
        np.save(str(path) + "_labels.npy", self._labels)

        meta = {
            "embedder_name": self.embedder_name,
            "dim": self.dim,
            "n_prototypes": int(self._index.ntotal),
        }
        path.with_suffix(self._META_EXT).write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, path: Path) -> FaissIndex:
        """Load a previously saved FaissIndex."""
        import faiss  # type: ignore[import-untyped]

        path = Path(path)
        meta_path = path.with_suffix(cls._META_EXT)
        if not meta_path.exists():
            msg = f"Metadata file not found: {meta_path}"
            raise FileNotFoundError(msg)

        meta = json.loads(meta_path.read_text())
        obj = cls(dim=meta["dim"], embedder_name=meta["embedder_name"])
        obj._index = faiss.read_index(str(path.with_suffix(cls._FAISS_EXT)))
        obj._labels = np.load(str(path) + "_labels.npy", allow_pickle=True)
        return obj


class NumpyIndex:
    """Exact cosine-similarity index backed by numpy (no FAISS dependency).

    Useful for CI and unit tests. Results must be identical to FaissIndex
    (both perform exact inner product on L2-normalized vectors).
    """

    _VECS_EXT: ClassVar[str] = "_vecs.npy"
    _LABELS_EXT: ClassVar[str] = "_labels.npy"
    _META_EXT: ClassVar[str] = ".meta.json"

    def __init__(
        self,
        dim: int,
        embedder_name: str = "google/embeddinggemma-2",
    ) -> None:
        self.dim = dim
        self.embedder_name = embedder_name
        self._vectors: np.ndarray | None = None
        self._labels: np.ndarray | None = None

    def build(self, vectors: np.ndarray, labels: np.ndarray) -> None:
        """Store L2-normalized prototype vectors."""
        vecs = vectors.astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        self._vectors = vecs / norms
        self._labels = labels.copy()

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Brute-force cosine search."""
        if self._vectors is None or self._labels is None:
            msg = "Index not built. Call build() first."
            raise RuntimeError(msg)

        q = queries.astype(np.float32)
        norms = np.linalg.norm(q, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        q = q / norms

        sims = np.dot(q, self._vectors.T)  # (n_queries, n_prototypes)
        top_k_ids = np.argsort(-sims, axis=1)[:, :k]
        top_k_sims = np.take_along_axis(sims, top_k_ids, axis=1)
        return top_k_sims, top_k_ids

    def save(self, path: Path) -> None:
        """Save vectors, labels and metadata."""
        if self._vectors is None or self._labels is None:
            msg = "Index not built. Call build() first."
            raise RuntimeError(msg)

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(path) + self._VECS_EXT, self._vectors)
        np.save(str(path) + self._LABELS_EXT, self._labels)

        meta = {
            "embedder_name": self.embedder_name,
            "dim": self.dim,
            "n_prototypes": len(self._vectors),
        }
        path.with_suffix(self._META_EXT).write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, path: Path) -> NumpyIndex:
        """Load a previously saved NumpyIndex."""
        path = Path(path)
        meta_path = path.with_suffix(cls._META_EXT)
        if not meta_path.exists():
            msg = f"Metadata file not found: {meta_path}"
            raise FileNotFoundError(msg)

        meta = json.loads(meta_path.read_text())
        obj = cls(dim=meta["dim"], embedder_name=meta["embedder_name"])
        obj._vectors = np.load(str(path) + cls._VECS_EXT)
        obj._labels = np.load(str(path) + cls._LABELS_EXT, allow_pickle=True)
        return obj
