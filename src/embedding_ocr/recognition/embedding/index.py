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
from typing import Any, ClassVar, Protocol, runtime_checkable

import numpy as np

_FAISS_BACKEND_ALIASES: dict[str, str] = {
    "faiss": "flat_ip",
    "faiss_flat_ip": "flat_ip",
    "flat_ip": "flat_ip",
    "faiss_sq8": "sq8",
    "sq8": "sq8",
    "faiss_ivf_sq8": "ivf_sq8",
    "ivf_sq8": "ivf_sq8",
}


def normalize_faiss_backend(backend: str) -> str:
    """Return the canonical FAISS backend name."""
    key = backend.strip().lower()
    try:
        return _FAISS_BACKEND_ALIASES[key]
    except KeyError:
        expected = ", ".join(sorted(_FAISS_BACKEND_ALIASES))
        msg = f"Unknown FAISS index backend {backend!r}; expected one of: {expected}"
        raise ValueError(msg) from None


def is_faiss_backend(backend: str) -> bool:
    """Return whether ``backend`` names a FAISS index variant."""
    return backend.strip().lower() in _FAISS_BACKEND_ALIASES


def _infer_faiss_backend(index: Any) -> str:
    name = type(index).__name__.lower()
    if "ivf" in name and "scalarquantizer" in name:
        return "ivf_sq8"
    if "scalarquantizer" in name:
        return "sq8"
    return "flat_ip"


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
    """Cosine-similarity index backed by FAISS.

    Supports three storage modes while keeping inner-product (cosine) search:

    * ``flat_ip``: exact float32 ``IndexFlatIP`` (baseline).
    * ``sq8``: exhaustive 8-bit scalar quantization (4x smaller vectors).
    * ``ivf_sq8``: IVF coarse quantizer + 8-bit scalar quantization
      (smaller and sub-linear search; approximate).

    Vectors and queries are L2-normalized before indexing/searching. Metadata
    records the backend and quantization parameters so ``load`` can restore the
    correct FAISS index type.
    """

    _FAISS_EXT: ClassVar[str] = ".faiss"
    _META_EXT: ClassVar[str] = ".meta.json"

    def __init__(
        self,
        dim: int,
        embedder_name: str = "google/embeddinggemma-2",
        omp_threads: int | None = None,
        backend: str = "ivf_sq8",
        nlist: int = 1024,
        nprobe: int = 16,
        quantizer_bits: int = 8,
        train_size: int | None = None,
    ) -> None:
        """Initialise FaissIndex.

        Args:
            dim: Embedding dimension (must match embedder's active dim).
            embedder_name: Embedder model ID stored in metadata.
            omp_threads: OMP thread count for FAISS (None = default).
            backend: ``flat_ip``, ``sq8``, or ``ivf_sq8`` (aliases accepted).
            nlist: Requested IVF cluster count; clamped to a feasible value
                from the number of prototypes.
            nprobe: IVF clusters probed per query (clamped to effective nlist).
            quantizer_bits: Scalar quantizer width; only 8 is supported.
            train_size: Optional maximum number of vectors used to train the
                quantizer/coarse centroids. ``None`` uses all vectors.
        """
        import faiss  # type: ignore[import-untyped]

        if quantizer_bits != 8:
            msg = "Only 8-bit scalar quantization is supported (quantizer_bits=8)"
            raise ValueError(msg)

        self.dim = dim
        self.embedder_name = embedder_name
        self.backend = normalize_faiss_backend(backend)
        self.nlist = max(1, int(nlist))
        self.nprobe = max(1, int(nprobe))
        self.quantizer_bits = quantizer_bits
        self.train_size = None if train_size is None else max(0, int(train_size))
        self._labels: np.ndarray | None = None
        self._index: Any | None = None

        if omp_threads is not None:
            faiss.omp_set_num_threads(omp_threads)

    def _new_index(self, nlist: int) -> Any:
        import faiss  # type: ignore[import-untyped]

        if self.backend == "flat_ip":
            return faiss.IndexFlatIP(self.dim)
        if self.backend == "sq8":
            return faiss.IndexScalarQuantizer(
                self.dim,
                faiss.ScalarQuantizer.QT_8bit,
                faiss.METRIC_INNER_PRODUCT,
            )
        quantizer = faiss.IndexFlatIP(self.dim)
        return faiss.IndexIVFScalarQuantizer(
            quantizer,
            self.dim,
            nlist,
            faiss.ScalarQuantizer.QT_8bit,
            faiss.METRIC_INNER_PRODUCT,
        )

    def _effective_nlist(self, n: int) -> int:
        # FAISS recommends roughly 39 training points per centroid. Clamp so
        # small vocabularies (e.g. the character index) can still train.
        max_feasible = max(1, n // 39)
        return max(1, min(self.nlist, max_feasible))

    def _training_vectors(self, vectors: np.ndarray) -> np.ndarray:
        if self.backend == "flat_ip":
            return vectors
        if self.train_size is None or self.train_size <= 0:
            return vectors
        if self.train_size >= len(vectors):
            return vectors
        rng = np.random.default_rng(0)
        indices = rng.choice(len(vectors), size=self.train_size, replace=False)
        return vectors[indices]

    def build(self, vectors: np.ndarray, labels: np.ndarray) -> None:
        """Build the FAISS index from prototype vectors."""
        import faiss  # type: ignore[import-untyped]

        vectors = np.array(vectors, dtype=np.float32, copy=True)
        if vectors.ndim != 2 or vectors.shape[0] == 0:
            msg = f"Expected non-empty (n, dim) vectors, got shape {vectors.shape}"
            raise ValueError(msg)
        if vectors.shape[1] != self.dim:
            msg = f"Vector dim {vectors.shape[1]} does not match index dim {self.dim}"
            raise ValueError(msg)
        if labels.ndim != 1 or len(labels) != len(vectors):
            msg = f"labels must have shape ({len(vectors)},), got {labels.shape}"
            raise ValueError(msg)

        faiss.normalize_L2(vectors)  # ensure L2-normalized

        nlist = self._effective_nlist(len(vectors))
        index = self._new_index(nlist)
        if self.backend != "flat_ip":
            train_vectors = np.ascontiguousarray(self._training_vectors(vectors))
            if train_vectors.shape[0] == 0:
                msg = "Cannot train the quantizer with zero vectors"
                raise ValueError(msg)
            index.train(train_vectors)
            if not index.is_trained:
                msg = f"FAISS index {self.backend!r} failed to train"
                raise RuntimeError(msg)

        index.add(np.ascontiguousarray(vectors))
        if self.backend == "ivf_sq8":
            index.nprobe = max(1, min(self.nprobe, nlist))
        self._index = index
        self._labels = labels.copy()

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Search for nearest neighbours."""
        if self._index is None or self._labels is None:
            msg = "Index not built. Call build() first."
            raise RuntimeError(msg)
        if k <= 0:
            msg = f"k must be positive, got {k}"
            raise ValueError(msg)

        import faiss  # type: ignore[import-untyped]

        q = np.array(queries, dtype=np.float32, copy=True)
        if q.ndim != 2 or q.shape[1] != self.dim:
            msg = f"Expected queries with shape (n, {self.dim}), got {q.shape}"
            raise ValueError(msg)
        q = np.ascontiguousarray(q)
        faiss.normalize_L2(q)
        similarities, proto_ids = self._index.search(q, k)
        if self.backend != "flat_ip":
            similarities = np.clip(similarities, -1.0, 1.0)
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

        nlist = int(getattr(self._index, "nlist", 0))
        nprobe = int(getattr(self._index, "nprobe", 0))
        meta = {
            "embedder_name": self.embedder_name,
            "dim": self.dim,
            "n_prototypes": int(self._index.ntotal),
            "backend": self.backend,
            "index_type": type(self._index).__name__,
            "nlist": nlist,
            "nprobe": nprobe,
            "quantizer_bits": self.quantizer_bits,
            "train_size": self.train_size,
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
        raw_index: Any = faiss.read_index(str(path.with_suffix(cls._FAISS_EXT)))
        backend = meta.get("backend")
        if backend is None:
            backend = _infer_faiss_backend(raw_index)
        else:
            backend = normalize_faiss_backend(str(backend))

        obj = cls(
            dim=int(meta["dim"]),
            embedder_name=str(meta["embedder_name"]),
            backend=backend,
            nlist=int(meta.get("nlist") or 1),
            nprobe=int(meta.get("nprobe") or 1),
            quantizer_bits=int(meta.get("quantizer_bits") or 8),
            train_size=meta.get("train_size"),
        )
        if int(raw_index.d) != obj.dim:
            msg = f"Index dim {raw_index.d} does not match metadata dim {obj.dim}"
            raise RuntimeError(msg)

        obj._index = raw_index
        labels = np.load(str(path) + "_labels.npy", allow_pickle=True)
        if len(labels) != int(raw_index.ntotal):
            msg = (
                f"Label count {len(labels)} does not match index size "
                f"{raw_index.ntotal}"
            )
            raise RuntimeError(msg)
        obj._labels = labels
        if backend == "ivf_sq8":
            raw_index.nprobe = max(1, min(obj.nprobe, int(raw_index.nlist)))
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
