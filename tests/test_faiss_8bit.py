"""Tests for FAISS 8-bit scalar-quantized index backends."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from embedding_ocr.config import IndexSettings
from embedding_ocr.recognition.embedding.index import (
    FaissIndex,
    is_faiss_backend,
    normalize_faiss_backend,
)
from embedding_ocr.recognition.factory import build_index


def _random_unit_vectors(n: int, dim: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vecs = rng.standard_normal((n, dim)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs


def test_backend_aliases() -> None:
    assert normalize_faiss_backend("faiss") == "flat_ip"
    assert normalize_faiss_backend("faiss_sq8") == "sq8"
    assert normalize_faiss_backend("faiss_ivf_sq8") == "ivf_sq8"
    assert is_faiss_backend("faiss_ivf_sq8")
    assert not is_faiss_backend("numpy")
    with pytest.raises(ValueError, match="Unknown FAISS index backend"):
        normalize_faiss_backend("hnsw")


@pytest.mark.parametrize("backend", ["sq8", "ivf_sq8"])
def test_quantized_backends_self_retrieval(backend: str) -> None:
    dim = 64
    n = 200
    vecs = _random_unit_vectors(n, dim)
    labels = np.arange(n)

    index = FaissIndex(
        dim=dim,
        backend=backend,
        nlist=8,
        nprobe=8,
        train_size=128,
    )
    index.build(vecs, labels)
    sims, ids = index.search(vecs[:20], k=5)

    assert sims.shape == (20, 5)
    assert ids.shape == (20, 5)
    assert np.isfinite(sims).all()
    assert (sims <= 1.0 + 1e-6).all()
    assert (ids[:, 0] == labels[:20]).all()


def test_quantized_backends_save_and_load(tmp_path: Path) -> None:
    dim = 64
    vecs = _random_unit_vectors(300, dim)
    labels = np.arange(300)
    queries = _random_unit_vectors(10, dim, seed=1)

    for backend in ("sq8", "ivf_sq8"):
        path = tmp_path / backend
        index = FaissIndex(dim=dim, backend=backend, nlist=8, nprobe=4)
        index.build(vecs, labels)
        expected_sims, expected_ids = index.search(queries, k=5)
        index.save(path)

        loaded = FaissIndex.load(path)
        assert loaded.backend == backend
        assert loaded.dim == dim
        actual_sims, actual_ids = loaded.search(queries, k=5)
        np.testing.assert_array_equal(actual_ids, expected_ids)
        np.testing.assert_allclose(actual_sims, expected_sims, atol=1e-6)


def test_sq8_index_is_smaller_than_flat(tmp_path: Path) -> None:
    dim = 64
    vecs = _random_unit_vectors(500, dim)
    labels = np.arange(500)

    flat = FaissIndex(dim=dim, backend="flat_ip")
    flat.build(vecs, labels)
    flat.save(tmp_path / "flat")

    sq8 = FaissIndex(dim=dim, backend="sq8")
    sq8.build(vecs, labels)
    sq8.save(tmp_path / "sq8")

    flat_size = (tmp_path / "flat.faiss").stat().st_size
    sq8_size = (tmp_path / "sq8.faiss").stat().st_size
    assert sq8_size < flat_size


def test_legacy_flat_metadata_loads(tmp_path: Path) -> None:
    dim = 32
    vecs = _random_unit_vectors(50, dim)
    labels = np.arange(50)
    path = tmp_path / "legacy"
    index = FaissIndex(dim=dim, backend="flat_ip")
    index.build(vecs, labels)
    index.save(path)

    meta_path = path.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text())
    meta.pop("backend", None)
    meta.pop("index_type", None)
    meta_path.write_text(json.dumps(meta))

    loaded = FaissIndex.load(path)
    assert loaded.backend == "flat_ip"
    _, ids = loaded.search(vecs[:5], k=3)
    assert ids.shape == (5, 3)


def test_factory_builds_quantized_index() -> None:
    settings = IndexSettings(
        backend="faiss_ivf_sq8",
        nlist=8,
        nprobe=4,
        quantizer_bits=8,
        omp_threads=1,
    )
    index = build_index(settings, dim=32)
    assert isinstance(index, FaissIndex)
    assert index.backend == "ivf_sq8"
    assert index.nlist == 8
    assert index.nprobe == 4


def test_quantizer_bits_must_be_eight() -> None:
    with pytest.raises(ValueError, match="8-bit"):
        FaissIndex(dim=32, backend="sq8", quantizer_bits=4)
