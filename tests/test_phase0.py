"""Unit tests for Phase 0 source modules.

These tests must pass without the real EmbeddingGemma2 model loaded.
Tests that need the real model are marked with @pytest.mark.model and
are skipped in CI (no network / no GPU).
"""

from __future__ import annotations

import numpy as np
import pytest

from embedding_ocr.data.glyph_renderer import (
    ALL_CHAR_CLASSES,
    ALPHANUMERIC_CLASSES,
    CASE_PAIRS,
    DIGITS,
    LATIN_CLASSES,
    LOOKALIKE_PAIRS,
    render_all_glyphs,
    render_glyph,
    render_glyph_with_features,
)
from embedding_ocr.recognition.base import RecognitionResult
from embedding_ocr.recognition.embedding.index import FaissIndex, NumpyIndex
from embedding_ocr.utils import get_device

# ── Glyph renderer ───────────────────────────────────────────────────────────


def test_latin_classes_count() -> None:
    assert len(LATIN_CLASSES) == 52


def test_digits_and_symbols_count() -> None:
    assert len(DIGITS) == 10
    assert len(ALPHANUMERIC_CLASSES) == 62
    assert len(ALL_CHAR_CLASSES) > 80


def test_render_glyph_default_font() -> None:
    """render_glyph must not raise and must return an RGB image."""
    from PIL import Image

    img = render_glyph("A", font_path="/nonexistent/font.ttf")
    assert isinstance(img, Image.Image)
    assert img.size == (64, 64)
    assert img.mode == "RGB"


def test_render_glyph_digits_and_symbols() -> None:
    """Check digits and symbols render without error."""
    img_digit = render_glyph("7", font_path="/nonexistent/font.ttf")
    img_punct = render_glyph("?", font_path="/nonexistent/font.ttf")
    assert img_digit.size == (64, 64)
    assert img_punct.size == (64, 64)


def test_render_glyph_with_features() -> None:
    """Test feature extraction (aspect ratio, relative height)."""
    img, feats = render_glyph_with_features("c", font_path="/nonexistent/font.ttf")
    assert "aspect_ratio" in feats
    assert "relative_height" in feats
    assert "relative_width" in feats
    assert feats["relative_height"] > 0.0


def test_render_glyph_pixel_not_all_white() -> None:
    """A rendered glyph should have at least some dark pixels."""
    img = render_glyph("A", font_path="/nonexistent/font.ttf")
    arr = np.array(img)
    assert arr.min() < 200, "Expected some dark pixels in the glyph render"


def test_render_all_glyphs_count() -> None:
    """render_all_glyphs returns one image per (font, char) pair."""
    images, labels = render_all_glyphs(
        font_paths=["/nonexistent/a.ttf", "/nonexistent/b.ttf"],
        classes=["a", "b", "c"],
    )
    assert len(images) == 6
    assert len(labels) == 6
    assert labels[:3] == ["a", "b", "c"]


def test_case_pairs_completeness() -> None:
    for low, up in CASE_PAIRS:
        assert low.islower()
        assert up.isupper()
        assert low.upper() == up


def test_lookalikes_contain_digits() -> None:
    assert ("O", "0") in LOOKALIKE_PAIRS
    assert ("l", "1") in LOOKALIKE_PAIRS


# ── RecognitionResult ────────────────────────────────────────────────────────


def test_recognition_result_valid() -> None:
    r = RecognitionResult(text="hello", confidence=0.95)
    assert r.text == "hello"
    assert r.confidence == 0.95


def test_recognition_result_frozen() -> None:
    r = RecognitionResult(text="x", confidence=0.5)
    with pytest.raises((AttributeError, TypeError)):
        r.text = "y"  # type: ignore[misc]


def test_recognition_result_confidence_range() -> None:
    with pytest.raises(ValueError, match="confidence must be in"):
        RecognitionResult(text="x", confidence=1.5)
    with pytest.raises(ValueError, match="confidence must be in"):
        RecognitionResult(text="x", confidence=-0.1)


# ── NumpyIndex ───────────────────────────────────────────────────────────────

RNG = np.random.default_rng(42)


def _random_unit_vectors(n: int, dim: int) -> np.ndarray:
    vecs = RNG.standard_normal((n, dim)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


def test_numpy_index_self_retrieval() -> None:
    """Each prototype should retrieve itself at rank 0."""
    dim = 64
    n = 20
    vecs = _random_unit_vectors(n, dim)
    labels = np.arange(n)

    idx = NumpyIndex(dim=dim)
    idx.build(vecs, labels)
    sims, ids = idx.search(vecs, k=1)

    assert (ids[:, 0] == labels).all(), "Self-retrieval failed"
    assert np.allclose(sims[:, 0], 1.0, atol=1e-5), "Self-similarity should be ~1.0"


def test_numpy_index_top_k_shape() -> None:
    dim, n, k = 32, 10, 5
    vecs = _random_unit_vectors(n, dim)
    idx = NumpyIndex(dim=dim)
    idx.build(vecs, np.arange(n))
    sims, ids = idx.search(vecs[:3], k=k)
    assert sims.shape == (3, k)
    assert ids.shape == (3, k)


def test_numpy_index_not_built_raises() -> None:
    idx = NumpyIndex(dim=32)
    with pytest.raises(RuntimeError):
        idx.search(_random_unit_vectors(1, 32), k=1)


def test_numpy_index_save_load(tmp_path: pytest.TempPathFactory) -> None:
    dim = 32
    vecs = _random_unit_vectors(10, dim)
    labels = np.arange(10)

    idx = NumpyIndex(dim=dim)
    idx.build(vecs, labels)
    save_path = tmp_path / "test_idx"
    idx.save(save_path)

    idx2 = NumpyIndex.load(save_path)
    sims1, ids1 = idx.search(vecs, k=3)
    sims2, ids2 = idx2.search(vecs, k=3)
    np.testing.assert_array_equal(ids1, ids2)
    np.testing.assert_allclose(sims1, sims2, atol=1e-6)


# ── FaissIndex vs NumpyIndex agreement ──────────────────────────────────────


def test_faiss_numpy_agreement() -> None:
    """FaissIndex and NumpyIndex must return identical top-k on the same data."""
    dim = 64
    n = 30
    k = 5
    vecs = _random_unit_vectors(n, dim)
    labels = np.arange(n)
    queries = _random_unit_vectors(10, dim)

    np_idx = NumpyIndex(dim=dim)
    np_idx.build(vecs, labels)
    np_sims, np_ids = np_idx.search(queries, k=k)

    faiss_idx = FaissIndex(dim=dim)
    faiss_idx.build(vecs, labels)
    fa_sims, fa_ids = faiss_idx.search(queries, k=k)

    np.testing.assert_array_equal(
        np_ids,
        fa_ids,
        err_msg="FaissIndex and NumpyIndex return different top-k ids",
    )
    np.testing.assert_allclose(
        np_sims,
        fa_sims,
        atol=1e-5,
        err_msg="FaissIndex and NumpyIndex similarities differ",
    )


def test_faiss_index_self_retrieval() -> None:
    dim = 64
    n = 20
    vecs = _random_unit_vectors(n, dim)
    labels = np.arange(n)

    idx = FaissIndex(dim=dim)
    idx.build(vecs, labels)
    sims, ids = idx.search(vecs, k=1)
    assert (ids[:, 0] == labels).all()


# ── Utils ────────────────────────────────────────────────────────────────────


def test_get_device_cpu() -> None:
    assert get_device("cpu") == "cpu"


def test_get_device_auto_returns_string() -> None:
    d = get_device("auto")
    assert d in ("cpu", "cuda")


def test_set_seed_determinism() -> None:
    rng1 = np.random.default_rng(123)
    a = rng1.random(5)
    rng2 = np.random.default_rng(123)
    b = rng2.random(5)
    np.testing.assert_array_equal(a, b)


# ── Matryoshka truncation + re-normalization ─────────────────────────────────


def test_matryoshka_truncation_renormalization() -> None:
    """After slicing to dim=128 and re-normalizing, norms should be ~1."""
    full_dim = 768
    vecs = _random_unit_vectors(10, full_dim)
    truncated = vecs[:, :128]
    norms = np.linalg.norm(truncated, axis=1)
    assert not np.allclose(norms, 1.0, atol=1e-3)

    renormed = truncated / norms[:, None]
    norms2 = np.linalg.norm(renormed, axis=1)
    np.testing.assert_allclose(norms2, 1.0, atol=1e-6)
