"""Tests for vocabulary candidate selection and truncation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from embedding_ocr.recognition.embedding.hierarchical.vocabulary import (
    EmbeddingVocabulary,
)


class DummyEmbedder:
    name = "dummy"
    dim = 8

    def embed_images(self, images: list[np.ndarray]) -> np.ndarray:
        return np.ones((len(images), self.dim), dtype=np.float32) / np.sqrt(self.dim)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        return np.ones((len(texts), self.dim), dtype=np.float32) / np.sqrt(self.dim)


class FakeTokenizer:
    all_special_ids = [0]

    def get_vocab(self) -> dict[str, int]:
        return {
            "<pad>": 0,
            "\u2581fjord": 1,
            "fjord": 2,
            "\u2581the": 3,
            "a": 4,
        }


def test_full_vocabulary_keeps_rare_latin_tokens(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import transformers

    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: FakeTokenizer(),
    )
    vocab = EmbeddingVocabulary(
        embedder=DummyEmbedder(),
        cache_dir=tmp_path / "cache",
        script="latin",
    )
    vocab.extract_and_organize_tokens(max_words=None, max_subwords=None)

    words = {token.text for token in vocab.word_tokens}
    subwords = {token.text for token in vocab.subword_tokens}
    assert words == {"fjord", "the"}
    assert subwords == {"fjord"}


def test_vocabulary_uses_configured_index_backend(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import transformers

    from embedding_ocr.config import IndexSettings

    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: FakeTokenizer(),
    )
    vocab = EmbeddingVocabulary(
        embedder=DummyEmbedder(),
        cache_dir=tmp_path / "cache",
        script="latin",
        index_settings=IndexSettings(
            backend="faiss_ivf_sq8",
            nlist=4,
            nprobe=2,
            omp_threads=1,
        ),
    )
    vocab.extract_and_organize_tokens(max_words=None, max_subwords=None)
    vocab.build_indexes()

    indexes = [vocab.word_index, vocab.subword_index, vocab.char_index]
    present = [index for index in indexes if index is not None]
    assert present
    assert all(getattr(index, "backend", None) == "ivf_sq8" for index in present)
