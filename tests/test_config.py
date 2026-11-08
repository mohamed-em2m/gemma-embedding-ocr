"""Tests for typed configuration loading and persistence."""

from __future__ import annotations

from pathlib import Path

import pytest

from embedding_ocr.config import AppConfig, load_config, save_config


def test_default_config_is_valid() -> None:
    config = AppConfig()
    assert config.embedder.dim == 768
    assert config.recognition.backend == "hierarchical"
    assert config.detector.backend == "full"


def test_load_default_yaml() -> None:
    config = load_config("configs/default.yaml")
    assert config.embedder.backend == "gemma"
    assert config.recognition.vocabulary.tokenizer_name == "google/embeddinggemma-2"
    assert config.index.top_k == 30
    assert config.recognition.alignment.strategy == "pruned"
    assert config.recognition.alignment.max_edges_per_start == 2
    assert config.recognition.alignment.fallback_exhaustive is True


def test_load_arabic_alignment_config() -> None:
    config = load_config("configs/embedding_align.yaml")
    assert config.recognition.script == "arabic"
    assert config.recognition.rtl is True
    assert config.recognition.use_alignment is True
    assert config.recognition.vocabulary.use_visual_prototypes is True
    assert config.recognition.alignment.max_cut_candidates is None


def test_save_config_round_trip(tmp_path: Path) -> None:
    config = load_config("configs/embedding_seg.yaml")
    output = tmp_path / "resolved.yaml"
    save_config(config, output)

    restored = load_config(output)
    assert restored.embedder.dim == config.embedder.dim
    assert restored.recognition.backend == "embedding"
    assert restored.index.path == config.index.path


def test_missing_config_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_config("configs/does-not-exist.yaml")


def test_invalid_config_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("- not-a-mapping\n")
    with pytest.raises(ValueError, match="mapping"):
        load_config(path)
