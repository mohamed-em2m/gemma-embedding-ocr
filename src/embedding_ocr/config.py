"""Typed configuration for EmbeddingOCR.

All runtime settings are declared here and loaded from YAML. The aim is that
swapping a recognizer or embedder only changes configuration, not code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class _Model(BaseModel):
    model_config = ConfigDict(extra="ignore")


class EmbedderSettings(_Model):
    """Embedder backend settings."""

    backend: str = "gemma"
    model_id: str | None = None
    dim: int = 768
    device: str = "auto"
    dtype: str = "float32"
    batch_size: int = 0


class IndexSettings(_Model):
    """Vector index and prototype storage settings."""

    backend: str = "faiss_ivf_sq8"
    dim: int | None = None
    top_k: int = 30
    path: Path | None = None
    omp_threads: int | None = None
    nlist: int = 1024
    nprobe: int = 16
    quantizer_bits: int = 8
    train_size: int | None = None


class DetectorSettings(_Model):
    """Text detection settings."""

    backend: str = "auto"
    rtl: bool | None = None
    device: str = "auto"
    min_component_area: int = 8
    craft_cuda: bool = False
    lang: str = "en"
    paddle_model: str = "PP-OCRv5_server_det"
    paddle_mobile_model: str = "PP-OCRv5_mobile_det"
    min_confidence: float = 0.3
    box_padding: int = 2
    yolo_model: Path | None = None
    yolo_conf: float = 0.25
    yolo_iou: float = 0.5
    yolo_imgsz: int = 1280
    yolo_device: str | None = None
    yolo_max_det: int = 1000


class VocabularySettings(_Model):
    """Vocabulary and prototype cache settings for hierarchical recognition."""

    tokenizer_name: str = "google/embeddinggemma-2"
    cache_dir: Path = Path("artifacts/vocab_cache")
    use_visual_prototypes: bool = False
    prototype_fonts: list[str] = Field(default_factory=list)
    slant_angles: list[float] = Field(default_factory=lambda: [0.0])
    max_words: int | None = 25000
    max_subwords: int | None = 15000
    extra_lexicon: list[str] = Field(default_factory=list)


class AlignmentSettings(_Model):
    """Segmentation-free alignment candidate-graph settings."""

    strategy: Literal["pruned", "exhaustive"] = "pruned"
    max_cut_candidates: int | None = None
    min_cut_spacing: float | None = None
    max_edges_per_start: int = 3
    fallback_exhaustive: bool = True


class RecognitionSettings(_Model):
    """Recognition backend settings."""

    backend: str = "hierarchical"
    script: str = "latin"
    classes: list[str] = Field(default_factory=list)
    rtl: bool | None = None
    top_k: int = 15
    min_word_sim: float = 0.74
    min_word_margin: float = 0.008
    min_subword_sim: float = 0.72
    min_subword_margin: float = 0.005
    min_char_sim: float = 0.65
    max_recursion_depth: int = 4
    preprocess_handwriting: bool = False
    use_alignment: bool = True
    min_alignment_score: float = 0.35
    alignment: AlignmentSettings = Field(default_factory=AlignmentSettings)
    tesseract_psm: int = 7
    tesseract_lang: str = "eng"
    paddle_model: str = "en_PP-OCRv4_mobile_rec"
    easyocr_langs: list[str] = Field(default_factory=lambda: ["en"])
    easyocr_gpu: bool = True
    dtlr_root: Path | None = None
    dtlr_config: Path | None = None
    dtlr_weights: Path | None = None
    dtlr_charset: Path | None = None
    dtlr_device: str = "cuda"
    yolo_model: Path | None = None
    yolo_classes: Path | None = None
    yolo_conf: float = 0.25
    yolo_iou: float = 0.5
    yolo_imgsz: int = 1280
    yolo_device: str | None = None
    yolo_max_det: int = 1000
    yolo_space_gap_ratio: float = 0.9
    vocabulary: VocabularySettings = Field(default_factory=VocabularySettings)


class PipelineSettings(_Model):
    """End-to-end pipeline settings."""

    preprocess: bool = False
    deslant: bool = True
    binarize: bool = True
    padding: int = 4
    detection_cache: Path | None = None
    seed: int = 42


class AppConfig(_Model):
    """Top-level application configuration."""

    embedder: EmbedderSettings = Field(default_factory=EmbedderSettings)
    detector: DetectorSettings = Field(default_factory=DetectorSettings)
    recognition: RecognitionSettings = Field(default_factory=RecognitionSettings)
    index: IndexSettings = Field(default_factory=IndexSettings)
    pipeline: PipelineSettings = Field(default_factory=PipelineSettings)


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load configuration from YAML, falling back to defaults.

    Args:
        path: Optional YAML file. ``None`` returns the built-in defaults.

    Returns:
        Validated :class:`AppConfig`.

    Raises:
        FileNotFoundError: If ``path`` is given but does not exist.
        pydantic.ValidationError: If the YAML is structurally invalid.
    """
    if path is None:
        return AppConfig()

    config_path = Path(path)
    if not config_path.exists():
        msg = f"Config file not found: {config_path}"
        raise FileNotFoundError(msg)

    raw = yaml.safe_load(config_path.read_text()) or {}
    if not isinstance(raw, dict):
        msg = f"Config root must be a mapping, got {type(raw).__name__}"
        raise ValueError(msg)
    return AppConfig.model_validate(raw)


def save_config(config: AppConfig, path: str | Path) -> None:
    """Write a config to YAML for reproducibility."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False))
