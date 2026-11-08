"""Configuration-driven construction of embedders, indexes, and recognizers."""

from __future__ import annotations

from pathlib import Path
from typing import cast

from embedding_ocr.config import AppConfig, EmbedderSettings, IndexSettings
from embedding_ocr.data.glyph_renderer import ALPHANUMERIC_CLASSES
from embedding_ocr.data.scripts import is_rtl_script, normalize_script_name
from embedding_ocr.pipeline.detect import build_detector
from embedding_ocr.pipeline.pipeline import OCRPipeline
from embedding_ocr.recognition.base import RecognitionEngine
from embedding_ocr.recognition.embedding.embedder import Embedder
from embedding_ocr.recognition.embedding.engine import EmbeddingRecognizer
from embedding_ocr.recognition.embedding.hierarchical.alignment import (
    EmbeddingAlignmentSegmenter,
    VocabularyPrototypeDB,
)
from embedding_ocr.recognition.embedding.hierarchical.engine import (
    HierarchicalWordRecognizer,
)
from embedding_ocr.recognition.embedding.hierarchical.vocabulary import (
    EmbeddingVocabulary,
)
from embedding_ocr.recognition.embedding.index import (
    FaissIndex,
    NumpyIndex,
    VectorIndex,
)
from embedding_ocr.recognition.embedding.prototypes import PrototypeDB


def build_embedder(settings: EmbedderSettings) -> Embedder:
    """Create the configured embedder backend."""
    backend = settings.backend.lower()
    if backend in {"gemma", "embeddinggemma", "embeddinggemma2"}:
        from embedding_ocr.recognition.embedding.embedder import EmbeddingGemma2

        return EmbeddingGemma2(
            dim=settings.dim,
            device=settings.device,
            batch_size=settings.batch_size,
            dtype=settings.dtype,
        )
    if backend == "siglip":
        from embedding_ocr.recognition.embedding.embedder import SiglipEmbedder

        return SiglipEmbedder(
            dim=settings.dim,
            device=settings.device,
            batch_size=settings.batch_size,
            dtype=settings.dtype,
            model_id=settings.model_id,
        )
    if backend == "jina":
        from embedding_ocr.recognition.embedding.embedder import JinaEmbeddingsV4

        return JinaEmbeddingsV4(
            dim=settings.dim,
            device=settings.device,
            batch_size=settings.batch_size,
            dtype=settings.dtype,
            model_id=settings.model_id,
        )
    msg = f"Unknown embedder backend: {settings.backend!r}"
    raise ValueError(msg)


def build_index(
    settings: IndexSettings, dim: int, embedder_name: str = "unknown"
) -> VectorIndex:
    """Create an empty vector index from configuration."""
    backend = settings.backend.lower()
    if backend in {"faiss", "faiss_flat_ip", "flat_ip"}:
        return FaissIndex(
            dim=dim,
            embedder_name=embedder_name,
            omp_threads=settings.omp_threads,
        )
    if backend == "numpy":
        return NumpyIndex(dim=dim, embedder_name=embedder_name)
    msg = f"Unknown index backend: {settings.backend!r}"
    raise ValueError(msg)


def _load_index(settings: IndexSettings, path: Path) -> VectorIndex:
    backend = settings.backend.lower()
    if backend in {"faiss", "faiss_flat_ip", "flat_ip"}:
        return FaissIndex.load(path)
    if backend == "numpy":
        return NumpyIndex.load(path)
    msg = f"Unknown index backend: {settings.backend!r}"
    raise ValueError(msg)


def load_prototype_db(settings: AppConfig, embedder: Embedder) -> PrototypeDB:
    """Load a fitted prototype database from ``config.index.path``."""
    path = settings.index.path
    if path is None:
        msg = "recognition requires index.path pointing to a saved prototype DB"
        raise ValueError(msg)
    base = Path(path)
    index = _load_index(settings.index, base)
    saved_dim = getattr(index, "dim", None)
    if saved_dim is not None and int(saved_dim) != embedder.dim:
        msg = (
            f"Prototype index dim {saved_dim} does not match embedder dim "
            f"{embedder.dim} at {base}."
        )
        raise ValueError(msg)
    saved_name = getattr(index, "embedder_name", None)
    if saved_name is not None and saved_name != embedder.name:
        msg = (
            f"Prototype index was built with {saved_name!r}, but the active "
            f"embedder is {embedder.name!r}."
        )
        raise ValueError(msg)
    db = PrototypeDB(
        embedder=embedder,
        index=index,
        script=settings.recognition.script,
    )
    db.load_cache(base)
    return db


def build_vocabulary(settings: AppConfig, embedder: Embedder) -> EmbeddingVocabulary:
    """Build (or reload cached) word/subword/char vocabulary indexes."""
    vocab_settings = settings.recognition.vocabulary
    vocab = EmbeddingVocabulary(
        embedder=embedder,
        tokenizer_name=vocab_settings.tokenizer_name,
        cache_dir=vocab_settings.cache_dir,
        prototype_fonts=vocab_settings.prototype_fonts or None,
        use_visual_prototypes=vocab_settings.use_visual_prototypes,
        slant_angles=vocab_settings.slant_angles,
        script=settings.recognition.script,
    )
    vocab.extract_and_organize_tokens(
        extra_lexicon=vocab_settings.extra_lexicon,
        max_words=vocab_settings.max_words,
        max_subwords=vocab_settings.max_subwords,
    )
    vocab.build_indexes()
    return vocab


def build_recognizer(
    settings: AppConfig, embedder: Embedder | None = None
) -> RecognitionEngine:
    """Create the configured recognition engine."""
    rec = settings.recognition
    backend = rec.backend.lower()

    if backend == "tesseract":
        from embedding_ocr.recognition.tesseract import TesseractRecognizer

        return TesseractRecognizer(psm=rec.tesseract_psm, lang=rec.tesseract_lang)
    if backend == "paddle":
        from embedding_ocr.recognition.paddle import PaddleRecognizer

        return PaddleRecognizer(model_name=rec.paddle_model)
    if backend == "easyocr":
        from embedding_ocr.recognition.easyocr import EasyOCRRecognizer

        return EasyOCRRecognizer(lang_list=rec.easyocr_langs, gpu=rec.easyocr_gpu)
    if backend == "dtlr":
        from embedding_ocr.recognition.dtlr import DTLRRecognizer

        return DTLRRecognizer(
            root=rec.dtlr_root,
            config_path=rec.dtlr_config,
            weights=rec.dtlr_weights,
            charset_path=rec.dtlr_charset,
            device=rec.dtlr_device,
        )
    if backend in {"yolo", "yolo_units"}:
        if rec.yolo_model is None:
            message = "recognition.yolo_model must point to a YOLO checkpoint"
            raise ValueError(message)
        from embedding_ocr.recognition.yolo_units import (
            YoloUnitDetector,
            YoloUnitRecognizer,
        )

        detector = YoloUnitDetector.from_checkpoint(
            model_path=rec.yolo_model,
            classes_path=rec.yolo_classes,
            conf=rec.yolo_conf,
            iou=rec.yolo_iou,
            imgsz=rec.yolo_imgsz,
            device=rec.yolo_device,
            max_det=rec.yolo_max_det,
        )
        return YoloUnitRecognizer(
            detector=detector,
            rtl=rec.rtl,
            space_gap_ratio=rec.yolo_space_gap_ratio,
        )
    if backend == "embedding":
        if embedder is None:
            embedder = build_embedder(settings.embedder)
        proto_db = load_prototype_db(settings, embedder)
        return EmbeddingRecognizer(
            prototype_db=proto_db,
            top_k=settings.index.top_k,
            script=rec.script,
            rtl=rec.rtl,
        )
    if backend == "hierarchical":
        if embedder is None:
            embedder = build_embedder(settings.embedder)
        vocabulary = build_vocabulary(settings, embedder)
        aligner = None
        if rec.use_alignment:
            if settings.index.path is not None and Path(settings.index.path).exists():
                proto_db = load_prototype_db(settings, embedder)
                aligner = EmbeddingAlignmentSegmenter(
                    prototype_db=proto_db,
                    top_k=min(5, settings.index.top_k),
                    min_similarity=0.35,
                    strategy=rec.alignment.strategy,
                    max_cut_candidates=rec.alignment.max_cut_candidates,
                    min_cut_spacing=rec.alignment.min_cut_spacing,
                    max_edges_per_start=rec.alignment.max_edges_per_start,
                    fallback_exhaustive=rec.alignment.fallback_exhaustive,
                )
            else:
                aligner = EmbeddingAlignmentSegmenter(
                    prototype_db=VocabularyPrototypeDB(embedder, vocabulary),
                    top_k=5,
                    min_similarity=0.35,
                    strategy=rec.alignment.strategy,
                    max_cut_candidates=rec.alignment.max_cut_candidates,
                    min_cut_spacing=rec.alignment.min_cut_spacing,
                    max_edges_per_start=rec.alignment.max_edges_per_start,
                    fallback_exhaustive=rec.alignment.fallback_exhaustive,
                )
        return HierarchicalWordRecognizer(
            embedder=embedder,
            vocabulary=vocabulary,
            script=rec.script,
            rtl=rec.rtl,
            unit_aligner=aligner,
            use_alignment=rec.use_alignment,
            min_alignment_score=rec.min_alignment_score,
            min_word_sim=rec.min_word_sim,
            min_word_margin=rec.min_word_margin,
            min_subword_sim=rec.min_subword_sim,
            min_subword_margin=rec.min_subword_margin,
            min_char_sim=rec.min_char_sim,
            max_recursion_depth=rec.max_recursion_depth,
            preprocess_handwriting=rec.preprocess_handwriting,
            top_k=rec.top_k,
        )

    msg = f"Unknown recognizer backend: {rec.backend!r}"
    raise ValueError(msg)


def resolve_rtl(settings: AppConfig) -> bool:
    """Resolve reading direction from explicit config or script."""
    if settings.recognition.rtl is not None:
        return settings.recognition.rtl
    return is_rtl_script(normalize_script_name(settings.recognition.script))


def build_pipeline(settings: AppConfig) -> OCRPipeline:
    """Build a complete pipeline from a validated config."""
    needs_embedder = settings.recognition.backend.lower() in {
        "embedding",
        "hierarchical",
    }
    embedder = build_embedder(settings.embedder) if needs_embedder else None
    recognizer = build_recognizer(settings, embedder)
    detector = build_detector(settings.detector, rtl=resolve_rtl(settings))
    return OCRPipeline(
        detector=detector,
        recognizer=recognizer,
        settings=settings.pipeline,
        rtl=resolve_rtl(settings),
    )


def build_prototypes(
    settings: AppConfig, fonts: list[str] | None = None
) -> PrototypeDB:
    """Build a prototype DB from config fonts/classes and return it.

    This is used by the ``build-prototypes`` CLI command. The caller is
    responsible for saving the returned DB.
    """
    embedder = build_embedder(settings.embedder)
    index = _empty_index(settings, embedder)
    classes = settings.recognition.classes or None
    if (
        classes is None
        and normalize_script_name(settings.recognition.script) == "latin"
    ):
        classes = ALPHANUMERIC_CLASSES
    db = PrototypeDB(
        embedder=embedder,
        index=index,
        classes=classes,
        script=settings.recognition.script,
    )
    resolved_fonts = fonts or settings.recognition.vocabulary.prototype_fonts
    if not resolved_fonts:
        msg = "no fonts supplied and vocabulary.prototype_fonts is empty"
        raise ValueError(msg)
    db.build_from_fonts(resolved_fonts)
    return db


def _empty_index(settings: AppConfig, embedder: Embedder) -> VectorIndex:
    dim = settings.index.dim or embedder.dim
    return cast(
        VectorIndex,
        build_index(settings.index, dim=dim, embedder_name=embedder.name),
    )
