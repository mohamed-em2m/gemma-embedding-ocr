"""Hierarchical Word Recognition module."""

from embedding_ocr.recognition.embedding.hierarchical.alignment import (
    AlignedUnit,
    AlignmentResult,
    AlignmentStats,
    EmbeddingAlignmentSegmenter,
    PrototypeSearchDB,
    VocabularyPrototypeDB,
)
from embedding_ocr.recognition.embedding.hierarchical.arabic_units import (
    ArabicUnit,
    ArabicUnitSegmenter,
)
from embedding_ocr.recognition.embedding.hierarchical.craft_detector import (
    CraftWordDetector,
)
from embedding_ocr.recognition.embedding.hierarchical.engine import (
    HierarchicalWordRecognizer,
)
from embedding_ocr.recognition.embedding.hierarchical.segmentation import (
    CutCandidate,
    VisualWordSegmenter,
)
from embedding_ocr.recognition.embedding.hierarchical.vocabulary import (
    EmbeddingVocabulary,
    VocabToken,
)

__all__ = [
    "AlignedUnit",
    "AlignmentResult",
    "AlignmentStats",
    "ArabicUnit",
    "ArabicUnitSegmenter",
    "CraftWordDetector",
    "CutCandidate",
    "EmbeddingAlignmentSegmenter",
    "EmbeddingVocabulary",
    "PrototypeSearchDB",
    "HierarchicalWordRecognizer",
    "VisualWordSegmenter",
    "VocabToken",
    "VocabularyPrototypeDB",
]
