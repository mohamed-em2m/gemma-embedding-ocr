"""Embedding recognition sub-module."""

from __future__ import annotations

from embedding_ocr.recognition.embedding.embedder import Embedder, EmbeddingGemma2
from embedding_ocr.recognition.embedding.engine import EmbeddingRecognizer
from embedding_ocr.recognition.embedding.index import FaissIndex, NumpyIndex, VectorIndex
from embedding_ocr.recognition.embedding.matcher import MatchPrediction, PrototypeMatcher
from embedding_ocr.recognition.embedding.prototypes import PrototypeDB
from embedding_ocr.recognition.embedding.segmenter import CharacterSegmenter, SegmentedCharacter

__all__ = [
    "CharacterSegmenter",
    "Embedder",
    "EmbeddingGemma2",
    "EmbeddingRecognizer",
    "FaissIndex",
    "MatchPrediction",
    "NumpyIndex",
    "PrototypeDB",
    "PrototypeMatcher",
    "SegmentedCharacter",
    "VectorIndex",
]
