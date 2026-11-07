"""Prototype database builder and storage for EmbeddingOCR."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from embedding_ocr.data.glyph_renderer import classes_for_script, render_glyph
from embedding_ocr.data.scripts import normalize_script_name
from embedding_ocr.recognition.embedding.embedder import Embedder
from embedding_ocr.recognition.embedding.index import VectorIndex


@dataclass
class PrototypeMetadata:
    """Metadata describing a prototype index."""

    embedder_name: str
    dim: int
    classes: list[str]
    fonts: list[str]
    canvas_size: int
    n_prototypes: int
    script: str = "latin"


class PrototypeDB:
    """Manages character prototype embeddings and vector matching."""

    def __init__(
        self,
        embedder: Embedder,
        index: VectorIndex,
        classes: Sequence[str] | None = None,
        canvas_size: int = 64,
        font_size: int = 44,
        script: str = "latin",
    ) -> None:
        self.embedder = embedder
        self.index = index
        self.script = normalize_script_name(script)
        self.classes = (
            list(classes) if classes is not None else classes_for_script(self.script)
        )
        self.canvas_size = canvas_size
        self.font_size = font_size
        self.labels: list[str] = []
        self.fonts: list[str] = []

    def build_from_fonts(self, font_paths: Sequence[str | Path]) -> None:
        """Render character prototypes from fonts, embed them, and index them.

        Args:
            font_paths: List of file paths to TTF/OTF fonts.
        """
        self.fonts = [str(p) for p in font_paths]
        images: list[np.ndarray] = []
        labels: list[str] = []

        for font_p in font_paths:
            for char in self.classes:
                pil_img = render_glyph(
                    char=char,
                    font_path=font_p,
                    canvas_size=self.canvas_size,
                    font_size=self.font_size,
                    script=self.script,
                )
                images.append(np.array(pil_img, dtype=np.uint8))
                labels.append(char)

        # Extract embeddings
        embeddings = self.embedder.embed_images(images)

        # Index them (labels mapped to integer indices, store mapping)
        self.labels = labels
        int_labels = np.arange(len(labels))
        self.index.build(embeddings, int_labels)

    def save(self, path: Path | str) -> None:
        """Persist prototype index and metadata to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.index.save(path)
        meta = {
            "classes": self.classes,
            "fonts": self.fonts,
            "canvas_size": self.canvas_size,
            "font_size": self.font_size,
            "labels": self.labels,
            "script": self.script,
        }
        meta_file = path.with_suffix(".labels.json")
        import json

        meta_file.write_text(json.dumps(meta, indent=2))

    def load_cache(self, path: Path | str) -> None:
        """Load prototype labels from disk metadata corresponding to saved index."""
        path = Path(path)
        meta_file = path.with_suffix(".labels.json")
        if not meta_file.exists():
            msg = f"Prototype metadata file {meta_file} not found."
            raise FileNotFoundError(msg)
        import json

        meta = json.loads(meta_file.read_text())
        self.script = normalize_script_name(meta.get("script", self.script))
        saved_classes = meta.get("classes")
        self.classes = (
            list(saved_classes)
            if saved_classes is not None
            else classes_for_script(self.script)
        )
        self.fonts = meta.get("fonts", self.fonts)
        self.canvas_size = meta.get("canvas_size", self.canvas_size)
        self.font_size = meta.get("font_size", self.font_size)
        self.labels = meta.get("labels", [])

    def search(
        self, query_embeddings: np.ndarray, top_k: int = 20
    ) -> tuple[np.ndarray, list[list[str]]]:
        """Search nearest prototypes for query embeddings.

        Args:
            query_embeddings: Array of shape (n_queries, dim).
            top_k: Number of nearest prototypes to retrieve.

        Returns:
            Tuple of (similarities: np.ndarray, retrieved_character_labels: list[list[str]]).
        """
        if not self.labels:
            empty = np.empty((len(query_embeddings), 0), dtype=np.float32)
            return empty, [[] for _ in range(len(query_embeddings))]

        # FAISS returns -1 ids when k exceeds the number of prototypes; clamp
        # so callers never index labels[-1] accidentally.
        k = max(1, min(top_k, len(self.labels)))
        similarities, ids = self.index.search(query_embeddings, k=k)
        retrieved_labels = [[self.labels[int(idx)] for idx in row] for row in ids]
        return similarities, retrieved_labels
