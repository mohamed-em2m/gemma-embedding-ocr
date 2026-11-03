"""Embedding vocabulary builder and manager using EmbeddingGemma 2 tokenizer.

Extracts tokens from the tokenizer, categorizes them into:
1. Complete words (e.g. leading with whitespace marker \u2581 or whole dictionary words)
2. Subwords (morphemes, affixes, partial fragments)
3. Characters (single alphanumeric and ASCII glyphs)

Builds and caches text embeddings and candidate indices.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from embedding_ocr.recognition.embedding.embedder import Embedder
from embedding_ocr.recognition.embedding.index import FaissIndex, VectorIndex


@dataclass(frozen=True)
class VocabToken:
    """Decoded vocabulary candidate."""

    text: str
    token_id: int
    raw_token: str
    category: str  # 'word' | 'subword' | 'char'


class EmbeddingVocabulary:
    """Manages tokenizer vocabulary, text embeddings, and FAISS candidate indexes."""

    def __init__(
        self,
        embedder: Embedder,
        tokenizer_name: str = "google/embeddinggemma-2",
        cache_dir: str | Path = "artifacts/vocab_cache",
    ) -> None:
        self.embedder = embedder
        self.tokenizer_name = tokenizer_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.word_tokens: list[VocabToken] = []
        self.subword_tokens: list[VocabToken] = []
        self.char_tokens: list[VocabToken] = []

        self.word_index: VectorIndex | None = None
        self.subword_index: VectorIndex | None = None
        self.char_index: VectorIndex | None = None

        self._word_lookup: dict[int, VocabToken] = {}
        self._subword_lookup: dict[int, VocabToken] = {}
        self._char_lookup: dict[int, VocabToken] = {}

    def extract_and_organize_tokens(
        self,
        extra_lexicon: Sequence[str] | None = None,
        max_words: int | None = 25000,
        max_subwords: int | None = 15000,
    ) -> None:
        """Extract all valid Latin tokens from the tokenizer and organize into hierarchy."""
        from transformers import AutoTokenizer  # type: ignore[import-untyped]

        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        vocab = tokenizer.get_vocab()
        special_ids = set(tokenizer.all_special_ids)

        words_dict: dict[str, VocabToken] = {}
        subwords_dict: dict[str, VocabToken] = {}
        chars_dict: dict[str, VocabToken] = {}

        # 1. Process tokenizer vocabulary
        for token, tid in vocab.items():
            if (
                tid in special_ids
                or token.startswith("<")
                or token.startswith("[")
                or "bytefallback" in token
            ):
                continue

            decoded = tokenizer.decode([tid])
            clean = decoded.strip()
            if not clean:
                continue

            # Pure Latin ASCII printable tokens
            if not all(c.isascii() and (c.isalnum() or c in "-_'.,:;!?") for c in clean):
                continue

            if len(clean) == 1:
                if clean not in chars_dict:
                    chars_dict[clean] = VocabToken(
                        text=clean, token_id=tid, raw_token=token, category="char"
                    )
            elif token.startswith("\u2581") or token.startswith(" "):
                if clean not in words_dict:
                    words_dict[clean] = VocabToken(
                        text=clean, token_id=tid, raw_token=token, category="word"
                    )
            else:
                if clean not in subwords_dict:
                    subwords_dict[clean] = VocabToken(
                        text=clean, token_id=tid, raw_token=token, category="subword"
                    )

        # 2. Add extra full-word lexicon words if provided (e.g. multi-token domain words)
        if extra_lexicon:
            for word in extra_lexicon:
                clean_w = word.strip()
                if clean_w and clean_w not in words_dict and clean_w not in chars_dict:
                    words_dict[clean_w] = VocabToken(
                        text=clean_w,
                        token_id=-1,  # multi-token / custom word
                        raw_token=f"\u2581{clean_w}",
                        category="word",
                    )

        # Truncate to desired candidate budget for efficiency
        word_list = list(words_dict.values())
        if max_words and len(word_list) > max_words:
            # Sort by token_id (most frequent in tokenizer are lower IDs)
            word_list.sort(key=lambda t: (t.token_id if t.token_id >= 0 else 999999))
            word_list = word_list[:max_words]

        subword_list = list(subwords_dict.values())
        if max_subwords and len(subword_list) > max_subwords:
            subword_list.sort(key=lambda t: t.token_id)
            subword_list = subword_list[:max_subwords]

        char_list = list(chars_dict.values())
        char_list.sort(key=lambda t: t.token_id)

        self.word_tokens = word_list
        self.subword_tokens = subword_list
        self.char_tokens = char_list

        self._word_lookup = {i: tok for i, tok in enumerate(self.word_tokens)}
        self._subword_lookup = {i: tok for i, tok in enumerate(self.subword_tokens)}
        self._char_lookup = {i: tok for i, tok in enumerate(self.char_tokens)}

    def build_indexes(self, force_recompute: bool = False) -> None:
        """Precompute and cache text embeddings and build FAISS indices for all tiers."""
        categories = [
            ("words", self.word_tokens, "word_index"),
            ("subwords", self.subword_tokens, "subword_index"),
            ("chars", self.char_tokens, "char_index"),
        ]

        for cat_name, tokens, attr_name in categories:
            emb_file = self.cache_dir / f"{cat_name}_emb_{self.embedder.dim}d.npy"
            meta_file = self.cache_dir / f"{cat_name}_meta.json"

            if not force_recompute and emb_file.exists() and meta_file.exists():
                vectors = np.load(emb_file)
                # Verify length matches
                if len(vectors) == len(tokens):
                    index = FaissIndex(dim=self.embedder.dim)
                    index.build(vectors, np.arange(len(tokens)))
                    setattr(self, attr_name, index)
                    continue

            # Embed candidate texts
            texts = [tok.text for tok in tokens]
            if not texts:
                continue

            vectors = self.embedder.embed_texts(texts)
            np.save(emb_file, vectors)

            # Save metadata
            meta = [
                {"text": t.text, "token_id": t.token_id, "category": t.category}
                for t in tokens
            ]
            meta_file.write_text(json.dumps(meta, indent=2))

            index = FaissIndex(dim=self.embedder.dim)
            index.build(vectors, np.arange(len(tokens)))
            setattr(self, attr_name, index)

    def search_words(self, query_embs: np.ndarray, top_k: int = 10) -> tuple[np.ndarray, list[list[VocabToken]]]:
        """Search nearest word candidates."""
        if self.word_index is None:
            raise RuntimeError("Word index is not built.")
        sims, idxs = self.word_index.search(query_embs, top_k)
        tokens = [[self._word_lookup[i] for i in row if i in self._word_lookup] for row in idxs]
        return sims, tokens

    def search_subwords(self, query_embs: np.ndarray, top_k: int = 10) -> tuple[np.ndarray, list[list[VocabToken]]]:
        """Search nearest subword candidates."""
        if self.subword_index is None:
            raise RuntimeError("Subword index is not built.")
        sims, idxs = self.subword_index.search(query_embs, top_k)
        tokens = [[self._subword_lookup[i] for i in row if i in self._subword_lookup] for row in idxs]
        return sims, tokens

    def search_chars(self, query_embs: np.ndarray, top_k: int = 10) -> tuple[np.ndarray, list[list[VocabToken]]]:
        """Search nearest char candidates."""
        if self.char_index is None:
            raise RuntimeError("Char index is not built.")
        sims, idxs = self.char_index.search(query_embs, top_k)
        tokens = [[self._char_lookup[i] for i in row if i in self._char_lookup] for row in idxs]
        return sims, tokens
