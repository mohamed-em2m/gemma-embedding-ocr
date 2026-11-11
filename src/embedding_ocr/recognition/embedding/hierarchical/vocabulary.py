"""Embedding vocabulary builder and manager using EmbeddingGemma 2 tokenizer.

Extracts tokens from the tokenizer, categorizes them into:
1. Complete words (e.g. leading with whitespace marker \u2581 or whole dictionary words)
2. Subwords (morphemes, affixes, partial fragments)
3. Characters (single alphanumeric and ASCII glyphs)

Builds and caches text embeddings and candidate indices.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from embedding_ocr.config import IndexSettings
from embedding_ocr.data.scripts import is_text_in_script, normalize_script_name
from embedding_ocr.recognition.embedding.embedder import Embedder
from embedding_ocr.recognition.embedding.index import FaissIndex, VectorIndex

DEFAULT_HANDWRITING_FONTS = [
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/opentype/dancingscript/DancingScript-Regular.otf",
    "/usr/share/fonts/opentype/kaushanscript/KaushanScript-Regular.otf",
    "/usr/share/fonts/truetype/sjfonts/SteveHand.ttf",
    "/usr/share/fonts/truetype/kristi/Kristi.ttf",
]

DEFAULT_ARABIC_FONTS = [
    "/usr/share/fonts/google-noto-vf/NotoNaskhArabic[wght].ttf",
    "/usr/share/fonts/google-noto-vf/NotoSansArabic[wght].ttf",
    "/usr/share/fonts/google-droid-sans-fonts/DroidKufi-Regular.ttf",
    "/usr/share/fonts/paktype-naskh-basic-fonts/PakTypeNaskhBasic.ttf",
    "/usr/share/fonts/vazirmatn-vf-fonts/Vazirmatn[wght].ttf",
]


def _shape_arabic_fallback(text: str) -> str:
    """Shape Arabic text for renderers without libraqm support.

    The optional ``arabic-reshaper`` + ``python-bidi`` pair is used when
    available. If neither is installed, the raw logical string is returned.
    """
    try:
        import arabic_reshaper  # type: ignore[import-untyped]
        from bidi.algorithm import get_display  # type: ignore[import-untyped]
    except ImportError:
        return text
    return str(get_display(arabic_reshaper.reshape(text)))


def render_prototype_image(
    text: str,
    font_path: str,
    font_size: int = 36,
    shear: float = 0.0,
    pad: int = 12,
    script: str = "latin",
    direction: str | None = None,
    language: str | None = None,
) -> np.ndarray:
    """Render a text string as a synthetic image prototype.

    Args:
        text: Text string to render.
        font_path: Path to .ttf or .otf font.
        font_size: Font size in pt.
        shear: Horizontal slant angle tangent (e.g. 0.2 for italic slant).
        pad: Padding in pixels.
        script: Script of ``text``. Arabic uses RTL rendering when Pillow has
            libraqm, with a reshaper/bidi fallback otherwise.
        direction: Optional Pillow direction override (``"rtl"``/``"ltr"``).
        language: Optional Pillow language override (e.g. ``"ar"``).

    Returns:
        RGB numpy array (H, W, 3) of uint8.
    """
    font: ImageFont.ImageFont | ImageFont.FreeTypeFont
    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception:
        font = ImageFont.load_default()

    normalized_script = normalize_script_name(script)
    if normalized_script == "arabic":
        direction = direction or "rtl"
        language = language or "ar"

    render_text = text
    dummy = Image.new("RGB", (1, 1))
    dummy_draw = ImageDraw.Draw(dummy)
    try:
        bbox = dummy_draw.textbbox(
            (0, 0),
            render_text,
            font=font,
            direction=direction,
            language=language,
        )
    except (OSError, ValueError, KeyError):
        # libraqm is unavailable; fall back to pre-shaped visual-order text.
        if normalized_script == "arabic":
            render_text = _shape_arabic_fallback(text)
        direction = None
        language = None
        bbox = dummy_draw.textbbox((0, 0), render_text, font=font)

    w = max(1, int(bbox[2] - bbox[0]))
    h = max(1, int(bbox[3] - bbox[1]))
    img = Image.new("RGB", (w + pad * 2, h + pad * 2), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text(
        (pad - bbox[0], pad - bbox[1]),
        render_text,
        fill=(0, 0, 0),
        font=font,
        direction=direction,
        language=language,
    )

    if abs(shear) > 1e-4:
        # Affine transform for horizontal slant: x_src = x_tgt - shear * y_tgt
        matrix = (1, -shear, 0, 0, 1, 0)
        img = img.transform(
            img.size,
            Image.Transform.AFFINE,
            matrix,
            Image.Resampling.BILINEAR,
            fillcolor=(255, 255, 255),
        )
    return np.array(img, dtype=np.uint8)


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
        prototype_fonts: list[str] | None = None,
        use_visual_prototypes: bool = False,
        slant_angles: list[float] | None = None,
        script: str = "latin",
        index_settings: IndexSettings | None = None,
    ) -> None:
        self.embedder = embedder
        self.tokenizer_name = tokenizer_name
        self.script = normalize_script_name(script)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        if prototype_fonts is not None:
            self.prototype_fonts = list(prototype_fonts)
        else:
            default_fonts = (
                DEFAULT_ARABIC_FONTS
                if self.script == "arabic"
                else DEFAULT_HANDWRITING_FONTS
            )
            self.prototype_fonts = [f for f in default_fonts if Path(f).exists()]
            if not self.prototype_fonts:
                # Last-resort fallback so visual prototypes still work when the
                # script-specific font package is absent.
                self.prototype_fonts = [
                    f
                    for f in DEFAULT_HANDWRITING_FONTS + DEFAULT_ARABIC_FONTS
                    if Path(f).exists()
                ]
        self.use_visual_prototypes = use_visual_prototypes
        self.slant_angles = slant_angles or [0.0]
        self.index_settings = index_settings or IndexSettings()

        self.word_tokens: list[VocabToken] = []
        self.subword_tokens: list[VocabToken] = []
        self.char_tokens: list[VocabToken] = []

        self.word_index: VectorIndex | None = None
        self.subword_index: VectorIndex | None = None
        self.char_index: VectorIndex | None = None

        self._word_lookup: dict[int, VocabToken] = {}
        self._subword_lookup: dict[int, VocabToken] = {}
        self._char_lookup: dict[int, VocabToken] = {}

    def _new_index(self) -> FaissIndex:
        settings = self.index_settings
        return FaissIndex(
            dim=self.embedder.dim,
            omp_threads=settings.omp_threads,
            backend=settings.backend,
            nlist=settings.nlist,
            nprobe=settings.nprobe,
            quantizer_bits=settings.quantizer_bits,
            train_size=settings.train_size,
        )

    def extract_and_organize_tokens(
        self,
        extra_lexicon: Sequence[str] | None = None,
        max_words: int | None = 25000,
        max_subwords: int | None = 15000,
    ) -> None:
        """Extract valid tokens for the configured script and organize them."""
        from transformers import AutoTokenizer  # type: ignore[import-untyped]

        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        vocab = tokenizer.get_vocab()
        special_ids = set(tokenizer.all_special_ids)

        words_dict: dict[str, VocabToken] = {}
        subwords_dict: dict[str, VocabToken] = {}
        chars_dict: dict[str, VocabToken] = {}

        # 1. Direct fast string parse on SentencePiece tokens without individual decode overhead
        for token, tid in vocab.items():
            if (
                tid in special_ids
                or token.startswith("<")
                or token.startswith("[")
                or "bytefallback" in token
            ):
                continue

            # Strip SentencePiece space markers: ' ', '▁' (U+2581)
            clean = token.lstrip(
                " \u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588\u2580\u2591\u2592\u2593"
            )
            if not clean:
                continue

            # Script-aware printable tokens (Latin by default, Arabic and
            # other supported scripts when configured).
            if not is_text_in_script(clean, self.script):
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
                if (
                    clean_w
                    and is_text_in_script(clean_w, self.script)
                    and clean_w not in chars_dict
                ):
                    words_dict[clean_w] = VocabToken(
                        text=clean_w,
                        token_id=-1,  # priority extra word
                        raw_token=f"\u2581{clean_w}",
                        category="word",
                    )

        # Truncate to desired candidate budget for efficiency (prioritizing custom/extra words)
        word_list = list(words_dict.values())
        if max_words and len(word_list) > max_words:
            # Sort deterministically: priority extra words first (token_id == -1), then by token frequency, then text
            word_list.sort(
                key=lambda t: (-100 if t.token_id == -1 else t.token_id, t.text)
            )
            word_list = word_list[:max_words]

        subword_list = list(subwords_dict.values())
        if max_subwords and len(subword_list) > max_subwords:
            subword_list.sort(key=lambda t: (t.token_id, t.text))
            subword_list = subword_list[:max_subwords]

        char_list = list(chars_dict.values())
        char_list.sort(key=lambda t: (t.token_id, t.text))

        self.word_tokens = word_list
        self.subword_tokens = subword_list
        self.char_tokens = char_list

        self._word_lookup = {i: tok for i, tok in enumerate(self.word_tokens)}
        self._subword_lookup = {i: tok for i, tok in enumerate(self.subword_tokens)}
        self._char_lookup = {i: tok for i, tok in enumerate(self.char_tokens)}

    def build_indexes(self, force_recompute: bool = False) -> None:
        """Precompute and cache embeddings and build FAISS indices for all tiers.

        If `use_visual_prototypes` is True, renders prototypes across the configured
        `prototype_fonts` and `slant_angles`, embedding images instead of pure text strings.
        Maps multiple prototype vectors back to the corresponding token candidates.
        """
        if self.use_visual_prototypes and not self.prototype_fonts:
            msg = (
                "Visual prototypes require at least one installed font; "
                f"none found for script {self.script!r}."
            )
            raise ValueError(msg)

        categories = [
            ("words", self.word_tokens, "word_index", "_word_lookup"),
            ("subwords", self.subword_tokens, "subword_index", "_subword_lookup"),
            ("chars", self.char_tokens, "char_index", "_char_lookup"),
        ]

        # Signature hash to identify visual font configurations. The script is
        # part of the tag so Arabic and Latin caches never collide. The embedder
        # tag includes the model, prompt, tokenizer, and dimension so fine-tuned
        # checkpoints never reuse base-model vectors.
        mode_tag = f"{'visual' if self.use_visual_prototypes else 'text'}_{self.script}"
        fonts_hash = hashlib.md5(
            "".join(self.prototype_fonts + [str(s) for s in self.slant_angles]).encode()
        ).hexdigest()[:8]
        embedder_tag = hashlib.md5(
            (
                f"{self.embedder.name}|{self.embedder.dim}|"
                f"{getattr(self.embedder, 'prompt_name', '')}|"
                f"{self.tokenizer_name}"
            ).encode()
        ).hexdigest()[:8]

        for cat_name, tokens, attr_name, lookup_attr in categories:
            base = (
                f"{cat_name}_{mode_tag}_{fonts_hash}_{embedder_tag}_"
                f"{self.embedder.dim}d"
            )
            emb_file = self.cache_dir / f"{base}.npy"
            meta_file = self.cache_dir / f"{base}_meta.json"
            mapping_file = self.cache_dir / f"{base}_map.json"
            identity_file = self.cache_dir / f"{base}_embedder.json"

            if (
                not force_recompute
                and emb_file.exists()
                and meta_file.exists()
                and mapping_file.exists()
            ):
                cached_meta = json.loads(meta_file.read_text())
                if len(cached_meta) == len(tokens) and all(
                    cached_meta[i].get("text") == tokens[i].text
                    for i in range(len(tokens))
                ):
                    vectors = np.load(emb_file)
                    id_mapping = json.loads(mapping_file.read_text())
                    # Reconstruct lookup table for vector indices
                    proto_lookup: dict[int, VocabToken] = {}
                    for v_idx_str, t_idx in id_mapping.items():
                        proto_lookup[int(v_idx_str)] = tokens[t_idx]
                    setattr(self, lookup_attr, proto_lookup)

                    index = self._new_index()
                    index.build(vectors, np.arange(len(vectors)))
                    setattr(self, attr_name, index)
                    continue

            if not tokens:
                continue

            active_id_mapping: dict[str, int] = {}
            active_proto_lookup: dict[int, VocabToken] = {}

            if not self.use_visual_prototypes:
                # 1. Standard text-only embedding
                texts = [tok.text for tok in tokens]
                vectors = self.embedder.embed_texts(texts)
                for i, tok in enumerate(tokens):
                    active_id_mapping[str(i)] = i
                    active_proto_lookup[i] = tok
            else:
                # 2. Multi-font visual prototypes
                all_images: list[np.ndarray] = []
                img_token_indices: list[int] = []

                for t_idx, tok in enumerate(tokens):
                    for font_p in self.prototype_fonts:
                        for slant in self.slant_angles:
                            img = render_prototype_image(
                                tok.text,
                                font_path=font_p,
                                shear=slant,
                                script=self.script,
                            )
                            all_images.append(img)
                            img_token_indices.append(t_idx)

                # Batch embed visual prototype images
                batch_size = 64
                embs_list: list[np.ndarray] = []
                for b_start in range(0, len(all_images), batch_size):
                    batch_imgs = all_images[b_start : b_start + batch_size]
                    batch_embs = self.embedder.embed_images(batch_imgs)
                    embs_list.append(batch_embs)
                vectors = np.vstack(embs_list)

                for v_idx, t_idx in enumerate(img_token_indices):
                    active_id_mapping[str(v_idx)] = t_idx
                    active_proto_lookup[v_idx] = tokens[t_idx]

            np.save(emb_file, vectors)
            mapping_file.write_text(json.dumps(active_id_mapping))

            # Save metadata
            meta = [
                {"text": t.text, "token_id": t.token_id, "category": t.category}
                for t in tokens
            ]
            meta_file.write_text(json.dumps(meta, indent=2))
            identity_file.write_text(
                json.dumps(
                    {
                        "embedder": self.embedder.name,
                        "dim": self.embedder.dim,
                        "prompt_name": getattr(self.embedder, "prompt_name", None),
                        "tokenizer": self.tokenizer_name,
                        "visual_prototypes": self.use_visual_prototypes,
                        "fonts": self.prototype_fonts,
                        "slant_angles": self.slant_angles,
                    },
                    indent=2,
                )
            )

            setattr(self, lookup_attr, active_proto_lookup)
            index = self._new_index()
            index.build(vectors, np.arange(len(vectors)))
            setattr(self, attr_name, index)

    def _aggregate_search_results(
        self,
        sims: np.ndarray,
        idxs: np.ndarray,
        lookup: dict[int, VocabToken],
        top_k: int,
    ) -> tuple[np.ndarray, list[list[VocabToken]]]:
        """Aggregate multiple prototype hits per token text using max-similarity."""
        agg_sims_list: list[list[float]] = []
        agg_tokens_list: list[list[VocabToken]] = []

        for row_sims, row_idxs in zip(sims, idxs, strict=False):
            seen_texts: dict[str, tuple[float, VocabToken]] = {}
            for s, idx in zip(row_sims, row_idxs, strict=False):
                if idx not in lookup:
                    continue
                tok = lookup[idx]
                sim_val = float(s)
                # Keep max similarity per token text
                if tok.text not in seen_texts or sim_val > seen_texts[tok.text][0]:
                    seen_texts[tok.text] = (sim_val, tok)

            # Sort by descending similarity
            sorted_candidates = sorted(
                seen_texts.values(), key=lambda item: item[0], reverse=True
            )[:top_k]

            agg_sims_list.append([sc[0] for sc in sorted_candidates])
            agg_tokens_list.append([sc[1] for sc in sorted_candidates])

        return np.array(agg_sims_list, dtype=np.float32), agg_tokens_list

    def search_words(
        self, query_embs: np.ndarray, top_k: int = 10
    ) -> tuple[np.ndarray, list[list[VocabToken]]]:
        """Search nearest word candidates with max-per-class aggregation."""
        if self.word_index is None:
            raise RuntimeError("Word index is not built.")
        # Retrieve more raw hits to ensure top_k distinct classes after
        # aggregation. Text prototypes need extra depth because the full
        # tokenizer vocabulary is large and rare words sit far from the head.
        multiplier = (
            len(self.prototype_fonts) * len(self.slant_angles)
            if self.use_visual_prototypes
            else 8
        )
        raw_k = min(max(top_k * multiplier, top_k), 200)
        sims, idxs = self.word_index.search(query_embs, raw_k)
        return self._aggregate_search_results(
            sims, idxs, self._word_lookup, top_k=top_k
        )

    def search_subwords(
        self, query_embs: np.ndarray, top_k: int = 10
    ) -> tuple[np.ndarray, list[list[VocabToken]]]:
        """Search nearest subword candidates with max-per-class aggregation."""
        if self.subword_index is None:
            raise RuntimeError("Subword index is not built.")
        multiplier = (
            len(self.prototype_fonts) * len(self.slant_angles)
            if self.use_visual_prototypes
            else 8
        )
        raw_k = min(max(top_k * multiplier, top_k), 200)
        sims, idxs = self.subword_index.search(query_embs, raw_k)
        return self._aggregate_search_results(
            sims, idxs, self._subword_lookup, top_k=top_k
        )

    def search_chars(
        self, query_embs: np.ndarray, top_k: int = 10
    ) -> tuple[np.ndarray, list[list[VocabToken]]]:
        """Search nearest char candidates with max-per-class aggregation."""
        if self.char_index is None:
            raise RuntimeError("Char index is not built.")
        multiplier = (
            len(self.prototype_fonts) * len(self.slant_angles)
            if self.use_visual_prototypes
            else 8
        )
        raw_k = min(max(top_k * multiplier, top_k), 200)
        sims, idxs = self.char_index.search(query_embs, raw_k)
        return self._aggregate_search_results(
            sims, idxs, self._char_lookup, top_k=top_k
        )
