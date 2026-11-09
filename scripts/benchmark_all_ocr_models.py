"""Unified benchmark comparing ALL OCR algorithms (classical, embedding, conventional, and VLMs).

Evaluated algorithms & engines:
1. Classical CV:
   * ``ncc``       - Normalized Cross-Correlation Template Matching
   * ``ssim``      - Structural Similarity Index (SSIM)
2. Visual Embedding (EmbeddingGemma 2, SigLIP SO400M, Jina v4):
   * ``gemma``     - EmbeddingGemma 2 + Prototype Matcher (Max-Agg + Height Cue)
   * ``gemma-1nn`` - EmbeddingGemma 2 (1-NN Vector Search)
   * ``gemma-knn`` - EmbeddingGemma 2 (5-NN Distance-Weighted Vote)
   * ``siglip``    - SigLIP SO400M + Prototype Matcher
   * ``jina``      - Jina Embeddings v4 + Prototype Matcher
3. Conventional OCR:
   * ``tesseract`` - Tesseract OCR engine (pytesseract)
   * ``easyocr``   - EasyOCR (PyTorch CRAFT + CRNN)
   * ``paddle``    - PaddleOCR (PP-OCRv4)
4. New Generative OCR VLMs:
   * ``lighton``   - LightOnOCR-3-0.8B (Transformers)
   * ``surya``     - datalab-to/surya-ocr-2.5 (surya-ocr package)
   * ``dots``      - dots-studio/dots.mocr (Transformers + SDPA, opt-in)
   * ``glm``       - zai-org/GLM-OCR (Transformers)
   * ``oxr``       - SoMarkAI/OXR-1.0 (SoMark parse API / OpenAI-compatible)

Usage:
    uv run --no-sync python scripts/benchmark_all_ocr_models.py --help
    uv run --no-sync python scripts/benchmark_all_ocr_models.py --limit 5
    uv run --no-sync python scripts/benchmark_all_ocr_models.py --engines gemma tesseract easyocr
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import gc
import json
import re
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np

from embedding_ocr.data.glyph_renderer import (
    ALPHANUMERIC_CLASSES,
    LATIN_CLASSES,
    render_glyph,
    render_glyph_with_features,
)
from embedding_ocr.recognition.base import RecognitionEngine

DEFAULT_INDEX_FONTS: list[str] = [
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
]
DEFAULT_TEST_FONTS: list[str] = [
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSerif-BoldItalic.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Italic.ttf",
]

ALL_ENGINES = (
    "gemma",
    "gemma-1nn",
    "gemma-knn",
    "siglip",
    "jina",
    "ncc",
    "ssim",
    "tesseract",
    "easyocr",
    "paddle",
    "lighton",
    "surya",
    "dots",
    "glm",
    "oxr",
)
DEFAULT_ENGINES = tuple(name for name in ALL_ENGINES if name != "dots")


class Predictor(Protocol):
    """Common evaluation interface."""

    name: str

    def predict(
        self, images: list[np.ndarray], relative_heights: list[float]
    ) -> list[str]:
        """Return recognized text for each input image."""
        ...


def clean_prediction(text: str) -> str:
    """Strip wrappers, markdown fences, and echoes."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = [line for line in cleaned.splitlines() if not line.startswith("```")]
        cleaned = "\n".join(lines).strip()

    if cleaned.startswith(("{", "[")):
        with contextlib.suppress(json.JSONDecodeError):
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                cleaned = str(
                    parsed.get("text")
                    or parsed.get("content")
                    or next(iter(parsed.values()), "")
                )
            elif isinstance(parsed, list) and parsed:
                cleaned = str(parsed[0])

    for prefix in ("Text Recognition:", "Text:", "OCR:"):
        if prefix in cleaned:
            cleaned = cleaned.split(prefix, 1)[-1].strip()

    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    return lines[0] if lines else ""


def extract_glyph(text: str) -> str:
    """Extract a single target glyph character from chatty outputs."""
    if len(text) == 1:
        return text

    stripped = text.strip(" `'\".,:;!?()[]{}")
    if len(stripped) == 1 and stripped.isalnum():
        return stripped

    quoted = re.findall(r"['\"`]([A-Za-z0-9])['\"`]", text)
    if len(set(quoted)) == 1:
        return quoted[0]

    patterns = (
        r"(?:character|letter|glyph|text)\s*(?:is|:)\s*['\"`]?([A-Za-z0-9])",
        r"^['\"`]?([A-Za-z0-9])['\"`]?\s*(?:$|\.)",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1)

    candidates = [ch for ch in text if ch.isalnum()]
    unique = list(dict.fromkeys(candidates))
    return unique[0] if len(unique) == 1 else text


@dataclass
class EngineResult:
    name: str
    category: str
    status: str = "ok"
    exact: int = 0
    case_insensitive: int = 0
    extracted: int = 0
    extracted_case_insensitive: int = 0
    total: int = 0
    latency_ms: float = 0.0
    error: str = ""
    errors: list[dict[str, str]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if self.total:
            data["exact_pct"] = round(self.exact / self.total * 100.0, 2)
            data["case_insensitive_pct"] = round(
                self.case_insensitive / self.total * 100.0, 2
            )
            data["extracted_pct"] = round(self.extracted / self.total * 100.0, 2)
            data["extracted_case_insensitive_pct"] = round(
                self.extracted_case_insensitive / self.total * 100.0, 2
            )
        return data


# --- Classical CV Predictors ---


class NCCPredictor:
    """Normalized Cross-Correlation Template Matching."""

    name = "Pixel NCC Template Match"

    def __init__(self, gallery_images: list[np.ndarray], gallery_labels: list[str]):
        self.gallery_grays = [
            cv2.cvtColor(im, cv2.COLOR_RGB2GRAY) for im in gallery_images
        ]
        self.gallery_labels = gallery_labels

    def predict(
        self, images: list[np.ndarray], relative_heights: list[float]
    ) -> list[str]:
        preds: list[str] = []
        for img in images:
            q_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            best_score = -1.0
            best_label = ""
            for t_gray, label in zip(
                self.gallery_grays, self.gallery_labels, strict=False
            ):
                res = cv2.matchTemplate(q_gray, t_gray, cv2.TM_CCOEFF_NORMED)
                score = float(res[0][0])
                if score > best_score:
                    best_score = score
                    best_label = label
            preds.append(best_label)
        return preds


class SSIMPredictor:
    """Structural Similarity Index (SSIM) Matcher."""

    name = "Structural Similarity (SSIM)"

    def __init__(self, gallery_images: list[np.ndarray], gallery_labels: list[str]):
        from skimage.metrics import structural_similarity as ssim_fn

        self.ssim_fn = ssim_fn
        self.gallery_grays = [
            cv2.cvtColor(im, cv2.COLOR_RGB2GRAY) for im in gallery_images
        ]
        self.gallery_labels = gallery_labels

    def predict(
        self, images: list[np.ndarray], relative_heights: list[float]
    ) -> list[str]:
        preds: list[str] = []
        for img in images:
            q_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            best_score = -2.0
            best_label = ""
            for t_gray, label in zip(
                self.gallery_grays, self.gallery_labels, strict=False
            ):
                score = float(self.ssim_fn(q_gray, t_gray))
                if score > best_score:
                    best_score = score
                    best_label = label
            preds.append(best_label)
        return preds


# --- Embedding Predictors ---


class EmbeddingPredictor:
    """Embedding backend + Prototype Matcher (Max-Agg + Height)."""

    def __init__(
        self,
        embedder_name: str,
        index_fonts: list[str],
        classes: list[str],
        dim: int = 768,
        top_k: int = 25,
        device: str = "auto",
        strategy: str = "matcher",
    ) -> None:
        from embedding_ocr.recognition.embedding.embedder import (
            EmbeddingGemma2,
            JinaEmbeddingsV4,
            SiglipEmbedder,
        )
        from embedding_ocr.recognition.embedding.index import FaissIndex
        from embedding_ocr.recognition.embedding.matcher import PrototypeMatcher
        from embedding_ocr.recognition.embedding.prototypes import PrototypeDB

        self.strategy = strategy
        self.top_k = top_k
        if embedder_name == "gemma":
            embedder = EmbeddingGemma2(dim=dim, device=device)
            label = "EmbeddingGemma"
        elif embedder_name == "siglip":
            embedder = SiglipEmbedder(dim=dim, device=device)
            label = "SigLIP SO400M"
        elif embedder_name == "jina":
            embedder = JinaEmbeddingsV4(dim=dim, device=device)
            label = "Jina Embeddings v4"
        else:
            msg = f"Unknown embedder: {embedder_name}"
            raise ValueError(msg)

        self.embedder = embedder
        self.db = PrototypeDB(
            embedder=self.embedder,
            index=FaissIndex(dim=self.embedder.dim),
            classes=classes,
        )
        self.db.build_from_fonts(index_fonts)
        self.matcher = PrototypeMatcher()
        self.name = f"{label} ({self.embedder.dim}d)"

        if strategy == "1nn":
            self.name = f"{label} ({self.embedder.dim}d, 1-NN)"
        elif strategy == "knn":
            self.name = f"{label} ({self.embedder.dim}d, 5-NN Weighted)"
        else:
            self.name = f"{label} ({self.embedder.dim}d, Max-Agg + Matcher)"

    def predict(
        self, images: list[np.ndarray], relative_heights: list[float]
    ) -> list[str]:
        embeddings = self.embedder.embed_images(images)
        similarities, labels = self.db.search(embeddings, top_k=self.top_k)
        predictions: list[str] = []

        for i, rel_height in enumerate(relative_heights):
            sims_i = similarities[i]
            labs_i = labels[i]

            if self.strategy == "1nn":
                predictions.append(labs_i[0])
            elif self.strategy == "knn":
                weights: collections.defaultdict[str, float] = collections.defaultdict(
                    float
                )
                for s, l_char in zip(sims_i[:5], labs_i[:5], strict=False):
                    weights[l_char] += max(0.0, float(s))
                best_char = max(weights.items(), key=lambda kv: kv[1])[0]
                predictions.append(best_char)
            else:
                match = self.matcher.match_character(
                    similarities=sims_i,
                    labels=labs_i,
                    relative_height=rel_height,
                )
                predictions.append(match.character)

        return predictions


# --- Generic Adapter Predictor ---


class EnginePredictor:
    """Wrapper calling any RecognitionEngine."""

    def __init__(self, name: str, engine: RecognitionEngine) -> None:
        self.name = name
        self.engine = engine

    def predict(
        self, images: list[np.ndarray], relative_heights: list[float]
    ) -> list[str]:
        return [self.engine.recognize(img).text for img in images]


def load_engine_adapter(name: str, args: argparse.Namespace) -> RecognitionEngine:
    """Instantiate adapter by name."""
    if name == "tesseract":
        from embedding_ocr.recognition.tesseract import TesseractRecognizer

        return TesseractRecognizer()

    if name == "easyocr":
        from embedding_ocr.recognition.easyocr import EasyOCRRecognizer

        return EasyOCRRecognizer(gpu=args.device != "cpu")

    if name == "paddle":
        from embedding_ocr.recognition.paddle import PaddleRecognizer

        return PaddleRecognizer(use_gpu=args.device != "cpu")

    if name == "lighton":
        from embedding_ocr.recognition.lighton_ocr import LightOnOCRRecognizer

        return LightOnOCRRecognizer(
            model_id=args.lighton_model,
            device=args.device,
            prompt=args.lighton_prompt,
        )

    if name == "dots":
        from embedding_ocr.recognition.dots_mocr import DotsMOCRRecognizer

        return DotsMOCRRecognizer(
            model_id=args.dots_model,
            device=args.device,
            prompt=args.dots_prompt,
        )

    if name == "surya":
        from embedding_ocr.recognition.surya_ocr import SuryaOCRRecognizer

        return SuryaOCRRecognizer(
            model_id=args.surya_model,
            device=args.device,
            backend=args.surya_backend,
            inference_url=args.surya_inference_url,
        )

    if name == "glm":
        from embedding_ocr.recognition.glm_ocr import GLMOCRRecognizer

        return GLMOCRRecognizer(
            model_id=args.glm_model,
            device=args.device,
        )

    if name == "oxr":
        from embedding_ocr.recognition.oxr import OXRRecognizer

        return OXRRecognizer(
            base_url=args.oxr_base_url,
            model=args.oxr_model,
            prompt=args.oxr_prompt,
            backend=args.oxr_backend,
            max_tokens=args.oxr_max_tokens,
            timeout=args.oxr_timeout,
            retries=args.oxr_retries,
            retry_backoff=args.oxr_retry_backoff,
            request_delay=args.oxr_request_delay,
        )

    msg = f"Unknown engine adapter: {name}"
    raise ValueError(msg)


def release_memory() -> None:
    gc.collect()
    with contextlib.suppress(ImportError):
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def render_queries(
    test_fonts: list[str],
    classes: list[str],
    limit: int | None,
) -> tuple[list[np.ndarray], list[str], list[float]]:
    images: list[np.ndarray] = []
    labels: list[str] = []
    relative_heights: list[float] = []

    for f_p in test_fonts:
        for ch in classes:
            if limit is not None and len(images) >= limit:
                return images, labels, relative_heights
            pil_img, feats = render_glyph_with_features(ch, f_p)
            images.append(np.array(pil_img, dtype=np.uint8))
            labels.append(ch)
            relative_heights.append(feats["relative_height"])

    return images, labels, relative_heights


def evaluate(
    predictor: Predictor,
    category: str,
    images: list[np.ndarray],
    labels: list[str],
    relative_heights: list[float],
    verbose: bool,
) -> EngineResult:
    result = EngineResult(name=predictor.name, category=category, total=len(labels))
    t0 = time.perf_counter()
    try:
        raw_predictions = predictor.predict(images, relative_heights)
    except Exception as exc:
        result.status = "unavailable"
        result.error = f"{type(exc).__name__}: {exc}"
        if verbose:
            traceback.print_exc()
        return result

    elapsed = time.perf_counter() - t0
    result.latency_ms = (elapsed / max(1, len(labels))) * 1000.0

    for label, raw in zip(labels, raw_predictions, strict=False):
        cleaned = clean_prediction(raw)
        extracted = extract_glyph(cleaned)
        if cleaned == label:
            result.exact += 1
        if cleaned.lower() == label.lower():
            result.case_insensitive += 1
        if extracted == label:
            result.extracted += 1
        if extracted.lower() == label.lower():
            result.extracted_case_insensitive += 1
        if cleaned != label and len(result.errors) < 20:
            result.errors.append(
                {"target": label, "prediction": cleaned, "raw": raw[:100]}
            )

    return result


def write_reports(
    results: list[EngineResult], n_queries: int, output_prefix: Path
) -> None:
    payload = {
        "n_queries": n_queries,
        "results": [r.as_dict() for r in results],
    }
    json_path = output_prefix.with_suffix(".json")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2))

    md_path = output_prefix.with_suffix(".md")
    lines = [
        "# Unified OCR Algorithm & Model Benchmark Comparison",
        "",
        f"Queries: **{n_queries}** rendered glyphs from held-out typography fonts.",
        "",
        "| Category | Algorithm / Engine | Status | Exact Top-1 | Case-Insens | Extracted Glyph | Latency (ms) |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: |",
    ]

    for r in results:
        if r.status != "ok" or not r.total:
            lines.append(f"| {r.category} | {r.name} | *unavailable* | - | - | - | - |")
            continue
        d = r.as_dict()
        lines.append(
            f"| {r.category} | {r.name} | **ok** | {d['exact_pct']:.1f}% | "
            f"{d['case_insensitive_pct']:.1f}% | {d['extracted_pct']:.1f}% | {r.latency_ms:.1f} ms |"
        )

    lines.append("")
    unavail = [r for r in results if r.status != "ok"]
    if unavail:
        lines.append("## Unavailable Engines / Services")
        lines.append("")
        for r in unavail:
            lines.append(f"- **{r.name}** ({r.category}): `{r.error}`")
        lines.append("")

    lines.append(
        "Notes: `Extracted Glyph` evaluates robust string normalization from chatty generative outputs. "
        "`Exact Top-1` requires direct character equality."
    )
    md_path.write_text("\n".join(lines) + "\n")
    print(f"\nSaved results:\n  - {json_path}\n  - {md_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--engines",
        nargs="+",
        default=["all"],
        choices=[*ALL_ENGINES, "all"],
        help=(
            "Engines/algorithms to evaluate. 'all' runs every default engine; "
            "list 'dots' explicitly to include the opt-in dots.mocr model."
        ),
    )
    parser.add_argument(
        "--classes",
        choices=["latin", "alnum"],
        default="latin",
        help="Character class set.",
    )
    parser.add_argument(
        "--index-fonts",
        nargs="+",
        default=None,
    )
    parser.add_argument(
        "--test-fonts",
        nargs="+",
        default=None,
    )
    parser.add_argument(
        "--font-list-file",
        type=Path,
        default=Path("data/standard_clean_fonts.txt"),
        help="Path to clean fonts catalog file to draw index and test fonts from.",
    )
    parser.add_argument(
        "--num-index-fonts",
        type=int,
        default=5,
        help="Number of index prototype fonts if auto-splitting.",
    )
    parser.add_argument(
        "--num-test-fonts",
        type=int,
        default=25,
        help="Number of test fonts if auto-splitting.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit total queries.")
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda")
    parser.add_argument("--gemma-dim", type=int, default=768)
    parser.add_argument(
        "--siglip-dim",
        type=int,
        default=1152,
        help="SigLIP SO400M embedding dimension (native 1152).",
    )
    parser.add_argument(
        "--jina-dim",
        type=int,
        default=2048,
        help="Jina v4 dimension: 128, 256, 512, 1024 or 2048 (native).",
    )
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument("--lighton-model", default="lightonai/LightOnOCR-3-0.8B")
    parser.add_argument("--lighton-prompt", default="")
    parser.add_argument("--dots-model", default="dots-studio/dots.mocr")
    parser.add_argument(
        "--dots-prompt", default="Output the single character in this image."
    )
    parser.add_argument("--surya-model", default="datalab-to/surya-ocr-2.5")
    parser.add_argument(
        "--surya-backend",
        default=None,
        choices=["vllm", "llamacpp", "mlx"],
        help="Inference backend for surya-ocr (default: auto).",
    )
    parser.add_argument(
        "--surya-inference-url",
        default=None,
        help="Reuse an already-running Surya server (SURYA_INFERENCE_URL).",
    )
    parser.add_argument("--glm-model", default="zai-org/GLM-OCR")
    parser.add_argument(
        "--oxr-base-url",
        default="https://free-oxr.somark.ai/api/v1",
        help=(
            "OXR endpoint: free API (default), local parsing API "
            "(http://127.0.0.1:8000/v1), or local vLLM (http://127.0.0.1:8080/v1)."
        ),
    )
    parser.add_argument("--oxr-model", default="SoMarkAI/OXR-1.0")
    parser.add_argument(
        "--oxr-backend",
        default="auto",
        choices=["auto", "parse", "chat"],
        help="OXR protocol: SoMark /parse/sync, OpenAI /chat/completions, or auto.",
    )
    parser.add_argument(
        "--oxr-prompt",
        default="Output the single character in this image.",
    )
    parser.add_argument("--oxr-max-tokens", type=int, default=32)
    parser.add_argument("--oxr-timeout", type=float, default=240.0)
    parser.add_argument("--oxr-retries", type=int, default=3)
    parser.add_argument("--oxr-retry-backoff", type=float, default=2.0)
    parser.add_argument("--oxr-request-delay", type=float, default=0.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/unified_ocr_comparison"),
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = list(DEFAULT_ENGINES) if "all" in args.engines else args.engines
    classes = LATIN_CLASSES if args.classes == "latin" else ALPHANUMERIC_CLASSES

    # Determine index and test fonts
    index_fonts = args.index_fonts
    test_fonts = args.test_fonts

    if (not index_fonts or not test_fonts) and args.font_list_file.exists():
        raw_fonts = [
            line.strip()
            for line in args.font_list_file.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]
        valid_fonts = [p for p in raw_fonts if Path(p).exists()]
        if valid_fonts:
            if not index_fonts:
                index_fonts = valid_fonts[: args.num_index_fonts]
            if not test_fonts:
                test_fonts = valid_fonts[
                    args.num_index_fonts : args.num_index_fonts + args.num_test_fonts
                ]

    if not index_fonts:
        index_fonts = DEFAULT_INDEX_FONTS
    if not test_fonts:
        test_fonts = DEFAULT_TEST_FONTS

    args.index_fonts = index_fonts
    args.test_fonts = test_fonts

    print("=" * 80)
    print("UNIFIED OCR BENCHMARK: Classical CV, EmbeddingGemma, Traditional OCR & VLMs")
    print("=" * 80)
    print(
        f"Index Prototype Fonts ({len(args.index_fonts)}): {[Path(f).name for f in args.index_fonts[:5]]}..."
    )
    print(
        f"Test Held-Out Fonts   ({len(args.test_fonts)}): {[Path(f).name for f in args.test_fonts[:5]]}..."
    )

    images, labels, relative_heights = render_queries(
        args.test_fonts, classes, args.limit
    )
    print(
        f"Rendered {len(images)} queries "
        f"({len(classes)} classes x {len(args.test_fonts)} test fonts)."
    )

    # Render prototype gallery for template matching
    gallery_images: list[np.ndarray] = []
    gallery_labels: list[str] = []
    for f_p in args.index_fonts:
        for ch in classes:
            pil_img = render_glyph(ch, f_p, canvas_size=64, font_size=44)
            gallery_images.append(np.array(pil_img))
            gallery_labels.append(ch)

    category_map = {
        "ncc": "Classical CV",
        "ssim": "Classical CV",
        "gemma": "EmbeddingOCR",
        "gemma-1nn": "EmbeddingOCR",
        "gemma-knn": "EmbeddingOCR",
        "siglip": "EmbeddingOCR",
        "jina": "EmbeddingOCR",
        "tesseract": "Conventional OCR",
        "easyocr": "Conventional OCR",
        "paddle": "Conventional OCR",
        "lighton": "Generative VLM",
        "surya": "Generative VLM",
        "dots": "Generative VLM",
        "glm": "Generative VLM",
        "oxr": "Generative VLM",
    }

    results: list[EngineResult] = []

    for name in selected:
        cat = category_map.get(name, "Other")
        print(f"\n=== [{cat}] {name} ===")
        try:
            if name == "ncc":
                predictor: Predictor = NCCPredictor(gallery_images, gallery_labels)
            elif name == "ssim":
                predictor = SSIMPredictor(gallery_images, gallery_labels)
            elif name in (
                "gemma",
                "gemma-1nn",
                "gemma-knn",
                "siglip",
                "jina",
            ):
                if name.startswith("gemma"):
                    embedder_name = "gemma"
                    dim = args.gemma_dim
                    strategy = (
                        "matcher"
                        if name == "gemma"
                        else ("1nn" if name == "gemma-1nn" else "knn")
                    )
                elif name == "siglip":
                    embedder_name = "siglip"
                    dim = args.siglip_dim
                    strategy = "matcher"
                else:
                    embedder_name = "jina"
                    dim = args.jina_dim
                    strategy = "matcher"
                predictor: Predictor = EmbeddingPredictor(
                    embedder_name=embedder_name,
                    index_fonts=args.index_fonts,
                    classes=classes,
                    dim=dim,
                    top_k=args.top_k,
                    device=args.device,
                    strategy=strategy,
                )
            else:
                engine = load_engine_adapter(name, args)
                predictor = EnginePredictor(name=name, engine=engine)
        except Exception as exc:
            res = EngineResult(
                name=name,
                category=cat,
                status="unavailable",
                total=len(labels),
                error=f"{type(exc).__name__}: {exc}",
            )
            if args.verbose:
                traceback.print_exc()
            print(f"  unavailable: {res.error}")
            results.append(res)
            release_memory()
            continue

        res = evaluate(predictor, cat, images, labels, relative_heights, args.verbose)
        if res.status == "ok":
            d = res.as_dict()
            print(
                "  Exact: {exact_pct:.1f}% | Case-Insens: {ci:.1f}% | "
                "Extracted: {ext:.1f}% | Latency: {lat:.1f} ms/item".format(
                    exact_pct=d["exact_pct"],
                    ci=d["case_insensitive_pct"],
                    ext=d["extracted_pct"],
                    lat=res.latency_ms,
                )
            )
        else:
            print(f"  unavailable: {res.error}")

        results.append(res)
        del predictor
        release_memory()

    write_reports(results, len(labels), args.output)


if __name__ == "__main__":
    main()
