"""One-shot comparison of embedding models vs new generative OCR VLMs.

Evaluated engines:
  * ``gemma``  - EmbeddingGemma 2 + FAISS prototype matching (this project)
  * ``siglip`` - SigLIP SO400M + the same prototype matching system
  * ``jina``   - Jina Embeddings v4 + the same prototype matching system
  * ``lighton`` - lightonai/LightOnOCR-3-0.8B (transformers)
  * ``surya``  - datalab-to/surya-ocr-2.5 (surya-ocr package)
  * ``oxr``    - SoMarkAI/OXR-1.0 (SoMark parse API or OpenAI-compatible vLLM)
  * ``dots``   - dots-studio/dots.mocr (transformers, opt-in)

All engines see the same rendered Latin glyph queries.  Models that cannot
be loaded (missing optional dependency, absent GPU, no OXR server) are
reported as ``unavailable`` instead of aborting the whole run, so a single
command always produces a comparison table for whatever is runnable.

Examples:
    uv run python scripts/compare_new_ocr_models.py
    uv run python scripts/compare_new_ocr_models.py --engines gemma lighton
    uv run python scripts/compare_new_ocr_models.py \
        --oxr-base-url http://127.0.0.1:8080/v1 --limit 52
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import re
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from embedding_ocr.data.glyph_renderer import (
    ALPHANUMERIC_CLASSES,
    LATIN_CLASSES,
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

ENGINE_NAMES = ("gemma", "siglip", "jina", "lighton", "surya", "oxr", "dots")
DEFAULT_ENGINES = ("gemma", "siglip", "jina", "lighton", "surya", "oxr")


class Predictor(Protocol):
    """Minimal interface used by the benchmark loop."""

    name: str

    def predict(
        self, images: list[np.ndarray], relative_heights: list[float]
    ) -> list[str]:
        """Return one raw prediction per input image."""
        ...


def _json_to_text(value: Any) -> str:
    """Flatten a parsed JSON layout into plain text."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("text", "content", "value", "transcription"):
            if key in value:
                return _json_to_text(value[key])
        return " ".join(_json_to_text(v) for v in value.values())
    if isinstance(value, list):
        return " ".join(_json_to_text(v) for v in value)
    return str(value)


def clean_prediction(text: str) -> str:
    """Remove markdown fences, JSON wrappers and prompt echoes."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = [line for line in cleaned.splitlines() if not line.startswith("```")]
        cleaned = "\n".join(lines).strip()

    if cleaned.startswith(("{", "[")):
        with contextlib.suppress(json.JSONDecodeError):
            cleaned = _json_to_text(json.loads(cleaned)).strip()

    for marker in ("Text Recognition:", "Text:", "OCR:"):
        if marker in cleaned:
            cleaned = cleaned.split(marker, 1)[-1].strip()

    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    return lines[0] if lines else ""


def extract_glyph(text: str) -> str:
    """Best-effort extraction of a single character from a chatty output."""
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
    """Aggregate result for one engine."""

    name: str
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
        """Return a JSON-serialisable view with percentages."""
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


class EmbeddingPredictor:
    """Embedding backend + FAISS prototype matcher."""

    def __init__(
        self,
        embedder_name: str,
        index_fonts: list[str],
        classes: list[str],
        dim: int = 768,
        top_k: int = 25,
        device: str = "auto",
    ) -> None:
        from embedding_ocr.recognition.embedding.embedder import (
            EmbeddingGemma2,
            JinaEmbeddingsV4,
            SiglipEmbedder,
        )
        from embedding_ocr.recognition.embedding.index import FaissIndex
        from embedding_ocr.recognition.embedding.matcher import PrototypeMatcher
        from embedding_ocr.recognition.embedding.prototypes import PrototypeDB

        if embedder_name == "gemma":
            embedder = EmbeddingGemma2(dim=dim, device=device)
        elif embedder_name == "siglip":
            embedder = SiglipEmbedder(dim=dim, device=device)
        elif embedder_name == "jina":
            embedder = JinaEmbeddingsV4(dim=dim, device=device)
        else:
            msg = f"Unknown embedder: {embedder_name}"
            raise ValueError(msg)

        self.embedder = embedder
        self.name = f"{embedder.name} ({self.embedder.dim}d)"
        self.db = PrototypeDB(
            embedder=self.embedder,
            index=FaissIndex(dim=self.embedder.dim),
            classes=classes,
        )
        self.db.build_from_fonts(index_fonts)
        self.matcher = PrototypeMatcher()
        self.top_k = top_k

    def predict(
        self, images: list[np.ndarray], relative_heights: list[float]
    ) -> list[str]:
        embeddings = self.embedder.embed_images(images)
        similarities, labels = self.db.search(embeddings, top_k=self.top_k)
        predictions: list[str] = []
        for i, rel_height in enumerate(relative_heights):
            match = self.matcher.match_character(
                similarities=similarities[i],
                labels=labels[i],
                relative_height=rel_height,
            )
            predictions.append(match.character)
        return predictions


class EnginePredictor:
    """Wrap a single RecognitionEngine and call it glyph by glyph."""

    def __init__(self, name: str, engine: RecognitionEngine) -> None:
        self.name = name
        self.engine = engine

    def predict(
        self, images: list[np.ndarray], relative_heights: list[float]
    ) -> list[str]:
        predictions: list[str] = []
        for image in images:
            result = self.engine.recognize(image)
            predictions.append(result.text)
        return predictions


def load_generic_engine(name: str, args: argparse.Namespace) -> RecognitionEngine:
    """Lazily import and construct one of the VLM adapters."""
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
        )
    if name == "oxr":
        from embedding_ocr.recognition.oxr import OXRRecognizer

        return OXRRecognizer(
            base_url=args.oxr_base_url,
            model=args.oxr_model,
            prompt=args.oxr_prompt,
            backend=args.oxr_backend,
        )
    msg = f"Unknown engine: {name}"
    raise ValueError(msg)


def render_queries(
    test_fonts: list[str],
    classes: list[str],
    limit: int | None,
) -> tuple[list[np.ndarray], list[str], list[float]]:
    """Render all glyph queries and collect relative-height features."""
    images: list[np.ndarray] = []
    labels: list[str] = []
    relative_heights: list[float] = []

    for font_path in test_fonts:
        for char in classes:
            if limit is not None and len(images) >= limit:
                return images, labels, relative_heights
            pil_image, features = render_glyph_with_features(char, font_path)
            images.append(np.array(pil_image, dtype=np.uint8))
            labels.append(char)
            relative_heights.append(features["relative_height"])
    return images, labels, relative_heights


def evaluate(
    predictor: Predictor,
    images: list[np.ndarray],
    labels: list[str],
    relative_heights: list[float],
    verbose: bool,
) -> EngineResult:
    """Run one predictor over all queries and aggregate accuracy."""
    result = EngineResult(name=predictor.name, total=len(labels))
    started = time.perf_counter()
    try:
        raw_predictions = predictor.predict(images, relative_heights)
    except Exception as exc:
        result.status = "unavailable"
        result.error = f"{type(exc).__name__}: {exc}"
        if verbose:
            traceback.print_exc()
        return result

    elapsed = time.perf_counter() - started
    result.latency_ms = elapsed / max(1, len(labels)) * 1000.0

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
        if cleaned != label and len(result.errors) < 50:
            result.errors.append(
                {"target": label, "prediction": cleaned, "raw": raw[:200]}
            )
    return result


def _release_memory() -> None:
    """Free Python and CUDA memory before loading the next large model."""
    gc.collect()
    with contextlib.suppress(ImportError):
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def write_markdown(results: list[EngineResult], path: Path, n_queries: int) -> None:
    """Write a compact human-readable report."""
    lines = [
        "# Embedding models vs generative OCR VLMs",
        "",
        f"Queries: **{n_queries}** rendered Latin glyphs from held-out fonts.",
        "",
        "| Engine | Status | Exact | Case-insensitive | Extracted glyph | Latency |",
        "| :--- | :--- | :---: | :---: | :---: | :---: |",
    ]
    for result in results:
        if result.status != "ok" or not result.total:
            lines.append(f"| {result.name} | unavailable | - | - | - | - |")
            continue
        data = result.as_dict()
        lines.append(
            "| {name} | ok | {exact:.1f}% | {ci:.1f}% | {ext:.1f}% | "
            "{lat:.1f} ms |".format(
                name=result.name,
                exact=data["exact_pct"],
                ci=data["case_insensitive_pct"],
                ext=data["extracted_pct"],
                lat=result.latency_ms,
            )
        )
    lines.append("")
    unavailable = [r for r in results if r.status != "ok"]
    if unavailable:
        lines.append("## Unavailable engines")
        lines.append("")
        for result in unavailable:
            lines.append(f"- **{result.name}**: `{result.error}`")
        lines.append("")
    lines.append(
        "`Extracted glyph` accepts a single character recovered from a chatty "
        "or JSON response; `Exact` requires the cleaned response to equal the "
        "target character."
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--engines",
        nargs="+",
        default=list(DEFAULT_ENGINES),
        choices=[*ENGINE_NAMES, "all"],
        help=(
            "Engines to run (default: gemma siglip jina lighton surya oxr). "
            "'all' also includes the opt-in dots engine."
        ),
    )
    parser.add_argument(
        "--classes",
        choices=["latin", "alnum"],
        default="latin",
        help="Character set to evaluate (default: 52 Latin letters).",
    )
    parser.add_argument(
        "--index-fonts",
        nargs="+",
        default=DEFAULT_INDEX_FONTS,
        help="Fonts used to build the EmbeddingGemma prototype index.",
    )
    parser.add_argument(
        "--test-fonts",
        nargs="+",
        default=DEFAULT_TEST_FONTS,
        help="Held-out fonts used for queries.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Stop after N queries (smoke testing).",
    )
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
    parser.add_argument(
        "--lighton-prompt",
        default="",
        help="Empty string is LightOnOCR's in-distribution transcription mode.",
    )
    parser.add_argument("--dots-model", default="dots-studio/dots.mocr")
    parser.add_argument(
        "--dots-prompt",
        default="Output the single character in this image.",
    )
    parser.add_argument("--surya-model", default="datalab-to/surya-ocr-2.5")
    parser.add_argument(
        "--surya-backend",
        default=None,
        choices=["vllm", "llamacpp", "mlx"],
        help="Inference backend for surya-ocr (default: auto).",
    )
    parser.add_argument(
        "--oxr-base-url",
        default="https://free-oxr.somark.ai/api/v1",
        help=(
            "OXR endpoint. Use the free API (default), a local parsing API "
            "(http://127.0.0.1:8000/v1) or a local vLLM server "
            "(http://127.0.0.1:8080/v1)."
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
        default="Extract all text from this image. Output only the text.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/new_ocr_models_comparison"),
        help="Output prefix; writes .json and .md.",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print tracebacks on failure."
    )
    return parser.parse_args()


def main() -> None:
    """Run the full comparison and write JSON + Markdown reports."""
    args = parse_args()
    selected = list(ENGINE_NAMES) if "all" in args.engines else args.engines
    classes = LATIN_CLASSES if args.classes == "latin" else ALPHANUMERIC_CLASSES

    images, labels, relative_heights = render_queries(
        args.test_fonts, classes, args.limit
    )
    print(
        f"Rendered {len(images)} queries "
        f"({len(classes)} classes x {len(args.test_fonts)} fonts)."
    )

    results: list[EngineResult] = []
    for name in selected:
        print(f"\n=== {name} ===")
        try:
            if name in ("gemma", "siglip", "jina"):
                if name == "gemma":
                    dim = args.gemma_dim
                elif name == "siglip":
                    dim = args.siglip_dim
                else:
                    dim = args.jina_dim
                predictor: Predictor = EmbeddingPredictor(
                    embedder_name=name,
                    index_fonts=args.index_fonts,
                    classes=classes,
                    dim=dim,
                    top_k=args.top_k,
                    device=args.device,
                )
            else:
                predictor = EnginePredictor(
                    name=name, engine=load_generic_engine(name, args)
                )
        except Exception as exc:
            result = EngineResult(
                name=name,
                status="unavailable",
                total=len(labels),
                error=f"{type(exc).__name__}: {exc}",
            )
            if args.verbose:
                traceback.print_exc()
            print(f"  unavailable: {result.error}")
            results.append(result)
            _release_memory()
            continue

        result = evaluate(predictor, images, labels, relative_heights, args.verbose)
        if result.status == "ok":
            data = result.as_dict()
            print(
                "  exact={exact_pct:.1f}%  case-insensitive={ci:.1f}%  "
                "extracted={ext:.1f}%  latency={lat:.1f} ms/item".format(
                    exact_pct=data["exact_pct"],
                    ci=data["case_insensitive_pct"],
                    ext=data["extracted_pct"],
                    lat=result.latency_ms,
                )
            )
        else:
            print(f"  unavailable: {result.error}")
        results.append(result)
        del predictor
        _release_memory()

    payload = {
        "n_queries": len(labels),
        "classes": len(classes),
        "test_fonts": args.test_fonts,
        "index_fonts": args.index_fonts,
        "results": [result.as_dict() for result in results],
    }
    json_path = args.output.with_suffix(".json")
    md_path = args.output.with_suffix(".md")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2))
    write_markdown(results, md_path, len(labels))

    print(f"\nSaved {json_path}")
    print(f"Saved {md_path}")


if __name__ == "__main__":
    main()
