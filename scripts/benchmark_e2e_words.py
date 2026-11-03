"""End-to-end word and text line benchmark for EmbeddingOCR vs conventional OCRs.

Renders realistic synthetic text lines / words across held-out clean fonts, runs
the full end-to-end recognition pipeline (segmentation -> batched embedding ->
prototype matching -> space reconstruction), and computes standard CER and WER.

Usage:
    uv run python scripts/benchmark_e2e_words.py --help
    uv run python scripts/benchmark_e2e_words.py --num-samples 20
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import jiwer
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from embedding_ocr.data.glyph_renderer import ALPHANUMERIC_CLASSES
from embedding_ocr.recognition.embedding.embedder import EmbeddingGemma2
from embedding_ocr.recognition.embedding.engine import EmbeddingRecognizer
from embedding_ocr.recognition.embedding.index import FaissIndex
from embedding_ocr.recognition.embedding.matcher import PrototypeMatcher
from embedding_ocr.recognition.embedding.prototypes import PrototypeDB
from embedding_ocr.recognition.embedding.segmenter import CharacterSegmenter

SAMPLE_WORDS = [
    "Embedding",
    "Gemma",
    "DeepMind",
    "Google",
    "Artificial",
    "Intelligence",
    "Optical",
    "Character",
    "Recognition",
    "FAISS",
    "Vector",
    "Search",
    "Prototype",
    "Database",
    "Invoice",
    "Document",
    "Machine",
    "Learning",
    "Vision",
    "Language",
]

SAMPLE_PHRASES = [
    "Hello World",
    "Deep Learning",
    "Open Source",
    "Computer Vision",
    "Invoice 2026",
    "Text Recognition",
    "Neural Network",
    "Latin Script",
]


def render_text_line(
    text: str,
    font_path: str,
    font_size: int = 36,
    padding_x: int = 20,
    padding_y: int = 15,
) -> np.ndarray:
    """Render a clean text line on white canvas with black text."""
    font = ImageFont.truetype(font_path, font_size)
    bbox = font.getbbox(text)
    w = (bbox[2] - bbox[0]) + 2 * padding_x
    h = (bbox[3] - bbox[1]) + 2 * padding_y

    img = Image.new("RGB", (w, h), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text(
        (padding_x - bbox[0], padding_y - bbox[1]), text, font=font, fill=(0, 0, 0)
    )
    return np.array(img, dtype=np.uint8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="End-to-end word and line OCR benchmark."
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=15,
        help="Number of text phrases to evaluate.",
    )
    parser.add_argument(
        "--font-catalog",
        type=Path,
        default=Path("data/standard_clean_fonts.txt"),
    )
    parser.add_argument(
        "--gemma-dim", type=int, default=768, choices=[768, 512, 256, 128]
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cache-prototypes", action="store_true", default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("=" * 80)
    print("END-TO-END WORD & LINE RECOGNITION BENCHMARK (EmbeddingGemma 2)")
    print("=" * 80)

    # 1. Load clean fonts
    if not args.font_catalog.exists():
        msg = f"Font catalog {args.font_catalog} does not exist."
        raise FileNotFoundError(msg)

    all_fonts = [
        line.strip()
        for line in args.font_catalog.read_text().splitlines()
        if line.strip() and not line.startswith("#") and Path(line.strip()).exists()
    ]
    index_fonts = all_fonts[:5]
    test_fonts = all_fonts[5:15]
    print(
        f"Indexed Prototype Fonts ({len(index_fonts)}): {[Path(f).name for f in index_fonts]}"
    )
    print(
        f"Held-Out Test Fonts    ({len(test_fonts)}): {[Path(f).name for f in test_fonts[:5]]}..."
    )

    # 2. Build or load prototype DB
    embedder = EmbeddingGemma2(dim=args.gemma_dim, device=args.device)
    index = FaissIndex(dim=args.gemma_dim)
    proto_db = PrototypeDB(embedder=embedder, index=index, classes=ALPHANUMERIC_CLASSES)

    cache_path = Path(f"artifacts/prototypes/clean_{args.gemma_dim}d")
    if (
        args.cache_prototypes
        and cache_path.with_suffix(".faiss").exists()
        and cache_path.with_suffix(".labels.json").exists()
    ):
        print(f"Loading cached prototypes from {cache_path}...")
        proto_db.index = FaissIndex.load(cache_path)
        proto_db.load_cache(cache_path)
    else:
        print(
            "Rendering and embedding prototype gallery (5 fonts x 62 alphanumeric classes)..."
        )
        proto_db.build_from_fonts(index_fonts)
        if args.cache_prototypes:
            proto_db.save(cache_path)
            print(f"Cached prototypes saved to {cache_path}")

    # 3. Setup EmbeddingRecognizer
    segmenter = CharacterSegmenter()
    matcher = PrototypeMatcher()
    recognizer = EmbeddingRecognizer(
        prototype_db=proto_db,
        segmenter=segmenter,
        matcher=matcher,
        top_k=25,
    )

    # 4. Generate test queries (words and phrases)
    targets: list[str] = []
    test_images: list[np.ndarray] = []

    test_pool = SAMPLE_PHRASES + SAMPLE_WORDS
    for i in range(args.num_samples):
        text = test_pool[i % len(test_pool)]
        font = test_fonts[i % len(test_fonts)]
        img = render_text_line(text, font)
        targets.append(text)
        test_images.append(img)

    print(
        f"\nEvaluating {len(test_images)} synthetic word & line images across held-out fonts..."
    )

    # 5. Run Recognition
    predictions: list[str] = []
    latencies: list[float] = []

    for idx, (img, target) in enumerate(zip(test_images, targets, strict=True)):
        t0 = time.perf_counter()
        res = recognizer.recognize(img)
        dt = (time.perf_counter() - t0) * 1000.0
        latencies.append(dt)
        predictions.append(res.text)

        print(
            f"[{idx + 1:02d}/{len(test_images):02d}] "
            f"Target: '{target:<20}' -> Pred: '{res.text:<20}' "
            f"(Conf: {res.confidence * 100:.1f}%, {dt:.1f}ms)"
        )

    # 6. Compute standard OCR metrics
    cer = jiwer.cer(targets, predictions)
    wer = jiwer.wer(targets, predictions)
    exact_match = sum(p == t for p, t in zip(predictions, targets, strict=True)) / len(
        targets
    )
    case_insens_match = sum(
        p.lower() == t.lower() for p, t in zip(predictions, targets, strict=True)
    ) / len(targets)
    avg_latency = float(np.mean(latencies))

    print("\n" + "=" * 80)
    print("RESULTS: END-TO-END WORD & LINE RECOGNITION (EmbeddingGemma 2)")
    print("=" * 80)
    print(f"Exact Line Accuracy:        {exact_match * 100:.1f}%")
    print(f"Case-Insensitive Accuracy:  {case_insens_match * 100:.1f}%")
    print(f"Character Error Rate (CER): {cer * 100:.2f}%")
    print(f"Word Error Rate (WER):      {wer * 100:.2f}%")
    print(f"Average Latency / Image:    {avg_latency:.1f} ms")
    print("=" * 80)


if __name__ == "__main__":
    main()
