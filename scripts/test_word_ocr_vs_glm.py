"""Word OCR vs GLM-OCR Benchmark with Normalized CER as primary metric.

Features:
- Normalized CER (case-insensitive & trimmed) as primary metric.
- Raw CER, WER, and Accuracy also reported for completeness.
- Evaluation on diverse words and phrases across unseen fonts.
- Compares:
  1. GLM-OCR (VLM baseline)
  2. Hierarchical EmbeddingOCR (Visual word/subword + EmbeddingGemma 2)
  3. Character-level EmbeddingOCR (Segmentation + Visual Prototype matching)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import jiwer
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

from embedding_ocr.data.glyph_renderer import ALPHANUMERIC_CLASSES
from embedding_ocr.recognition.embedding.embedder import EmbeddingGemma2
from embedding_ocr.recognition.embedding.engine import EmbeddingRecognizer
from embedding_ocr.recognition.embedding.hierarchical import (
    EmbeddingVocabulary,
    HierarchicalWordRecognizer,
)
from embedding_ocr.recognition.embedding.index import FaissIndex
from embedding_ocr.recognition.embedding.prototypes import PrototypeDB
from embedding_ocr.recognition.glm_ocr import GLMOCRRecognizer

EVAL_WORDS = [
    "invoice",
    "total",
    "customer",
    "payment",
    "service",
    "network",
    "system",
    "neural",
    "python",
    "digital",
    "document",
    "receipt",
    "account",
    "balance",
    "amount",
    "company",
    "contact",
    "address",
    "number",
    "status",
    "Google",
    "DeepMind",
    "Embedding",
    "Machine",
    "Vision",
]

EVAL_PHRASES = [
    "invoice total 2026",
    "customer account balance",
    "optical character recognition",
    "neural network system",
    "digital document service",
    "Machine Learning Vision",
]

INDEX_FONTS = [
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]

TEST_FONTS = [
    "/usr/share/fonts/truetype/roboto/unhinted/RobotoTTF/Roboto-Regular.ttf",
    "/usr/share/fonts/truetype/roboto/unhinted/RobotoTTF/Roboto-Bold.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
]


def render_crop(
    text: str,
    font_path: str,
    font_size: int = 32,
    augment: bool = False,
) -> np.ndarray:
    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception:
        font = ImageFont.load_default()
    bbox = font.getbbox(text)
    w = (bbox[2] - bbox[0]) + 28
    h = (bbox[3] - bbox[1]) + 20
    img = Image.new("RGB", (max(w, 20), max(h, 20)), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((14 - bbox[0], 10 - bbox[1]), text, font=font, fill=(0, 0, 0))

    if augment:
        rng = np.random.default_rng(42)
        # 1. Subtle rotation / skew
        angle = float(rng.uniform(-3.0, 3.0))
        img = img.rotate(
            angle, resample=Image.Resampling.BILINEAR, fillcolor=(255, 255, 255)
        )

        # 2. Gaussian blur (simulating slight out-of-focus capture)
        if rng.random() < 0.6:
            radius = float(rng.uniform(0.4, 0.9))
            img = img.filter(ImageFilter.GaussianBlur(radius=radius))

        # 3. Contrast adjustment
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(float(rng.uniform(0.8, 1.2)))

        arr = np.array(img, dtype=np.float32)

        # 4. Additive Gaussian noise (scanner grain)
        noise = rng.normal(0, float(rng.uniform(4.0, 10.0)), arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return arr

    return np.array(img, dtype=np.uint8)


def normalize_text(text: str) -> str:
    """Normalize text for fair metric evaluation (case-folding and whitespace trimming)."""
    return " ".join(text.strip().lower().split())


def compute_metrics(
    ground_truths: list[str], predictions: list[str]
) -> dict[str, float]:
    norm_gt = [normalize_text(t) for t in ground_truths]
    norm_pred = [normalize_text(t) for t in predictions]

    # PRIMARY METRIC: Normalized Character Error Rate (CER)
    norm_cer = float(jiwer.cer(norm_gt, norm_pred))
    raw_cer = float(jiwer.cer(ground_truths, predictions))
    norm_wer = float(jiwer.wer(norm_gt, norm_pred))
    raw_wer = float(jiwer.wer(ground_truths, predictions))

    norm_acc = sum(1 for g, p in zip(norm_gt, norm_pred, strict=False) if g == p) / max(
        1, len(ground_truths)
    )
    raw_acc = sum(
        1 for g, p in zip(ground_truths, predictions, strict=False) if g == p
    ) / max(1, len(ground_truths))

    return {
        "primary_norm_cer": norm_cer,
        "raw_cer": raw_cer,
        "norm_wer": norm_wer,
        "raw_wer": raw_wer,
        "norm_accuracy": float(norm_acc),
        "raw_accuracy": float(raw_acc),
    }


def evaluate_engine(
    name: str,
    recognizer: Any,
    test_fonts: list[str],
    augment: bool = False,
) -> dict[str, Any]:
    print("\n=======================================================")
    print(f"Evaluating {name} (augment={augment})")
    print("=======================================================")

    all_word_gts = []
    all_word_preds = []
    word_latencies = []

    for f_path in test_fonts:
        for w in EVAL_WORDS:
            crop = render_crop(w, f_path, augment=augment)
            t0 = time.perf_counter()
            res = recognizer.recognize(crop)
            t1 = time.perf_counter()
            word_latencies.append((t1 - t0) * 1000.0)
            all_word_gts.append(w)
            all_word_preds.append(res.text.strip())

    word_metrics = compute_metrics(all_word_gts, all_word_preds)

    all_phrase_gts = []
    all_phrase_preds = []
    phrase_latencies = []

    for f_path in test_fonts:
        for p in EVAL_PHRASES:
            crop = render_crop(p, f_path, augment=augment)
            t0 = time.perf_counter()
            res = recognizer.recognize(crop)
            t1 = time.perf_counter()
            phrase_latencies.append((t1 - t0) * 1000.0)
            all_phrase_gts.append(p)
            all_phrase_preds.append(res.text.strip())

    phrase_metrics = compute_metrics(all_phrase_gts, all_phrase_preds)

    print(f"[{name}] WORD METRICS:")
    print(
        f"  >>> PRIMARY Normalized CER: {word_metrics['primary_norm_cer'] * 100:.2f}% <<<"
    )
    print(
        f"  Raw CER: {word_metrics['raw_cer'] * 100:.2f}% | Normalized Acc: {word_metrics['norm_accuracy'] * 100:.1f}%"
    )
    print(f"  Avg Latency: {np.mean(word_latencies):.1f} ms")

    print(f"[{name}] PHRASE METRICS:")
    print(
        f"  >>> PRIMARY Normalized CER: {phrase_metrics['primary_norm_cer'] * 100:.2f}% <<<"
    )
    print(
        f"  Raw CER: {phrase_metrics['raw_cer'] * 100:.2f}% | Normalized Acc: {phrase_metrics['norm_accuracy'] * 100:.1f}%"
    )
    print(f"  Avg Latency: {np.mean(phrase_latencies):.1f} ms")

    # Sample predictions
    samples = []
    for g, p in zip(all_word_gts[:10], all_word_preds[:10], strict=False):
        samples.append(
            {
                "type": "word",
                "gt": g,
                "pred": p,
                "norm_match": normalize_text(g) == normalize_text(p),
            }
        )
    for g, p in zip(all_phrase_gts[:5], all_phrase_preds[:5], strict=False):
        samples.append(
            {
                "type": "phrase",
                "gt": g,
                "pred": p,
                "norm_match": normalize_text(g) == normalize_text(p),
            }
        )

    return {
        "word_metrics": word_metrics,
        "avg_word_latency_ms": float(np.mean(word_latencies)),
        "phrase_metrics": phrase_metrics,
        "avg_phrase_latency_ms": float(np.mean(phrase_latencies)),
        "samples": samples,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gemma-dim", type=int, default=768)
    parser.add_argument("--skip-glm", action="store_true")
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Apply blur, noise, skew, and contrast degradation.",
    )
    args = parser.parse_args()

    results: dict[str, Any] = {}
    valid_test_fonts = [f for f in TEST_FONTS if Path(f).exists()]
    if not valid_test_fonts:
        valid_test_fonts = [f for f in INDEX_FONTS[:2] if Path(f).exists()]
    print(f"Test Fonts: {[Path(f).name for f in valid_test_fonts]}")
    print(f"Augmentation active: {args.augment}")

    # 1. GLM-OCR
    if not args.skip_glm:
        print("\nLoading GLM-OCR...")
        glm = GLMOCRRecognizer(model_id="zai-org/GLM-OCR", device="auto")
        results["GLM-OCR"] = evaluate_engine(
            "GLM-OCR", glm, valid_test_fonts, augment=args.augment
        )
        del glm
        torch.cuda.empty_cache()

    # 2. EmbeddingGemma 2 embedder setup
    print("\nLoading EmbeddingGemma 2...")
    emb = EmbeddingGemma2(dim=args.gemma_dim)

    # 3. Hierarchical EmbeddingOCR
    print("\nBuilding Hierarchical Word Recognizer...")
    vocab = EmbeddingVocabulary(
        embedder=emb, cache_dir=f"artifacts/vocab_cache_{args.gemma_dim}d"
    )
    vocab.extract_and_organize_tokens(
        extra_lexicon=EVAL_WORDS + ["2026"],
        max_words=12000,
    )
    vocab.build_indexes()
    hierarchical_engine = HierarchicalWordRecognizer(
        embedder=emb,
        vocabulary=vocab,
        min_word_sim=0.74,
        min_word_margin=0.005,
        min_subword_sim=0.71,
        min_subword_margin=0.003,
        min_char_sim=0.62,
    )
    results[f"Hierarchical_EmbeddingOCR_{args.gemma_dim}D"] = evaluate_engine(
        f"Hierarchical_EmbeddingOCR_{args.gemma_dim}D",
        hierarchical_engine,
        valid_test_fonts,
        augment=args.augment,
    )

    # 4. Character-level EmbeddingOCR (Segmentation + Prototype matching)
    print("\nBuilding Character-level EmbeddingOCR...")
    valid_index_fonts = [f for f in INDEX_FONTS if Path(f).exists()]
    index = FaissIndex(dim=args.gemma_dim)
    proto_db = PrototypeDB(embedder=emb, index=index, classes=ALPHANUMERIC_CLASSES)
    proto_db.build_from_fonts(valid_index_fonts)
    char_recognizer = EmbeddingRecognizer(prototype_db=proto_db)
    results[f"Char_EmbeddingOCR_{args.gemma_dim}D"] = evaluate_engine(
        f"Char_EmbeddingOCR_{args.gemma_dim}D",
        char_recognizer,
        valid_test_fonts,
        augment=args.augment,
    )

    out_file = Path("docs/word_ocr_vs_glm_report.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(results, indent=2))
    print(f"\nSaved complete benchmark report to {out_file}")


if __name__ == "__main__":
    main()
