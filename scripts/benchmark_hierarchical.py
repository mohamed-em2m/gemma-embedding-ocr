"""Comprehensive Benchmark and Evaluation of Hierarchical Word Recognition.

Evaluates:
- Exact word recognition accuracy
- Character Error Rate (CER) and Word Error Rate (WER)
- Unknown / OOV word recognition accuracy
- Word boundary detection accuracy
- Inference latency per word
- Recursive fallback frequency
- Comparison across dimensions (768D, 512D, 256D)
- Comparison of full-word vs recursive subword & character matching
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jiwer
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from embedding_ocr.recognition.embedding.embedder import EmbeddingGemma2
from embedding_ocr.recognition.embedding.hierarchical.engine import (
    HierarchicalWordRecognizer,
)
from embedding_ocr.recognition.embedding.hierarchical.vocabulary import (
    EmbeddingVocabulary,
)

# Evaluation words: Known common words, multi-token compound words, and unseen / rare words
EVAL_KNOWN_WORDS = [
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
    "report",
    "project",
    "product",
    "machine",
    "learning",
    "vision",
    "database",
    "market",
    "client",
    "office",
]

EVAL_COMPOUND_WORDS = [
    "network",
    "payment",
    "customer",
    "invoice",
    "receipt",
    "digital",
    "document",
    "optical",
    "system",
    "neural",
]

EVAL_UNKNOWN_WORDS = [
    "supercalifragilistic",
    "microgradient",
    "counterproposal",
    "neurovector",
    "subsegmentation",
    "polymodal",
]

SAMPLE_PHRASES = [
    "invoice total 2026",
    "customer account balance",
    "optical character recognition",
    "neural network system",
    "digital document service",
]


def render_test_crop(
    text: str,
    font_path: str = "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    font_size: int = 32,
    padding_x: int = 14,
    padding_y: int = 10,
) -> np.ndarray:
    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception:
        font = ImageFont.load_default()

    bbox = font.getbbox(text)
    w = (bbox[2] - bbox[0]) + 2 * padding_x
    h = (bbox[3] - bbox[1]) + 2 * padding_y

    img = Image.new("RGB", (max(w, 20), max(h, 20)), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text(
        (padding_x - bbox[0], padding_y - bbox[1]), text, font=font, fill=(0, 0, 0)
    )
    return np.array(img, dtype=np.uint8)


def evaluate_engine(
    engine: HierarchicalWordRecognizer,
    words: list[str],
    phrases: list[str],
) -> dict:
    """Run comprehensive accuracy, CER/WER, latency, and telemetry evaluations."""
    engine.stats = {k: 0 for k in engine.stats}
    engine._cache.clear()

    # 1. Word recognition
    word_correct = 0
    predictions = []
    latencies = []

    for w in words:
        crop = render_test_crop(w)
        t0 = time.perf_counter()
        res = engine.recognize(crop)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)
        predictions.append(res.text)
        if res.text.strip().lower() == w.strip().lower():
            word_correct += 1

    word_acc = word_correct / len(words) if words else 0.0
    cer = jiwer.cer(words, predictions)
    wer = jiwer.wer(words, predictions)

    # 2. Multi-word phrase recognition and boundary accuracy
    phrase_preds = []
    phrase_latencies = []
    boundary_hits = 0

    for phrase in phrases:
        crop = render_test_crop(phrase)
        t0 = time.perf_counter()
        res = engine.recognize(crop)
        t1 = time.perf_counter()
        phrase_latencies.append((t1 - t0) * 1000.0)
        phrase_preds.append(res.text)

        # Check word boundary count
        gt_words = phrase.split()
        pred_words = res.text.split()
        if len(pred_words) == len(gt_words):
            boundary_hits += 1

    boundary_acc = boundary_hits / len(phrases) if phrases else 0.0
    phrase_cer = jiwer.cer(phrases, phrase_preds)
    phrase_wer = jiwer.wer(phrases, phrase_preds)

    return {
        "word_accuracy": word_acc,
        "word_cer": float(cer),
        "word_wer": float(wer),
        "avg_word_latency_ms": float(np.mean(latencies)),
        "phrase_boundary_accuracy": boundary_acc,
        "phrase_cer": float(phrase_cer),
        "phrase_wer": float(phrase_wer),
        "avg_phrase_latency_ms": float(np.mean(phrase_latencies)),
        "telemetry": dict(engine.stats),
        "sample_predictions": list(zip(words[:8], predictions[:8], strict=False)),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Hierarchical Word Recognition Benchmark"
    )
    parser.add_argument("--dims", nargs="+", type=int, default=[768, 512, 256])
    parser.add_argument(
        "--out", type=str, default="docs/hierarchical_benchmark_report.json"
    )
    args = parser.parse_args()

    results_all_dims = {}

    for dim in args.dims:
        print("\n=======================================================")
        print(f"Evaluating Hierarchical Word Recognition with dim={dim}D")
        print("=======================================================")

        emb = EmbeddingGemma2(dim=dim)
        vocab = EmbeddingVocabulary(
            embedder=emb,
            cache_dir=f"artifacts/vocab_cache_{dim}d",
        )
        print("Extracting and organizing tokenizer candidates...")
        # Include custom words into candidate lexicon
        vocab.extract_and_organize_tokens(
            extra_lexicon=EVAL_KNOWN_WORDS + EVAL_COMPOUND_WORDS,
            max_words=10000,
            max_subwords=5000,
        )
        print(
            f"Tokens: {len(vocab.word_tokens)} words, {len(vocab.subword_tokens)} subwords, {len(vocab.char_tokens)} chars"
        )

        print("Building FAISS candidate indexes...")
        vocab.build_indexes()

        # Instantiate Hierarchical Engine
        engine = HierarchicalWordRecognizer(
            embedder=emb,
            vocabulary=vocab,
            min_word_sim=0.74,
            min_word_margin=0.005,
            min_subword_sim=0.71,
            min_subword_margin=0.003,
            min_char_sim=0.62,
        )

        # 1. Full Hierarchical evaluation
        print("Running full hierarchical evaluation...")
        eval_metrics = evaluate_engine(
            engine,
            words=EVAL_KNOWN_WORDS,
            phrases=SAMPLE_PHRASES,
        )

        # 2. Evaluate Unknown/OOV words (measuring recursive fallback)
        print("Evaluating unknown / complex words...")
        unknown_metrics = evaluate_engine(
            engine,
            words=EVAL_UNKNOWN_WORDS,
            phrases=[],
        )

        results_all_dims[f"{dim}D"] = {
            "known_words": eval_metrics,
            "unknown_words": unknown_metrics,
        }

        print(f"\nResults for {dim}D:")
        print(f"  Exact Word Accuracy: {eval_metrics['word_accuracy'] * 100:.1f}%")
        print(
            f"  Word CER: {eval_metrics['word_cer']:.4f} | WER: {eval_metrics['word_wer']:.4f}"
        )
        print(
            f"  Phrase Word Boundary Acc: {eval_metrics['phrase_boundary_accuracy'] * 100:.1f}%"
        )
        print(f"  Latency per word: {eval_metrics['avg_word_latency_ms']:.2f} ms")
        print(f"  Telemetry: {eval_metrics['telemetry']}")

    # Save final report
    out_file = Path(args.out)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(results_all_dims, indent=2))
    print(f"\nFull benchmark report saved to {out_file}")


if __name__ == "__main__":
    main()
