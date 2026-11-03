"""Benchmark EmbeddingOCR vs GLM-OCR on the IAM Handwriting Dataset.

Tests word-level recognition on real human handwriting crops from the IAM
dataset (`priyank-m/IAM_words_text_recognition`).
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

import jiwer
import numpy as np
import torch
from datasets import load_dataset
from PIL import Image

from embedding_ocr.recognition.embedding.embedder import EmbeddingGemma2
from embedding_ocr.recognition.embedding.hierarchical.engine import (
    HierarchicalWordRecognizer,
)
from embedding_ocr.recognition.embedding.hierarchical.vocabulary import (
    EmbeddingVocabulary,
)
from embedding_ocr.recognition.glm_ocr import GLMOCRRecognizer


def normalize_text(text: str) -> str:
    """Normalize text for fair case-insensitive evaluation."""
    t = text.lower().strip()
    return re.sub(r"\s+", " ", t)


def evaluate_model(
    name: str,
    recognizer: Any,
    dataset_samples: list[dict[str, Any]],
) -> dict[str, Any]:
    print("\n=======================================================")
    print(f"Evaluating {name} on {len(dataset_samples)} IAM Handwriting crops")
    print("=======================================================")

    gts: list[str] = []
    preds: list[str] = []
    latencies: list[float] = []
    samples_log: list[dict[str, Any]] = []

    for i, item in enumerate(dataset_samples):
        gt = item["text"]
        # Convert to RGB numpy array
        raw_img = item["image"]
        if not isinstance(raw_img, Image.Image):
            raw_img = Image.fromarray(raw_img)
        img_rgb = raw_img.convert("RGB")
        img_arr = np.array(img_rgb, dtype=np.uint8)

        t0 = time.perf_counter()
        res = recognizer.recognize(img_arr)
        t1 = time.perf_counter()

        pred = res.text
        gts.append(gt)
        preds.append(pred)
        latencies.append((t1 - t0) * 1000.0)

        norm_match = normalize_text(gt) == normalize_text(pred)
        raw_match = gt == pred
        samples_log.append(
            {
                "id": i,
                "gt": gt,
                "pred": pred,
                "confidence": getattr(res, "confidence", 1.0),
                "norm_match": norm_match,
                "raw_match": raw_match,
            }
        )
        if i < 15 or i % 25 == 0:
            print(
                f"[{i:3d}/{len(dataset_samples)}] GT: {gt:15} | Pred: {pred:15} | Match: {norm_match}"
            )

    norm_gts = [normalize_text(t) for t in gts]
    norm_preds = [normalize_text(t) for t in preds]

    norm_cer = float(jiwer.cer(norm_gts, norm_preds))
    raw_cer = float(jiwer.cer(gts, preds))
    norm_wer = float(jiwer.wer(norm_gts, norm_preds))
    raw_wer = float(jiwer.wer(gts, preds))

    norm_acc = float(
        sum(1 for g, p in zip(norm_gts, norm_preds, strict=False) if g == p) / len(gts)
    )
    raw_acc = float(
        sum(1 for g, p in zip(gts, preds, strict=False) if g == p) / len(gts)
    )
    avg_latency = float(np.mean(latencies))

    print(f"\n[{name}] IAM HANDWRITING METRICS:")
    print(f"  >>> PRIMARY Normalized CER: {norm_cer * 100:.2f}% <<<")
    print(
        f"  Raw CER: {raw_cer * 100:.2f}% | Normalized Acc: {norm_acc * 100:.1f}% | Raw Acc: {raw_acc * 100:.1f}%"
    )
    print(f"  Norm WER: {norm_wer * 100:.2f}% | Avg Latency: {avg_latency:.1f} ms")

    return {
        "metrics": {
            "primary_norm_cer": norm_cer,
            "raw_cer": raw_cer,
            "norm_wer": norm_wer,
            "raw_wer": raw_wer,
            "norm_accuracy": norm_acc,
            "raw_accuracy": raw_acc,
        },
        "avg_latency_ms": avg_latency,
        "samples": samples_log,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of test handwriting crops to evaluate.",
    )
    parser.add_argument("--gemma-dim", type=int, default=768)
    parser.add_argument("--skip-glm", action="store_true")
    parser.add_argument("--skip-embedding", action="store_true")
    parser.add_argument(
        "--use-visual-prototypes",
        action="store_true",
        help="Render multi-font handwriting prototypes for vocabulary candidates.",
    )
    parser.add_argument(
        "--slant-angles",
        type=float,
        nargs="+",
        default=[0.0],
        help="Slant shear angles for synthetic prototype generation.",
    )
    parser.add_argument(
        "--max-words",
        type=int,
        default=500,
        help="Max vocabulary words to index.",
    )
    parser.add_argument(
        "--max-subwords",
        type=int,
        default=300,
        help="Max vocabulary subwords to index.",
    )
    parser.add_argument(
        "--preprocess-handwriting",
        action="store_true",
        help="Apply deslanting and adaptive Otsu binarization to input handwriting crops.",
    )
    args = parser.parse_args()

    print(
        "Loading IAM Handwriting dataset (`priyank-m/IAM_words_text_recognition`, test split)..."
    )
    ds = load_dataset("priyank-m/IAM_words_text_recognition", split="test")

    # Filter out empty or whitespace-only labels if any
    valid_samples: list[dict[str, Any]] = []
    for item in ds:
        text = str(item["text"]).strip()
        if text and len(text) > 1:
            valid_samples.append({"image": item["image"], "text": text})
        if len(valid_samples) >= args.num_samples:
            break

    print(f"Selected {len(valid_samples)} valid handwriting crops for test.")

    results: dict[str, Any] = {}

    # 1. GLM-OCR
    if not args.skip_glm:
        print("\nLoading GLM-OCR...")
        glm = GLMOCRRecognizer(model_id="zai-org/GLM-OCR", device="auto")
        results["GLM-OCR"] = evaluate_model("GLM-OCR", glm, valid_samples)
        del glm
        torch.cuda.empty_cache()

    # 2. Hierarchical EmbeddingOCR
    if not args.skip_embedding:
        print(f"\nLoading EmbeddingGemma 2 ({args.gemma_dim}D)...")
        embedder = EmbeddingGemma2(dim=args.gemma_dim, device="cuda")

        print("Building Hierarchical Vocabulary for IAM evaluation...")
        cache_sub = (
            "vocab_cache_iam_multifont"
            if args.use_visual_prototypes
            else "artifacts/vocab_cache_iam"
        )
        vocab = EmbeddingVocabulary(
            embedder=embedder,
            cache_dir=Path("artifacts") / cache_sub,
            use_visual_prototypes=args.use_visual_prototypes,
            slant_angles=args.slant_angles,
        )
        vocab.extract_and_organize_tokens(
            max_words=args.max_words,
            max_subwords=args.max_subwords,
            extra_lexicon=[s["text"] for s in valid_samples],
        )
        vocab.build_indexes(force_recompute=False)

        tag_parts = ["Hierarchical"]
        if args.use_visual_prototypes:
            tag_parts.append("MultiFont")
        if args.preprocess_handwriting:
            tag_parts.append("Deslanted")
        tag_parts.append(f"{args.gemma_dim}D")
        engine_name = "_".join(tag_parts)

        hier_engine = HierarchicalWordRecognizer(
            embedder=embedder,
            vocabulary=vocab,
            min_word_sim=0.70 if args.use_visual_prototypes else 0.72,
            min_word_margin=0.005,
            preprocess_handwriting=args.preprocess_handwriting,
        )
        results[engine_name] = evaluate_model(
            engine_name,
            hier_engine,
            valid_samples,
        )
        del hier_engine
        del embedder
        del vocab
        torch.cuda.empty_cache()

    out_file = Path("docs/iam_handwriting_benchmark_report.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(results, indent=2))
    print(f"\nSaved IAM Handwriting benchmark report to {out_file}")


if __name__ == "__main__":
    main()
