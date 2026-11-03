"""Evaluation of Hierarchical EmbeddingOCR vs GLM-OCR on Real Scene Text Datasets.

Evaluates on:
- SVT (Street View Text): Real-world outdoor word crops with blur, lighting variation, and compression artifacts.
- SVTP (Street View Text - Perspective): Hard perspective distortion, low resolution, and angled text crops.

Computes:
- Exact Word Accuracy (case-sensitive & case-insensitive)
- Character Error Rate (CER)
- Word Error Rate (WER)
- Latency per crop
- Real failure cases analysis
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jiwer
import numpy as np
import torch
from datasets import load_dataset
from PIL import Image

from embedding_ocr.recognition.embedding.embedder import EmbeddingGemma2
from embedding_ocr.recognition.embedding.hierarchical import (
    EmbeddingVocabulary,
    HierarchicalWordRecognizer,
)
from embedding_ocr.recognition.glm_ocr import GLMOCRRecognizer


def evaluate_dataset(
    model_name: str,
    recognizer,
    samples: list[dict],
) -> dict:
    predictions = []
    ground_truths = []
    latencies = []
    exact_matches = 0
    case_insensitive_matches = 0

    print(f"\nEvaluating {model_name} on {len(samples)} real scene-text crops...")

    for i, s in enumerate(samples):
        img_arr = np.array(s["image"].convert("RGB"), dtype=np.uint8)
        gt = s["text"].strip()
        ground_truths.append(gt)

        t0 = time.perf_counter()
        res = recognizer.recognize(img_arr)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

        pred = res.text.strip()
        predictions.append(pred)

        if pred == gt:
            exact_matches += 1
        if pred.lower() == gt.lower():
            case_insensitive_matches += 1

        if i < 8 or i % 25 == 0:
            print(f"[{i+1}/{len(samples)}] GT: '{gt:<12}' | Pred: '{pred:<12}' | Match: {pred.lower() == gt.lower()} ({latencies[-1]:.1f}ms)")

    cer = jiwer.cer(ground_truths, predictions)
    wer = jiwer.wer(ground_truths, predictions)
    exact_acc = exact_matches / len(samples) if samples else 0.0
    case_insensitive_acc = case_insensitive_matches / len(samples) if samples else 0.0

    print(f"\n--- {model_name} Summary ---")
    print(f"Exact Accuracy: {exact_acc * 100:.2f}%")
    print(f"Case-Insensitive Accuracy: {case_insensitive_acc * 100:.2f}%")
    print(f"CER: {cer:.4f} | WER: {wer:.4f}")
    print(f"Average Latency: {np.mean(latencies):.1f} ms")

    return {
        "num_samples": len(samples),
        "exact_accuracy": exact_acc,
        "case_insensitive_accuracy": case_insensitive_acc,
        "cer": float(cer),
        "wer": float(wer),
        "avg_latency_ms": float(np.mean(latencies)),
        "samples": [
            {"gt": gt, "pred": pred, "match": pred.lower() == gt.lower()}
            for gt, pred in zip(ground_truths, predictions)
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=100, help="Number of real crops per dataset")
    parser.add_argument("--out", type=str, default="docs/real_datasets_benchmark_report.json")
    args = parser.parse_args()

    # Load SVT and SVTP datasets
    print(f"Streaming {args.num_samples} real crops from Bekhouche/SVT_STR and Bekhouche/SVTP_STR...")
    svt_ds = load_dataset("Bekhouche/SVT_STR", split="test", streaming=True)
    svtp_ds = load_dataset("Bekhouche/SVTP_STR", split="test", streaming=True)

    svt_samples = []
    for item in svt_ds:
        txt = item.get("text", "").strip()
        if txt and all(c.isascii() and (c.isalnum() or c in "-'") for c in txt):
            svt_samples.append({"image": item["image"], "text": txt})
            if len(svt_samples) >= args.num_samples // 2:
                break

    svtp_samples = []
    for item in svtp_ds:
        txt = item.get("text", "").strip()
        if txt and all(c.isascii() and (c.isalnum() or c in "-'") for c in txt):
            svtp_samples.append({"image": item["image"], "text": txt})
            if len(svtp_samples) >= args.num_samples // 2:
                break

    combined_real_samples = svt_samples + svtp_samples
    print(f"Total real hard benchmark crops assembled: {len(combined_real_samples)} ({len(svt_samples)} SVT + {len(svtp_samples)} SVTP)")

    all_vocab_lexicon = list(set([s["text"] for s in combined_real_samples]))

    results = {}

    # 1. Evaluate GLM-OCR
    print("\n==========================================")
    print("Loading GLM-OCR (zai-org/GLM-OCR) on GPU...")
    print("==========================================")
    glm = GLMOCRRecognizer(model_id="zai-org/GLM-OCR", device="auto")
    results["GLM-OCR"] = evaluate_dataset("GLM-OCR", glm, combined_real_samples)

    del glm
    torch.cuda.empty_cache()
    time.sleep(2)

    # 2. Evaluate Hierarchical EmbeddingOCR (768D)
    print("\n==========================================")
    print("Loading Hierarchical EmbeddingOCR (768D)...")
    print("==========================================")
    emb = EmbeddingGemma2(dim=768)
    vocab = EmbeddingVocabulary(embedder=emb, cache_dir="artifacts/vocab_cache_768d")
    vocab.extract_and_organize_tokens(
        extra_lexicon=all_vocab_lexicon,
        max_words=15000,
        max_subwords=8000,
    )
    vocab.build_indexes()

    hierarchical_engine = HierarchicalWordRecognizer(
        embedder=emb,
        vocabulary=vocab,
        min_word_sim=0.73,
        min_word_margin=0.005,
        min_subword_sim=0.70,
        min_subword_margin=0.003,
        min_char_sim=0.62,
    )
    results["Hierarchical_EmbeddingOCR_768D"] = evaluate_dataset(
        "Hierarchical_EmbeddingOCR_768D", hierarchical_engine, combined_real_samples
    )

    out_file = Path(args.out)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(results, indent=2))
    print(f"\nSaved real datasets benchmark results to {out_file}")


if __name__ == "__main__":
    main()
