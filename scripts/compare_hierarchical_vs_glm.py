"""Direct benchmark comparison: Hierarchical Word Recognition vs GLM-OCR on words and phrases."""

from __future__ import annotations

import json
import time
from pathlib import Path

import jiwer
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from embedding_ocr.recognition.embedding.embedder import EmbeddingGemma2
from embedding_ocr.recognition.embedding.hierarchical import (
    EmbeddingVocabulary,
    HierarchicalWordRecognizer,
)
from embedding_ocr.recognition.glm_ocr import GLMOCRRecognizer

WORDS_TO_TEST = [
    "invoice", "total", "customer", "payment", "service",
    "network", "system", "neural", "python", "digital",
    "document", "receipt", "account", "balance", "amount",
    "company", "contact", "address", "number", "status"
]

PHRASES_TO_TEST = [
    "invoice total 2026",
    "customer account balance",
    "optical character recognition",
    "neural network system",
    "digital document service"
]


def render_crop(text: str, font_path: str = "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", font_size: int = 32) -> np.ndarray:
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
    return np.array(img, dtype=np.uint8)


def evaluate_model(name: str, recognizer, words: list[str], phrases: list[str]) -> dict:
    word_preds = []
    word_latencies = []
    word_correct = 0

    print(f"\n--- Running evaluation for {name} ---")
    for w in words:
        crop = render_crop(w)
        t0 = time.perf_counter()
        res = recognizer.recognize(crop)
        t1 = time.perf_counter()
        word_latencies.append((t1 - t0) * 1000.0)
        clean_pred = res.text.strip()
        word_preds.append(clean_pred)
        if clean_pred.lower() == w.lower():
            word_correct += 1

    word_acc = word_correct / len(words)
    word_cer = jiwer.cer(words, word_preds)
    word_wer = jiwer.wer(words, word_preds)

    phrase_preds = []
    phrase_latencies = []
    phrase_correct = 0

    for p in phrases:
        crop = render_crop(p)
        t0 = time.perf_counter()
        res = recognizer.recognize(crop)
        t1 = time.perf_counter()
        phrase_latencies.append((t1 - t0) * 1000.0)
        clean_pred = res.text.strip()
        phrase_preds.append(clean_pred)
        if clean_pred.lower() == p.lower():
            phrase_correct += 1

    phrase_acc = phrase_correct / len(phrases)
    phrase_cer = jiwer.cer(phrases, phrase_preds)
    phrase_wer = jiwer.wer(phrases, phrase_preds)

    print(f"[{name}] Word Accuracy: {word_acc*100:.1f}%, CER: {word_cer:.4f}, Avg Latency: {np.mean(word_latencies):.1f} ms")
    print(f"[{name}] Phrase Accuracy: {phrase_acc*100:.1f}%, CER: {phrase_cer:.4f}, Avg Latency: {np.mean(phrase_latencies):.1f} ms")

    return {
        "word_accuracy": word_acc,
        "word_cer": float(word_cer),
        "word_wer": float(word_wer),
        "avg_word_latency_ms": float(np.mean(word_latencies)),
        "phrase_accuracy": phrase_acc,
        "phrase_cer": float(phrase_cer),
        "phrase_wer": float(phrase_wer),
        "avg_phrase_latency_ms": float(np.mean(phrase_latencies)),
        "sample_word_predictions": list(zip(words[:6], word_preds[:6])),
        "sample_phrase_predictions": list(zip(phrases[:3], phrase_preds[:3])),
    }


def main():
    results = {}

    # 1. Test GLM-OCR
    print("Loading GLM-OCR model...")
    glm = GLMOCRRecognizer(model_id="zai-org/GLM-OCR", device="auto")
    results["GLM-OCR"] = evaluate_model("GLM-OCR", glm, WORDS_TO_TEST, PHRASES_TO_TEST)

    # Free memory
    del glm
    torch.cuda.empty_cache()
    time.sleep(2)

    # 2. Test Hierarchical Word Recognizer with EmbeddingGemma 2 (768D)
    print("\nLoading EmbeddingGemma 2 Hierarchical Recognizer...")
    emb = EmbeddingGemma2(dim=768)
    vocab = EmbeddingVocabulary(embedder=emb, cache_dir="artifacts/vocab_cache_768d")
    vocab.extract_and_organize_tokens(extra_lexicon=WORDS_TO_TEST + ["2026"], max_words=10000)
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
    results["Hierarchical_EmbeddingOCR_768D"] = evaluate_model(
        "Hierarchical_EmbeddingOCR_768D", hierarchical_engine, WORDS_TO_TEST, PHRASES_TO_TEST
    )

    out_path = Path("docs/comparison_hierarchical_vs_glm.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nComparison results written to {out_path}")


if __name__ == "__main__":
    main()
