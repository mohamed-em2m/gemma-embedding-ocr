"""Comprehensive cross-font letter recognition comparison:
EmbeddingGemma 2 vs Tesseract vs EasyOCR vs PaddleOCR vs GLM-OCR.

Evaluates Latin letter discovery across diverse unseen fonts (Sans, Serif, Mono).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from embedding_ocr.data.glyph_renderer import LATIN_CLASSES, render_glyph
from embedding_ocr.recognition.easyocr import EasyOCRRecognizer
from embedding_ocr.recognition.paddle import PaddleRecognizer
from embedding_ocr.recognition.tesseract import TesseractRecognizer
from embedding_ocr.recognition.embedding.embedder import EmbeddingGemma2
from embedding_ocr.recognition.embedding.index import FaissIndex
from embedding_ocr.recognition.embedding.prototypes import PrototypeDB
from embedding_ocr.recognition.embedding.matcher import PrototypeMatcher


def main() -> None:
    print("=" * 80)
    print("CROSS-FONT LATIN LETTER RECOGNITION COMPARISON")
    print("EmbeddingGemma 2 vs Tesseract vs EasyOCR vs PaddleOCR")
    print("=" * 80)

    # 1. Font Selection
    # Indexed Prototype Fonts (Set A)
    index_fonts = [
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    ]
    # Unseen / Held-out Test Fonts (Set B)
    test_fonts = [
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSerif-BoldItalic.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Italic.ttf",
    ]

    # Verify font paths exist
    index_fonts = [f for f in index_fonts if Path(f).exists()]
    test_fonts = [f for f in test_fonts if Path(f).exists()]

    print(f"Index Fonts (Prototypes): {[Path(f).name for f in index_fonts]}")
    print(f"Held-Out Evaluation Fonts: {[Path(f).name for f in test_fonts]}")

    # Evaluate 52 Latin classes (a-z, A-Z)
    classes = LATIN_CLASSES
    print(f"Evaluating {len(classes)} classes across {len(test_fonts)} held-out fonts ({len(classes)*len(test_fonts)} total queries).")

    # 2. Build EmbeddingGemma 2 Prototype Index
    print("\n[1/4] Initializing EmbeddingGemma 2 prototype engine...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedder = EmbeddingGemma2(dim=768, device=device)
    faiss_idx = FaissIndex(dim=768)
    proto_db = PrototypeDB(embedder=embedder, index=faiss_idx, classes=classes)
    proto_db.build_from_fonts(index_fonts)
    matcher = PrototypeMatcher()
    print("✓ EmbeddingGemma 2 prototypes ready.")

    # 3. Initialize Conventional OCR Engines
    print("\n[2/4] Initializing Conventional OCR Engines...")
    tesseract = TesseractRecognizer(psm=10)  # PSM 10 = single character
    print("✓ Tesseract ready (PSM 10)")
    easyocr = EasyOCRRecognizer(gpu=(device == "cuda"))
    print("✓ EasyOCR ready")
    paddle = PaddleRecognizer(model_name="en_PP-OCRv4_mobile_rec")
    print("✓ PaddleOCR ready")

    # 4. Generate Held-Out Test Set
    print("\n[3/4] Rendering test glyphs from held-out fonts...")
    test_data: list[tuple[str, str, np.ndarray]] = []
    for font_p in test_fonts:
        fname = Path(font_p).stem
        for char in classes:
            pil_img = render_glyph(char, font_path=font_p, canvas_size=64, font_size=44)
            test_data.append((char, fname, np.array(pil_img)))

    total_queries = len(test_data)
    print(f"✓ Generated {total_queries} test queries.")

    # 5. Run Comparison
    print("\n[4/4] Executing benchmark across all engines...")

    # A. EmbeddingGemma 2
    t0 = time.perf_counter()
    gemma_images = [img for _, _, img in test_data]
    gemma_embs = embedder.embed_images(gemma_images)
    sims, retrieved_labels = proto_db.search(gemma_embs, top_k=20)
    gemma_time = time.perf_counter() - t0

    gemma_exact = 0
    gemma_ci = 0
    gemma_top5 = 0
    for i, (true_char, _, _) in enumerate(test_data):
        pred = matcher.match_character(sims[i], retrieved_labels[i])
        if pred.character == true_char:
            gemma_exact += 1
        if pred.character.lower() == true_char.lower():
            gemma_ci += 1
        if true_char in retrieved_labels[i][:5]:
            gemma_top5 += 1

    # B. Tesseract
    t0 = time.perf_counter()
    tess_exact = 0
    tess_ci = 0
    for true_char, _, img in test_data:
        res = tesseract.recognize(img)
        p = res.text.strip()
        if p == true_char:
            tess_exact += 1
        if p.lower() == true_char.lower():
            tess_ci += 1
    tess_time = time.perf_counter() - t0

    # C. EasyOCR
    t0 = time.perf_counter()
    easy_exact = 0
    easy_ci = 0
    for true_char, _, img in test_data:
        res = easyocr.recognize(img)
        p = res.text.strip()
        if p == true_char:
            easy_exact += 1
        if p.lower() == true_char.lower():
            easy_ci += 1
    easy_time = time.perf_counter() - t0

    # D. PaddleOCR
    t0 = time.perf_counter()
    paddle_exact = 0
    paddle_ci = 0
    for true_char, _, img in test_data:
        res = paddle.recognize(img)
        p = res.text.strip()
        if p == true_char:
            paddle_exact += 1
        if p.lower() == true_char.lower():
            paddle_ci += 1
    paddle_time = time.perf_counter() - t0

    # Summary table
    print("\n" + "=" * 86)
    print(f"{'OCR System':<20} | {'Exact Top-1':<13} | {'Case-Insens Acc':<16} | {'Top-5 Acc':<12} | {'Avg Latency (ms)':<15}")
    print("-" * 86)
    results = [
        ("EmbeddingGemma 2", gemma_exact / total_queries * 100, gemma_ci / total_queries * 100, gemma_top5 / total_queries * 100, gemma_time / total_queries * 1000),
        ("Tesseract (PSM 10)", tess_exact / total_queries * 100, tess_ci / total_queries * 100, (tess_exact / total_queries * 100), tess_time / total_queries * 1000),
        ("EasyOCR", easy_exact / total_queries * 100, easy_ci / total_queries * 100, (easy_exact / total_queries * 100), easy_time / total_queries * 1000),
        ("PaddleOCR (rec)", paddle_exact / total_queries * 100, paddle_ci / total_queries * 100, (paddle_exact / total_queries * 100), paddle_time / total_queries * 1000),
    ]

    for name, exact, ci, top5, lat in results:
        top5_str = f"{top5:5.1f}%" if name == "EmbeddingGemma 2" else "N/A"
        print(f"{name:<20} | {exact:6.1f}%      | {ci:6.1f}%           | {top5_str:<12} | {lat:8.2f} ms")
    print("=" * 86)

    # Save summary to docs/decisions.md / docs
    report_path = Path("docs/ocr_engine_comparison_summary.json")
    report_data = {
        "n_queries": total_queries,
        "classes": len(classes),
        "results": {
            name: {"exact_acc": round(exact, 2), "ci_acc": round(ci, 2), "latency_ms": round(lat, 2)}
            for name, exact, ci, _, lat in results
        }
    }
    report_path.write_text(json.dumps(report_data, indent=2))
    print(f"\nSaved benchmark metrics to {report_path}")


if __name__ == "__main__":
    main()
