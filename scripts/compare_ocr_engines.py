"""Compare OCR engines: Tesseract, EasyOCR, PaddleOCR vs Ground Truth glyphs."""

from __future__ import annotations

import time

import numpy as np

from embedding_ocr.data.glyph_renderer import render_glyph
from embedding_ocr.recognition.easyocr import EasyOCRRecognizer
from embedding_ocr.recognition.paddle import PaddleRecognizer
from embedding_ocr.recognition.tesseract import TesseractRecognizer


def main() -> None:
    print("=== Initializing OCR Recognizers ===")
    tesseract = TesseractRecognizer(psm=10)
    print("✓ Tesseract initialized")

    easyocr = EasyOCRRecognizer(lang_list=["en"], gpu=True)
    print("✓ EasyOCR initialized")

    paddle = PaddleRecognizer(model_name="en_PP-OCRv4_mobile_rec")
    print("✓ PaddleOCR (en_PP-OCRv4_mobile_rec) initialized")

    # Sample test set: subset of letters across upper and lower case
    test_chars = [
        "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
        "a", "b", "c", "d", "e", "f", "g", "h", "i", "j",
        "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
        "k", "l", "m", "n", "o", "p", "q", "r", "s", "t",
    ]

    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

    engines = {
        "Tesseract": tesseract,
        "EasyOCR": easyocr,
        "PaddleOCR": paddle,
    }

    results: dict[str, dict[str, int | float]] = {
        name: {"exact_match": 0, "case_insensitive": 0, "total": len(test_chars), "time_sec": 0.0}
        for name in engines
    }

    print(f"\nEvaluating on {len(test_chars)} synthetic glyph images rendered with DejaVuSans...")

    for char in test_chars:
        pil_img = render_glyph(char, font_path=font_path, canvas_size=64, font_size=44)
        np_img = np.array(pil_img)

        for name, engine in engines.items():
            t0 = time.perf_counter()
            res = engine.recognize(np_img)
            dt = time.perf_counter() - t0

            results[name]["time_sec"] += dt
            pred = res.text.strip()

            if pred == char:
                results[name]["exact_match"] += 1
            if pred.lower() == char.lower():
                results[name]["case_insensitive"] += 1

    print("\n" + "=" * 68)
    print(f"{'Engine':<12} | {'Exact Acc':<12} | {'Case-Insens Acc':<16} | {'Avg Latency (ms)':<15}")
    print("-" * 68)
    for name, stat in results.items():
        total = int(stat["total"])
        exact_acc = (int(stat["exact_match"]) / total) * 100
        ci_acc = (int(stat["case_insensitive"]) / total) * 100
        avg_lat = (float(stat["time_sec"]) / total) * 1000
        print(f"{name:<12} | {exact_acc:6.1f}%      | {ci_acc:6.1f}%           | {avg_lat:8.2f} ms")
    print("=" * 68)


if __name__ == "__main__":
    main()
