"""Benchmark testing image matching & KNN strategies against EmbeddingGemma 2:
1. Pure Pixel Template Matching (Normalized Cross-Correlation / NCC)
2. SIFT / ORB Feature Matching
3. Structural Similarity Index (SSIM)
4. Direct k-NN (k=1, 3, 5 with distance weighting)
5. EmbeddingGemma 2 Vector Prototype Matching (IndexFlatIP)
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim

from embedding_ocr.data.glyph_renderer import LATIN_CLASSES, render_glyph
from embedding_ocr.recognition.embedding.embedder import EmbeddingGemma2
from embedding_ocr.recognition.embedding.index import FaissIndex
from embedding_ocr.recognition.embedding.matcher import PrototypeMatcher
from embedding_ocr.recognition.embedding.prototypes import PrototypeDB


def main() -> None:
    print("=" * 85)
    print("IMAGE MATCHING & KNN VS EMBEDDINGGEMMA 2 LATIN GLYPH BENCHMARK")
    print("=" * 85)

    # 1. Fonts setup
    index_fonts = [
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    ]
    test_fonts = [
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSerif-BoldItalic.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Italic.ttf",
    ]

    classes = LATIN_CLASSES  # 52 classes
    print(f"Index Fonts: {[Path(f).name for f in index_fonts]}")
    print(f"Test Fonts:  {[Path(f).name for f in test_fonts]}")
    print(f"Evaluating {len(classes)} classes across 3 held-out fonts (156 queries).")

    # Render prototypes (Gallery)
    gallery_images: list[np.ndarray] = []
    gallery_grays: list[np.ndarray] = []
    gallery_labels: list[str] = []

    for f_p in index_fonts:
        for c in classes:
            pil_img = render_glyph(c, f_p, canvas_size=64, font_size=44)
            arr = np.array(pil_img)
            gallery_images.append(arr)
            gallery_grays.append(cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY))
            gallery_labels.append(c)

    # Render test queries (Probe)
    test_data: list[tuple[str, np.ndarray, np.ndarray]] = []
    for f_p in test_fonts:
        for c in classes:
            pil_img = render_glyph(c, f_p, canvas_size=64, font_size=44)
            arr = np.array(pil_img)
            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
            test_data.append((c, arr, gray))

    total = len(test_data)

    # --- Method 1: Pixel Template Matching (Normalized Cross-Correlation) ---
    print("\n[1/5] Evaluating Pixel Template Matching (NCC)...")
    t0 = time.perf_counter()
    ncc_correct = 0
    ncc_ci = 0
    for true_c, _, q_gray in test_data:
        best_score = -1.0
        best_pred = ""
        for p_gray, p_lbl in zip(gallery_grays, gallery_labels, strict=False):
            # cv2.matchTemplate with TM_CCOEFF_NORMED
            res = cv2.matchTemplate(q_gray, p_gray, cv2.TM_CCOEFF_NORMED)
            score = float(res[0, 0])
            if score > best_score:
                best_score = score
                best_pred = p_lbl
        if best_pred == true_c:
            ncc_correct += 1
        if best_pred.lower() == true_c.lower():
            ncc_ci += 1
    ncc_time = (time.perf_counter() - t0) / total * 1000

    # --- Method 2: SSIM (Structural Similarity Index) ---
    print("[2/5] Evaluating Structural Similarity Index (SSIM)...")
    t0 = time.perf_counter()
    ssim_correct = 0
    ssim_ci = 0
    for true_c, _, q_gray in test_data:
        best_score = -2.0
        best_pred = ""
        for p_gray, p_lbl in zip(gallery_grays, gallery_labels, strict=False):
            score = ssim(q_gray, p_gray)
            if score > best_score:
                best_score = score
                best_pred = p_lbl
        if best_pred == true_c:
            ssim_correct += 1
        if best_pred.lower() == true_c.lower():
            ssim_ci += 1
    ssim_time = (time.perf_counter() - t0) / total * 1000

    # --- Method 3 & 4: EmbeddingGemma 2 with 1-NN vs k-NN (k=5) ---
    print("[3/5] Extracting EmbeddingGemma 2 representations...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedder = EmbeddingGemma2(dim=768, device=device)
    faiss_idx = FaissIndex(dim=768)
    proto_db = PrototypeDB(embedder=embedder, index=faiss_idx, classes=classes)
    proto_db.build_from_fonts(index_fonts)

    query_imgs = [item[1] for item in test_data]
    t0 = time.perf_counter()
    q_embs = embedder.embed_images(query_imgs)
    sims, retrieved_labels = proto_db.search(q_embs, top_k=15)
    gemma_embed_lat = (time.perf_counter() - t0) / total * 1000

    # k=1 NN
    k1_correct = 0
    k1_ci = 0
    for i, (true_c, _, _) in enumerate(test_data):
        pred = retrieved_labels[i][0]
        if pred == true_c:
            k1_correct += 1
        if pred.lower() == true_c.lower():
            k1_ci += 1

    # k=5 Weighted k-NN (Sum of similarity scores per class over top-5)
    k5_correct = 0
    k5_ci = 0
    for i, (true_c, _, _) in enumerate(test_data):
        class_weights: dict[str, float] = {}
        for rank in range(5):
            lbl = retrieved_labels[i][rank]
            weight = float(sims[i][rank])
            class_weights[lbl] = class_weights.get(lbl, 0.0) + weight
        pred = max(class_weights.items(), key=lambda x: x[1])[0]
        if pred == true_c:
            k5_correct += 1
        if pred.lower() == true_c.lower():
            k5_ci += 1

    # Aggregated Max per Class + Relative Height Matcher
    matcher = PrototypeMatcher()
    matcher_correct = 0
    matcher_ci = 0
    for i, (true_c, _, _) in enumerate(test_data):
        pred_match = matcher.match_character(sims[i], retrieved_labels[i])
        if pred_match.character == true_c:
            matcher_correct += 1
        if pred_match.character.lower() == true_c.lower():
            matcher_ci += 1

    # --- Summary Table ---
    print("\n" + "=" * 85)
    print(
        f"{'Method / Matching Strategy':<35} | {'Exact Top-1':<12} | {'Case-Insens':<12} | {'Latency (ms)':<12}"
    )
    print("-" * 85)
    benchmarks = [
        (
            "Pixel NCC Template Match",
            ncc_correct / total * 100,
            ncc_ci / total * 100,
            ncc_time,
        ),
        (
            "Structural Similarity (SSIM)",
            ssim_correct / total * 100,
            ssim_ci / total * 100,
            ssim_time,
        ),
        (
            "EmbeddingGemma (1-NN)",
            k1_correct / total * 100,
            k1_ci / total * 100,
            gemma_embed_lat,
        ),
        (
            "EmbeddingGemma (k-NN, k=5 weighted)",
            k5_correct / total * 100,
            k5_ci / total * 100,
            gemma_embed_lat,
        ),
        (
            "EmbeddingGemma (Max-Agg + Matcher)",
            matcher_correct / total * 100,
            matcher_ci / total * 100,
            gemma_embed_lat,
        ),
    ]

    for name, exact, ci, lat in benchmarks:
        print(f"{name:<35} | {exact:6.1f}%      | {ci:6.1f}%      | {lat:8.2f} ms")
    print("=" * 85)


if __name__ == "__main__":
    main()
