"""Phase 0 Spike: Glyph Embedding + FAISS Search (Extended).

Evaluates Latin glyphs, digits, and punctuation embedding and vector matching
with baseline & aspect ratio preservation using EmbeddingGemma 2.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import torch
import yaml
from PIL import Image
from sentence_transformers import SentenceTransformer

from embedding_ocr.data.glyph_renderer import (
    CASE_PAIRS,
    LOOKALIKE_PAIRS,
    render_glyph,
)


def build_faiss_index(vectors: np.ndarray) -> faiss.IndexFlatIP:
    """Build an exact inner-product FAISS index for L2-normalized vectors."""
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors.astype(np.float32))
    return index


def evaluate_retrieval(
    query_embs: np.ndarray,
    query_labels: list[str],
    index: faiss.IndexFlatIP,
    index_labels: list[str],
    top_k: int = 5,
) -> tuple[float, float, list[tuple[str, str, float]]]:
    """Compute Top-1, Top-5 accuracy and capture errors."""
    similarities, indices = index.search(query_embs.astype(np.float32), top_k)
    top1_correct = 0
    top5_correct = 0
    errors: list[tuple[str, str, float]] = []

    for i, true_label in enumerate(query_labels):
        retrieved_labels = [index_labels[idx] for idx in indices[i]]
        if retrieved_labels[0] == true_label:
            top1_correct += 1
        else:
            errors.append((true_label, retrieved_labels[0], float(similarities[i][0])))

        if true_label in retrieved_labels[:top_k]:
            top5_correct += 1

    n = len(query_labels)
    return top1_correct / n, top5_correct / n, errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 0 Extended Spike")
    parser.add_argument("--config", type=str, default="configs/spike_az.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    classes = list(cfg["rendering"]["classes"])
    print(f"Loaded {len(classes)} classes (letters, digits, punctuation/symbols).")

    use_cuda = torch.cuda.is_available() and cfg["embedder"]["device"] != "cpu"
    device = "cuda" if use_cuda else "cpu"
    print(f"Loading {cfg['embedder']['model_id']} on {device}...")
    model_kwargs: dict[str, Any] = {}
    if device == "cuda":
        model_kwargs["torch_dtype"] = torch.bfloat16
    model = SentenceTransformer(
        cfg["embedder"]["model_id"],
        model_kwargs=model_kwargs,
        device=device,
    )

    indexed_fonts = cfg["rendering"]["fonts_indexed"]
    heldout_fonts = cfg["rendering"]["fonts_heldout"]
    img_size = cfg["rendering"]["image_size"][0]
    font_size = cfg["rendering"]["font_size"]
    preserve_aspect = cfg["rendering"].get("preserve_aspect_and_baseline", True)

    print("Rendering glyphs with baseline & aspect ratio preservation...")
    indexed_images: list[Image.Image] = []
    indexed_labels: list[str] = []
    for font_path in indexed_fonts:
        for ch in classes:
            img = render_glyph(
                ch,
                font_path,
                img_size,
                font_size,
                preserve_aspect_and_baseline=preserve_aspect,
            )
            indexed_images.append(img)
            indexed_labels.append(ch)

    heldout_images: list[Image.Image] = []
    heldout_labels: list[str] = []
    for font_path in heldout_fonts:
        for ch in classes:
            img = render_glyph(
                ch,
                font_path,
                img_size,
                font_size,
                preserve_aspect_and_baseline=preserve_aspect,
            )
            heldout_images.append(img)
            heldout_labels.append(ch)

    n_indexed = len(indexed_images)
    n_heldout = len(heldout_images)
    print(f"Rendered {n_indexed} indexed glyphs ({len(indexed_fonts)} fonts)")
    print(f"Rendered {n_heldout} held-out glyphs ({len(heldout_fonts)} fonts).")

    results: dict[str, Any] = {
        "dimensions": {},
        "costs": {},
        "num_classes": len(classes),
        "fonts_indexed_count": len(indexed_fonts),
        "fonts_heldout_count": len(heldout_fonts),
    }

    # E5: Latency and Memory measurement at 768d
    print("\n--- Running E5: Cost & Latency Benchmark ---")
    t0 = time.perf_counter()
    for _ in range(5):
        _ = model.encode(  # type: ignore[call-overload]
            indexed_images[0], truncate_dim=768, normalize_embeddings=True
        )
    t1 = time.perf_counter()
    lat_b1 = (t1 - t0) / 5.0 * 1000.0  # ms

    sample_batch = indexed_images[:16]
    t0 = time.perf_counter()
    for _ in range(3):
        _ = model.encode(  # type: ignore[call-overload]
            sample_batch,
            batch_size=16,
            truncate_dim=768,
            normalize_embeddings=True,
        )
    t1 = time.perf_counter()
    lat_b16 = (t1 - t0) / (3 * 16) * 1000.0  # ms per image

    max_mem_mb = 0.0
    if device == "cuda":
        max_mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    results["costs"] = {
        "device": device,
        "latency_b1_ms": round(lat_b1, 2),
        "latency_b16_per_item_ms": round(lat_b16, 2),
        "gpu_max_mem_mb": round(max_mem_mb, 2),
    }
    print(
        f"Latency batch=1: {lat_b1:.2f} ms"
        f" | batch=16 per item: {lat_b16:.2f} ms"
        f" | GPU Mem: {max_mem_mb:.2f} MB"
    )

    batch_size = cfg["embedder"]["batch_size"]
    dims = cfg["embedder"]["dims"]

    for dim in dims:
        print(f"\n=================== Evaluating Dim: {dim} ===================")
        dim_res: dict[str, Any] = {}

        # 1. Text Embeddings
        text_embs = model.encode(
            classes,
            truncate_dim=dim,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        # 2. Image Embeddings for Indexed Fonts
        img_indexed_embs = model.encode(
            indexed_images,
            batch_size=batch_size,
            truncate_dim=dim,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        # 3. Image Embeddings for Held-out Fonts
        img_heldout_embs = model.encode(
            heldout_images,
            batch_size=batch_size,
            truncate_dim=dim,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        # E1: Text Index self-retrieval
        text_index = build_faiss_index(text_embs)
        e1_top1, e1_top5, _ = evaluate_retrieval(
            text_embs, classes, text_index, classes
        )
        dim_res["E1_text_self"] = {
            "top1": e1_top1,
            "top5": e1_top5,
        }
        print(
            f"[E1 Text Self] Top-1: {e1_top1 * 100:.1f}%, Top-5: {e1_top5 * 100:.1f}%"
        )

        # E2: Cross-modal (Held-out Images -> Text Index)
        e2_top1, e2_top5, e2_errs = evaluate_retrieval(
            img_heldout_embs, heldout_labels, text_index, classes
        )
        dim_res["E2_image_to_text"] = {
            "top1": e2_top1,
            "top5": e2_top5,
            "sample_errors": e2_errs[:15],
        }
        print(
            f"[E2 Image->Text] Top-1: {e2_top1 * 100:.1f}%, Top-5: {e2_top5 * 100:.1f}%"
        )

        # E3: Image -> Image across fonts (Held-out -> Indexed Index)
        img_index = build_faiss_index(img_indexed_embs)
        e3_top1, e3_top5, e3_errs = evaluate_retrieval(
            img_heldout_embs, heldout_labels, img_index, indexed_labels
        )
        dim_res["E3_image_to_image"] = {
            "top1": e3_top1,
            "top5": e3_top5,
            "total_queries": len(heldout_labels),
            "total_errors": len(e3_errs),
            "sample_errors": e3_errs[:20],
        }
        print(
            f"[E3 Image->Image] Top-1: {e3_top1 * 100:.2f}%,"
            f" Top-5: {e3_top5 * 100:.2f}%"
            f" (Errors: {len(e3_errs)} / {len(heldout_labels)})"
        )

        # Case-pair and Look-alike ambiguity analysis on E3
        case_confusions = 0
        lookalike_confusions = 0
        for true_l, pred_l, _ in e3_errs:
            for c_low, c_up in CASE_PAIRS:
                if (true_l == c_low and pred_l == c_up) or (
                    true_l == c_up and pred_l == c_low
                ):
                    case_confusions += 1
            for p1, p2 in LOOKALIKE_PAIRS:
                if (true_l == p1 and pred_l == p2) or (true_l == p2 and pred_l == p1):
                    lookalike_confusions += 1

        dim_res["case_pair_confusions_in_e3"] = case_confusions
        dim_res["lookalike_confusions_in_e3"] = lookalike_confusions
        print(
            f"Case-pair confusions in E3: {case_confusions}/{len(e3_errs)} | "
            f"Look-alike confusions: {lookalike_confusions}/{len(e3_errs)}"
        )

        results["dimensions"][dim] = dim_res

    out_path = Path("docs/spike_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved raw spike results to {out_path}")


if __name__ == "__main__":
    main()
