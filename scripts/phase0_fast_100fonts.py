"""Fast Multi-Font Spike: One-pass 768d embedding with Matryoshka slicing.

Processes fonts once in batches, slices vectors locally for 768, 512, 256, 128,
and evaluates exact FAISS cosine retrieval across alphabet, digits, and symbols.
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


def slice_and_normalize(vectors: np.ndarray, target_dim: int) -> np.ndarray:
    """Slice leading dimensions and re-apply L2 normalization."""
    sliced = vectors[:, :target_dim].copy()
    faiss.normalize_L2(sliced)
    return sliced


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast Multi-Font Spike")
    parser.add_argument("--config", type=str, default="configs/spike_az.yaml")
    parser.add_argument(
        "--indexed-fonts", type=str, default="data/fonts_indexed_100.txt"
    )
    parser.add_argument(
        "--heldout-fonts", type=str, default="data/fonts_heldout_100.txt"
    )
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    with open(args.indexed_fonts) as f:
        indexed_fonts = [line.strip() for line in f if line.strip()]

    with open(args.heldout_fonts) as f:
        heldout_fonts = [line.strip() for line in f if line.strip()]

    classes = list(cfg["rendering"]["classes"])
    print(f"Loaded {len(classes)} classes.")
    print(f"Indexed fonts: {len(indexed_fonts)} | Held-out: {len(heldout_fonts)}")

    use_cuda = torch.cuda.is_available() and cfg["embedder"]["device"] != "cpu"
    device = "cuda" if use_cuda else "cpu"
    print(f"Loading {cfg['embedder']['model_id']} on {device} (bfloat16)...")
    model_kwargs: dict[str, Any] = (
        {"torch_dtype": torch.bfloat16} if device == "cuda" else {}
    )
    model = SentenceTransformer(
        cfg["embedder"]["model_id"],
        model_kwargs=model_kwargs,
        device=device,
    )

    img_size = cfg["rendering"]["image_size"][0]
    font_size = cfg["rendering"]["font_size"]
    preserve_aspect = cfg["rendering"].get("preserve_aspect_and_baseline", True)

    print("\nRendering prototype glyphs...")
    t0 = time.perf_counter()
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

    t_render = time.perf_counter() - t0
    print(
        f"Rendered {len(indexed_images)} indexed and "
        f"{len(heldout_images)} held-out glyphs in {t_render:.1f}s."
    )

    batch_size = args.batch_size
    print(f"\nEncoding all images ONCE at native dim 768 (batch_size={batch_size})...")

    # 1. Text Embeddings
    print("Encoding text classes...")
    text_embs_768 = model.encode(
        classes,
        batch_size=batch_size,
        truncate_dim=768,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32)

    # 2. Prototype Images
    print(f"Encoding {len(indexed_images)} prototype images...")
    t_start = time.perf_counter()
    img_indexed_embs_768 = model.encode(
        indexed_images,
        batch_size=batch_size,
        truncate_dim=768,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32)
    t_proto = time.perf_counter() - t_start
    rate_proto = len(indexed_images) / t_proto
    print(f"Prototypes encoded in {t_proto:.1f}s ({rate_proto:.1f} glyphs/sec).")

    # 3. Held-out Query Images
    print(f"Encoding {len(heldout_images)} held-out query images...")
    t_start = time.perf_counter()
    img_heldout_embs_768 = model.encode(
        heldout_images,
        batch_size=batch_size,
        truncate_dim=768,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32)
    t_queries = time.perf_counter() - t_start
    rate_queries = len(heldout_images) / t_queries
    print(f"Queries encoded in {t_queries:.1f}s ({rate_queries:.1f} glyphs/sec).")

    results: dict[str, Any] = {
        "dimensions": {},
        "num_classes": len(classes),
        "fonts_indexed_count": len(indexed_fonts),
        "fonts_heldout_count": len(heldout_fonts),
        "total_prototypes": len(indexed_images),
        "total_queries": len(heldout_images),
    }

    dims = [768, 512, 256, 128]
    for dim in dims:
        print(f"\n=================== Vector Evaluation Dim: {dim} ===================")
        dim_res: dict[str, Any] = {}

        t_cur_text = slice_and_normalize(text_embs_768, dim)
        t_cur_proto = slice_and_normalize(img_indexed_embs_768, dim)
        t_cur_queries = slice_and_normalize(img_heldout_embs_768, dim)

        # E1: Text Index self-retrieval
        text_index = build_faiss_index(t_cur_text)
        e1_top1, e1_top5, _ = evaluate_retrieval(
            t_cur_text, classes, text_index, classes
        )
        dim_res["E1_text_self"] = {"top1": e1_top1, "top5": e1_top5}
        print(
            f"[E1 Text Self] Top-1: {e1_top1 * 100:.1f}%, Top-5: {e1_top5 * 100:.1f}%"
        )

        # E2: Cross-modal (Held-out Images -> Text Index)
        e2_top1, e2_top5, e2_errs = evaluate_retrieval(
            t_cur_queries, heldout_labels, text_index, classes
        )
        dim_res["E2_image_to_text"] = {"top1": e2_top1, "top5": e2_top5}
        print(
            f"[E2 Image->Text] Top-1: {e2_top1 * 100:.2f}%, Top-5: {e2_top5 * 100:.2f}%"
        )

        # E3: Image -> Image across fonts
        img_index = build_faiss_index(t_cur_proto)
        e3_top1, e3_top5, e3_errs = evaluate_retrieval(
            t_cur_queries, heldout_labels, img_index, indexed_labels
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

    out_path = Path("docs/spike_results_100fonts.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved raw results to {out_path}")


if __name__ == "__main__":
    main()
