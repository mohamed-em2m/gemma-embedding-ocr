"""Feasibility Experiment for Hierarchical Word Recognition using EmbeddingGemma 2.

Tests:
1. Exact character recognition via image-to-text embedding similarity.
2. Complete-word vs subword vs character similarity on clean and rendered text.
3. Contrastive margin between correct word candidate and distractors.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from embedding_ocr.recognition.embedding.embedder import EmbeddingGemma2


def render_text_crop(
    text: str,
    font_path: str = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    font_size: int = 36,
    padding_x: int = 16,
    padding_y: int = 12,
) -> np.ndarray:
    """Render a text crop cleanly."""
    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception:
        font = ImageFont.load_default()

    bbox = font.getbbox(text)
    w = (bbox[2] - bbox[0]) + 2 * padding_x
    h = (bbox[3] - bbox[1]) + 2 * padding_y

    img = Image.new("RGB", (max(w, 20), max(h, 20)), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((padding_x - bbox[0], padding_y - bbox[1]), text, font=font, fill=(0, 0, 0))
    return np.array(img, dtype=np.uint8)


def run_feasibility():
    print("Initializing EmbeddingGemma2 (dim=768)...")
    emb = EmbeddingGemma2(dim=768)

    # 1. Test Characters
    test_chars = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
                  "a", "b", "c", "d", "e", "f", "g", "h", "i", "j",
                  "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

    print("\n--- Experiment 1: Character Image-to-Text Matching ---")
    char_images = [render_text_crop(c) for c in test_chars]
    char_img_embs = emb.embed_images(char_images)
    char_txt_embs = emb.embed_texts(test_chars)

    # Cosine similarity matrix (N, N)
    sim_matrix = char_img_embs @ char_txt_embs.T

    top1_correct = 0
    top3_correct = 0
    for i, c in enumerate(test_chars):
        row = sim_matrix[i]
        top_indices = np.argsort(row)[::-1]
        top1_char = test_chars[top_indices[0]]
        top3_chars = [test_chars[idx] for idx in top_indices[:3]]

        is_top1 = (top1_char == c)
        is_top3 = (c in top3_chars)
        if is_top1:
            top1_correct += 1
        if is_top3:
            top3_correct += 1

    char_top1_acc = top1_correct / len(test_chars)
    char_top3_acc = top3_correct / len(test_chars)
    print(f"Character Image->Text Top-1 Accuracy: {char_top1_acc * 100:.1f}% ({top1_correct}/{len(test_chars)})")
    print(f"Character Image->Text Top-3 Accuracy: {char_top3_acc * 100:.1f}% ({top3_correct}/{len(test_chars)})")

    # 2. Test Words
    print("\n--- Experiment 2: Word Image-to-Text Matching ---")
    test_words = [
        "invoice", "total", "payment", "customer", "receipt",
        "system", "network", "python", "neural", "gemma",
        "embedding", "recognition", "digital", "document", "optical"
    ]
    distractor_words = [
        "invoices", "totally", "payments", "custom", "receive",
        "systems", "networking", "pythonic", "neuron", "gamma",
        "embedded", "recognize", "digit", "documentation", "optics"
    ]
    all_lexicon = list(set(test_words + distractor_words))

    word_images = [render_text_crop(w) for w in test_words]
    word_img_embs = emb.embed_images(word_images)
    word_txt_embs = emb.embed_texts(all_lexicon)

    word_sims = word_img_embs @ word_txt_embs.T

    word_top1 = 0
    word_top3 = 0
    margins = []

    for i, target in enumerate(test_words):
        row = word_sims[i]
        top_indices = np.argsort(row)[::-1]
        top1_w = all_lexicon[top_indices[0]]
        top2_w = all_lexicon[top_indices[1]]
        top3_list = [all_lexicon[idx] for idx in top_indices[:3]]

        sim1 = row[top_indices[0]]
        sim2 = row[top_indices[1]]
        margin = sim1 - sim2
        margins.append(float(margin))

        if top1_w == target:
            word_top1 += 1
        if target in top3_list:
            word_top3 += 1

        print(f"Target: {target:<12} Top-1: {top1_w:<12} (sim={sim1:.4f}, margin={margin:+.4f}) Top-2: {top2_w:<12} (sim={sim2:.4f})")

    word_top1_acc = word_top1 / len(test_words)
    word_top3_acc = word_top3 / len(test_words)
    print(f"\nWord Image->Text Top-1 Accuracy: {word_top1_acc * 100:.1f}% ({word_top1}/{len(test_words)})")
    print(f"Word Image->Text Top-3 Accuracy: {word_top3_acc * 100:.1f}% ({word_top3}/{len(test_words)})")
    print(f"Mean top-1 vs top-2 margin: {np.mean(margins):.4f}")

    # 3. Test Subword Splitting Feasibility
    print("\n--- Experiment 3: Subword Split Matching ---")
    compound_words = [
        ("network", ["net", "work"]),
        ("payment", ["pay", "ment"]),
        ("customer", ["cust", "omer"]),
        ("invoice", ["in", "voice"]),
    ]

    for full_w, parts in compound_words:
        print(f"\nEvaluating compound word '{full_w}' parts: {parts}")
        for part in parts:
            part_crop = render_text_crop(part)
            p_img_emb = emb.embed_images([part_crop])
            p_txt_embs = emb.embed_texts([part, full_w, "other", parts[0], parts[1]])
            cand_labels = [part, full_w, "other", parts[0], parts[1]]
            p_sims = (p_img_emb @ p_txt_embs.T)[0]
            best_idx = np.argmax(p_sims)
            print(f"  Part crop '{part}' matched '{cand_labels[best_idx]}' with sim {p_sims[best_idx]:.4f}")

    results = {
        "char_top1_acc": char_top1_acc,
        "char_top3_acc": char_top3_acc,
        "word_top1_acc": word_top1_acc,
        "word_top3_acc": word_top3_acc,
        "mean_margin": float(np.mean(margins)),
    }

    out_path = Path("docs/hierarchical_feasibility.json")
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nFeasibility metrics saved to {out_path}")


if __name__ == "__main__":
    run_feasibility()
