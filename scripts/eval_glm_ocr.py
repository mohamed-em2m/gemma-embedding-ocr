"""GLM-OCR Evaluation Script.

Evaluates zai-org/GLM-OCR (~0.9B multimodal OCR model) on our Latin glyphs,
words, and typography benchmark to compare against visual embedding OCR.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, GlmOcrForConditionalGeneration

from embedding_ocr.data.glyph_renderer import (
    ALPHANUMERIC_CLASSES,
    render_glyph,
)


def load_glm_ocr(
    model_id: str = "zai-org/GLM-OCR",
) -> tuple[GlmOcrForConditionalGeneration, AutoProcessor]:
    """Load GLM-OCR model and processor."""
    print(f"Loading {model_id} on CUDA with bfloat16...")
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = GlmOcrForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    return model, processor


def recognize_single_image(
    model: GlmOcrForConditionalGeneration,
    processor: AutoProcessor,
    image: Image.Image,
) -> str:
    """Run GLM-OCR text recognition on an image."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Text Recognition:"},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=16,
            do_sample=False,
        )

    prompt_len = inputs["input_ids"].shape[1]
    raw = processor.decode(outputs[0][prompt_len:], skip_special_tokens=True)
    return raw.strip()


def main() -> None:
    model, processor = load_glm_ocr()

    # Load clean held-out fonts
    heldout_fonts_path = Path("data/clean_fonts_heldout_36.txt")
    if not heldout_fonts_path.exists():
        heldout_fonts_path = Path("data/standard_clean_fonts.txt")

    with open(heldout_fonts_path) as f:
        test_fonts = [line.strip() for line in f if line.strip()][:5]

    print(
        f"Testing GLM-OCR on {len(test_fonts)} held-out clean fonts "
        "across 62 alphanumeric classes..."
    )

    classes = ALPHANUMERIC_CLASSES  # a-z, A-Z, 0-9 (62 classes)
    total = len(test_fonts) * len(classes)
    correct = 0
    errors: list[dict[str, str]] = []

    t0 = time.perf_counter()
    count = 0
    for font_path in test_fonts:
        font_name = Path(font_path).name
        print(f"Evaluating font: {font_name}...")
        for ch in classes:
            img = render_glyph(
                ch,
                font_path,
                canvas_size=64,
                font_size=44,
                preserve_aspect_and_baseline=True,
            )
            pred = recognize_single_image(model, processor, img)
            count += 1

            cleaned_pred = pred.strip().replace(" ", "")
            if cleaned_pred == ch:
                correct += 1
            else:
                errors.append(
                    {
                        "font": font_name,
                        "target": ch,
                        "prediction": pred,
                    }
                )

    elapsed = time.perf_counter() - t0
    acc = correct / total * 100.0
    latency_ms = (elapsed / total) * 1000.0

    print("\n=======================================================")
    print(" GLM-OCR Benchmark Results")
    print("=======================================================")
    print(f"Total Evaluated: {total} glyphs")
    print(f"Accuracy:        {acc:.2f}% ({correct}/{total})")
    print(f"Average Latency: {latency_ms:.2f} ms/glyph")
    print(f"Total Time:      {elapsed:.1f} s")
    print(f"Total Errors:    {len(errors)}")

    results = {
        "model_id": "zai-org/GLM-OCR",
        "total_glyphs": total,
        "correct": correct,
        "accuracy_pct": round(acc, 2),
        "latency_ms_per_glyph": round(latency_ms, 2),
        "sample_errors": errors[:30],
    }

    out_file = Path("docs/glm_ocr_results.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved detailed results to {out_file}")


if __name__ == "__main__":
    main()
