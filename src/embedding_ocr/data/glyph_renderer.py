"""Glyph rendering utilities for Phase 0 and beyond.

Renders individual characters (letters, digits, punctuation, math symbols)
to PIL images for embedding experiments with aspect-ratio and baseline preservation.
"""

from __future__ import annotations

import string
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Canonical character classes
LATIN_LETTERS: list[str] = list(string.ascii_lowercase + string.ascii_uppercase)
DIGITS: list[str] = list(string.digits)
PUNCTUATION: list[str] = list(".,:;!?\"'-()[]{}/@#%&*+=<>")
ALPHANUMERIC_CLASSES: list[str] = LATIN_LETTERS + DIGITS
ALL_CHAR_CLASSES: list[str] = ALPHANUMERIC_CLASSES + PUNCTUATION

# Backward compatibility alias
LATIN_CLASSES: list[str] = LATIN_LETTERS

# Known ambiguous case pairs (same shape, differ only in size when cropped)
CASE_PAIRS: list[tuple[str, str]] = [
    ("c", "C"),
    ("o", "O"),
    ("s", "S"),
    ("v", "V"),
    ("w", "W"),
    ("x", "X"),
    ("z", "Z"),
    ("u", "U"),
    ("p", "P"),
    ("k", "K"),
    ("m", "M"),
]

# Other known look-alike pairs to track (including digits like 0/O, 1/l/I)
LOOKALIKE_PAIRS: list[tuple[str, str]] = [
    ("I", "l"),
    ("i", "j"),
    ("g", "q"),
    ("b", "d"),
    ("p", "q"),
    ("n", "h"),
    ("u", "v"),
    ("O", "0"),
    ("o", "0"),
    ("l", "1"),
    ("I", "1"),
    ("S", "5"),
    ("s", "5"),
    ("Z", "2"),
    ("z", "2"),
    ("B", "8"),
]


def render_glyph(
    char: str,
    font_path: str | Path,
    canvas_size: int = 64,
    font_size: int = 44,
    preserve_aspect_and_baseline: bool = True,
) -> Image.Image:
    """Render a single character on a white canvas.

    Args:
        char: The character to render.
        font_path: Path to the TrueType font file.
        canvas_size: Width and height of the square canvas in pixels.
        font_size: Font size in points.
        preserve_aspect_and_baseline: If True, uses typography baseline and
            cap-height metric so that lowercase glyphs naturally sit on the baseline
            and have accurate relative height (resolving case-pair ambiguity).

    Returns:
        PIL Image with the character rendered on a white background.
    """
    image = Image.new("RGB", (canvas_size, canvas_size), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype(str(font_path), font_size)
    except Exception:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), char, font=font)
    w = bbox[2] - bbox[0]

    # Center horizontally
    x = (canvas_size - w) / 2.0 - bbox[0]

    if preserve_aspect_and_baseline:
        # Align relative to the font's reference capital letter 'H' baseline
        ref_bbox = draw.textbbox((0, 0), "H", font=font)
        ref_h = ref_bbox[3] - ref_bbox[1]
        baseline_y = (canvas_size - ref_h) / 2.0 - ref_bbox[1] + ref_h
        y = baseline_y - ref_h - ref_bbox[1]
        draw.text((x, y), char, fill=(0, 0, 0), font=font)
    else:
        # Standard independent bounding-box centering
        h = bbox[3] - bbox[1]
        y = (canvas_size - h) / 2.0 - bbox[1]
        draw.text((x, y), char, fill=(0, 0, 0), font=font)

    return image


def render_glyph_with_features(
    char: str,
    font_path: str | Path,
    canvas_size: int = 64,
    font_size: int = 44,
) -> tuple[Image.Image, dict[str, float]]:
    """Render a character and extract typography geometric features.

    Features include:
        - aspect_ratio: width / height
        - relative_height: height / reference_cap_height
        - relative_width: width / canvas_size
    """
    img = render_glyph(
        char,
        font_path,
        canvas_size=canvas_size,
        font_size=font_size,
        preserve_aspect_and_baseline=True,
    )
    try:
        font = ImageFont.truetype(str(font_path), font_size)
    except Exception:
        font = ImageFont.load_default()

    dummy = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(dummy)
    bbox = draw.textbbox((0, 0), char, font=font)
    w = max(1.0, float(bbox[2] - bbox[0]))
    h = max(1.0, float(bbox[3] - bbox[1]))

    ref_bbox = draw.textbbox((0, 0), "H", font=font)
    ref_h = max(1.0, float(ref_bbox[3] - ref_bbox[1]))

    features = {
        "aspect_ratio": w / h,
        "relative_height": h / ref_h,
        "relative_width": w / float(canvas_size),
    }
    return img, features


def render_glyph_array(
    char: str,
    font_path: str | Path,
    canvas_size: int = 64,
    font_size: int = 44,
    preserve_aspect_and_baseline: bool = True,
) -> np.ndarray:
    """Render a glyph and return as a uint8 numpy array (H, W, 3)."""
    img = render_glyph(
        char,
        font_path,
        canvas_size,
        font_size,
        preserve_aspect_and_baseline,
    )
    return np.array(img, dtype=np.uint8)


def render_all_glyphs(
    font_paths: list[str | Path],
    classes: list[str] | None = None,
    canvas_size: int = 64,
    font_size: int = 44,
    preserve_aspect_and_baseline: bool = True,
) -> tuple[list[Image.Image], list[str]]:
    """Render all characters for a list of fonts.

    Args:
        font_paths: List of font file paths.
        classes: Character classes to render (default: LATIN_CLASSES).
        canvas_size: Canvas size in pixels.
        font_size: Font size in points.
        preserve_aspect_and_baseline: Baseline alignment switch.

    Returns:
        Tuple of (images, labels) where images[i] corresponds to labels[i].
    """
    if classes is None:
        classes = LATIN_CLASSES

    images: list[Image.Image] = []
    labels: list[str] = []

    for font_path in font_paths:
        for char in classes:
            img = render_glyph(
                char,
                font_path,
                canvas_size,
                font_size,
                preserve_aspect_and_baseline,
            )
            images.append(img)
            labels.append(char)

    return images, labels
