"""Render short text labels and apply OCR-specific hard augmentations."""

from __future__ import annotations

import io
import logging
import random
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

LOGGER = logging.getLogger(__name__)
DEFAULT_CANVAS_SIZE = 128
FONT_CANDIDATES: tuple[str, ...] = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/liberation-sans-fonts/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
)


def resolve_default_font() -> str:
    """Return an installed font suitable for tests and fallback rendering."""
    for candidate in FONT_CANDIDATES:
        if Path(candidate).exists():
            return candidate
    matches = sorted(Path("/usr/share/fonts").rglob("*.ttf"))
    if matches:
        return str(matches[0])
    msg = "No TTF font found; install fonts-dejavu or pass --fonts-dir."
    raise FileNotFoundError(msg)


def render_text(
    text: str,
    font_path: str | Path,
    *,
    canvas_size: int = DEFAULT_CANVAS_SIZE,
    font_size: int | None = None,
    margin: int = 10,
    seed: int | None = None,
) -> Image.Image:
    """Render text centered on a white square canvas.

    Text longer than one line is wrapped instead of being shrunk to an
    unreadable size: at 128 px a 20-word label becomes a few lines of legible
    glyphs. Letter height relative to the canvas is retained as a weak case cue.
    """
    rng = random.Random(seed)
    font_path = str(font_path)
    chosen_size = font_size or rng.randint(28, 54)
    image = Image.new("RGB", (canvas_size, canvas_size), "white")
    draw = ImageDraw.Draw(image)
    available = max(1, canvas_size - 2 * margin)

    font, lines, boxes, width, height = _fit_lines(
        draw, text, font_path, chosen_size, available
    )
    _draw_lines(draw, lines, boxes, font, canvas_size, width, height)
    return image


def _fit_lines(
    draw: ImageDraw.ImageDraw,
    text: str,
    font_path: str,
    start_size: int,
    available: int,
) -> tuple[
    ImageFont.FreeTypeFont,
    list[str],
    list[tuple[float, float, float, float]],
    float,
    float,
]:
    """Pick the largest font size at which wrapped text fits the canvas."""
    size = max(8, start_size)
    while True:
        font = _load_font(font_path, size)
        lines = _wrap_lines(draw, text, font, available)
        boxes = [_float_box(draw.textbbox((0, 0), line, font=font)) for line in lines]
        width = max((box[2] - box[0] for box in boxes), default=1.0)
        line_height = max((box[3] - box[1] for box in boxes), default=1.0)
        gap = max(1.0, size * 0.15)
        height = line_height * len(lines) + gap * (len(lines) - 1)
        if (width <= available and height <= available) or size <= 8:
            return font, lines, boxes, max(1.0, width), max(1.0, height)
        size = max(8, size - 2)


def _float_box(
    box: tuple[int, int, int, int],
) -> tuple[float, float, float, float]:
    """Convert Pillow's integer bbox to floats for consistent layout math."""
    return float(box[0]), float(box[1]), float(box[2]), float(box[3])


def _wrap_lines(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont,
    max_width: int,
) -> list[str]:
    """Greedy word wrap that keeps at least one word per line."""
    lines: list[str] = []
    for paragraph in text.splitlines() or [text]:
        words = paragraph.split()
        if not words:
            continue
        current = words[0]
        for word in words[1:]:
            trial = f"{current} {word}"
            if draw.textlength(trial, font=font) <= max_width:
                current = trial
            else:
                lines.append(current)
                current = word
        lines.append(current)
    return lines or [text]


def _draw_lines(
    draw: ImageDraw.ImageDraw,
    lines: list[str],
    boxes: list[tuple[float, float, float, float]],
    font: ImageFont.FreeTypeFont,
    canvas_size: int,
    width: float,
    height: float,
) -> None:
    """Draw wrapped lines centered as one block."""
    line_height = max((box[3] - box[1] for box in boxes), default=1.0)
    gap = max(1.0, font.size * 0.15)
    y = (canvas_size - height) / 2
    for line, box in zip(lines, boxes, strict=True):
        line_width = box[2] - box[0]
        x = (canvas_size - line_width) / 2 - box[0]
        # Center each line inside its slot so mixed ascent/descent stays even.
        offset = (line_height - (box[3] - box[1])) / 2
        draw.text((x, y + offset - box[1]), line, fill="black", font=font)
        y += line_height + gap


def _load_font(font_path: str, size: int) -> ImageFont.FreeTypeFont:
    """Load a font, falling back to Pillow's bundled font on broken files."""
    try:
        font = ImageFont.truetype(font_path, size)
        font.getbbox("Aa0")
        return font
    except OSError as exc:
        LOGGER.warning("Unusable font %s: %s; using default font", font_path, exc)
        return ImageFont.load_default(size=size)  # type: ignore[return-value]


def augment_image(
    image: Image.Image,
    *,
    hard: bool = True,
    rng: random.Random | None = None,
) -> Image.Image:
    """Apply a random OCR degradation; ``hard=True`` uses stronger warps."""
    generator = rng or random.Random()
    result = image.copy()
    if generator.random() < (0.85 if hard else 0.35):
        result = _random_perspective(result, generator, strength=0.08 if hard else 0.03)
    if generator.random() < (0.6 if hard else 0.2):
        result = result.rotate(
            generator.uniform(-8.0, 8.0) if hard else generator.uniform(-2.0, 2.0),
            resample=Image.Resampling.BICUBIC,
            fillcolor="white",
        )
    if generator.random() < (0.5 if hard else 0.2):
        result = result.filter(
            ImageFilter.GaussianBlur(generator.uniform(0.4, 1.8 if hard else 0.8))
        )
    if generator.random() < (0.45 if hard else 0.15):
        result = _add_noise(result, generator, strength=18 if hard else 7)
    if generator.random() < (0.35 if hard else 0.1):
        result = _jpeg_roundtrip(result, generator, quality=35 if hard else 70)
    if generator.random() < (0.3 if hard else 0.1):
        result = ImageEnhance.Contrast(result).enhance(generator.uniform(0.45, 0.85))
    if generator.random() < (0.25 if hard else 0.05):
        result = _occlude(result, generator)
    if generator.random() < (0.15 if hard else 0.03):
        result = _downscale_upscale(result, generator, scale=0.45 if hard else 0.7)
    if generator.random() < 0.08:
        result = ImageEnhance.Brightness(result).enhance(generator.uniform(0.75, 1.15))
    return result


def render_variants(
    text: str,
    font_paths: Sequence[str | Path],
    *,
    count: int,
    augment: bool,
    seed: int = 42,
    canvas_size: int = DEFAULT_CANVAS_SIZE,
) -> list[Image.Image]:
    """Render several variants using different fonts/seeds."""
    if not font_paths:
        msg = "At least one font path is required."
        raise ValueError(msg)
    rng = random.Random(seed)
    images: list[Image.Image] = []
    for index in range(count):
        font = str(font_paths[index % len(font_paths)])
        image = render_text(
            text,
            font,
            canvas_size=canvas_size,
            seed=rng.randrange(1_000_000),
        )
        if augment:
            image = augment_image(
                image, hard=True, rng=random.Random(rng.randrange(1_000_000))
            )
        images.append(image)
    return images


def _random_perspective(
    image: Image.Image,
    rng: random.Random,
    *,
    strength: float,
) -> Image.Image:
    array = np.array(image)
    height, width = array.shape[:2]
    offset = int(min(height, width) * strength)
    source = np.asarray(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    target = np.asarray(
        [
            [rng.randint(0, offset), rng.randint(0, offset)],
            [width - 1 - rng.randint(0, offset), rng.randint(0, offset)],
            [width - 1 - rng.randint(0, offset), height - 1 - rng.randint(0, offset)],
            [rng.randint(0, offset), height - 1 - rng.randint(0, offset)],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(source, target)
    warped = cv2.warpPerspective(
        array,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    return Image.fromarray(warped)


def _add_noise(
    image: Image.Image,
    rng: random.Random,
    *,
    strength: int,
) -> Image.Image:
    array = np.array(image).astype(np.int16)
    noise = np.random.default_rng(rng.randrange(2**32)).normal(
        0, strength, array.shape[:2]
    )
    array = np.clip(array + noise[..., None], 0, 255).astype(np.uint8)
    return Image.fromarray(array)


def _jpeg_roundtrip(
    image: Image.Image,
    rng: random.Random,
    *,
    quality: int,
) -> Image.Image:
    del rng
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def _occlude(image: Image.Image, rng: random.Random) -> Image.Image:
    result = image.copy()
    draw = ImageDraw.Draw(result)
    width, height = result.size
    rect_width = rng.randint(max(4, width // 12), max(6, width // 4))
    rect_height = rng.randint(max(4, height // 12), max(6, height // 4))
    x = rng.randint(0, max(0, width - rect_width))
    y = rng.randint(0, max(0, height - rect_height))
    draw.rectangle((x, y, x + rect_width, y + rect_height), fill="white")
    return result


def _downscale_upscale(
    image: Image.Image,
    rng: random.Random,
    *,
    scale: float,
) -> Image.Image:
    del rng
    width, height = image.size
    small = image.resize(
        (max(8, int(width * scale)), max(8, int(height * scale))),
        Image.Resampling.BILINEAR,
    )
    return small.resize((width, height), Image.Resampling.BICUBIC)
