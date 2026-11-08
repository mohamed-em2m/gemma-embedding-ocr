"""Synthetic character-detection dataset generation for YOLO training.

The generator renders Latin and Arabic words with real font shaping, derives
one bounding box per grapheme cluster, and writes a YOLO-format dataset.  Fonts
are split before image generation so validation/test fonts never leak into
training.
"""

from __future__ import annotations

import json
import math
import random
import string
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from embedding_ocr.data.glyph_renderer import ALPHANUMERIC_CLASSES, ARABIC_CLASSES
from embedding_ocr.data.scripts import is_rtl_script, normalize_script_name

GENERATOR_VERSION = "1.0"

ARABIC_DIACRITICS = frozenset(
    chr(codepoint)
    for start, end in (
        (0x064B, 0x065F),
        (0x0670, 0x0670),
        (0x06D6, 0x06ED),
        (0x08D3, 0x08FF),
    )
    for codepoint in range(start, end + 1)
)
ARABIC_TATWEEL = "\u0640"
ARABIC_LIGATURES = ("لا", "لأ", "لإ", "لآ", "الله")

DEFAULT_ARABIC_WORDS = (
    "مرحبا",
    "العالم",
    "كتاب",
    "مدرسة",
    "السلام",
    "عليكم",
    "لغة",
    "عربية",
    "حاسوب",
    "صورة",
    "نص",
    "خط",
    "قراءة",
    "كتابة",
    "بيت",
    "شمس",
    "قمر",
    "ماء",
    "هواء",
    "أرض",
)

DEFAULT_LATIN_WORDS = (
    "hello",
    "world",
    "text",
    "image",
    "font",
    "letter",
    "detect",
    "training",
    "model",
    "dataset",
    "sample",
    "ocr",
    "vision",
    "computer",
    "python",
    "paper",
    "line",
    "word",
    "character",
    "document",
)

SCRIPT_NAMES = ("latin", "arabic")


def _dedupe(items: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            output.append(item)
    return output


DETECTION_CLASSES: list[str] = _dedupe(
    [*ALPHANUMERIC_CLASSES, *ARABIC_CLASSES, *ARABIC_LIGATURES]
)
CLASS_TO_ID: dict[str, int] = {
    name: index for index, name in enumerate(DETECTION_CLASSES)
}


@dataclass(frozen=True)
class FontInfo:
    """A font file and the scripts it covers."""

    path: Path
    scripts: tuple[str, ...]


@dataclass(frozen=True)
class UnitBox:
    """A grapheme-cluster box in layer pixel coordinates."""

    text: str
    bbox: tuple[float, float, float, float]


@dataclass(frozen=True)
class RenderedWord:
    """A rendered word as an alpha mask plus unit boxes."""

    image: Image.Image
    units: list[UnitBox]
    width: int
    height: int


@dataclass(frozen=True)
class DatasetLabel:
    """A detection label in image pixel coordinates."""

    class_id: int
    text: str
    bbox: tuple[float, float, float, float]


def _require_uharfbuzz() -> Any:
    try:
        import uharfbuzz as hb  # type: ignore[import-untyped]
    except ImportError as exc:
        message = (
            "uharfbuzz is required for Arabic shaping and grapheme boxes. "
            "Install the training extra with `uv sync --extra train`."
        )
        raise ImportError(message) from exc
    return hb


@lru_cache(maxsize=512)
def _load_font(
    font_path: str, font_size: int
) -> tuple[Any, Any, ImageFont.FreeTypeFont, float]:
    hb = _require_uharfbuzz()
    blob = hb.Blob.from_file_path(font_path)
    face = hb.Face(blob)
    hb_font = hb.Font(face)
    pil_font = ImageFont.truetype(font_path, font_size)
    return face, hb_font, pil_font, font_size / float(face.upem)


@lru_cache(maxsize=2048)
def _font_supports_text(font_path: str, text: str) -> bool:
    hb = _require_uharfbuzz()
    blob = hb.Blob.from_file_path(font_path)
    face = hb.Face(blob)
    hb_font = hb.Font(face)
    return all(
        hb_font.get_nominal_glyph(ord(char)) is not None
        for char in text
        if not char.isspace()
    )


def discover_fonts(fonts_dir: str | Path) -> list[FontInfo]:
    """Find fonts that cover the full Latin and/or Arabic training charset."""
    root = Path(fonts_dir)
    if not root.exists():
        message = f"Font directory not found: {root}"
        raise FileNotFoundError(message)

    paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".ttf", ".otf"}
    )
    fonts: list[FontInfo] = []
    latin_text = "".join(ALPHANUMERIC_CLASSES)
    arabic_text = "".join(ARABIC_CLASSES)
    for path in paths:
        try:
            supports_latin = _font_supports_text(str(path), latin_text)
            supports_arabic = _font_supports_text(str(path), arabic_text)
        except (OSError, RuntimeError, ValueError):
            continue
        scripts = tuple(
            script
            for script, supported in (
                ("latin", supports_latin),
                ("arabic", supports_arabic),
            )
            if supported
        )
        if scripts:
            fonts.append(FontInfo(path=path, scripts=scripts))
    return fonts


def _assign_split(assignment: dict[Path, str], font: FontInfo, split: str) -> None:
    priority = {"train": 0, "val": 1, "test": 2}
    current = assignment[font.path]
    if priority[split] > priority[current]:
        assignment[font.path] = split


def split_fonts(
    fonts: Sequence[FontInfo],
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 42,
) -> dict[str, list[FontInfo]]:
    """Split fonts into train/val/test with no font leakage.

    A font that supports both scripts participates in both strata.  If a
    validation/test split has no font for a script that had enough candidates,
    a ValueError is raised so the user can add fonts instead of silently
    training with an unrepresentative split.
    """
    if not fonts:
        message = "No usable fonts found."
        raise ValueError(message)
    if val_fraction < 0.0 or test_fraction < 0.0:
        message = "val_fraction and test_fraction must be non-negative."
        raise ValueError(message)
    if val_fraction + test_fraction >= 1.0:
        message = "val_fraction + test_fraction must be less than 1."
        raise ValueError(message)

    rng = random.Random(seed)
    assignment: dict[Path, str] = {font.path: "train" for font in fonts}
    pools: dict[str, list[FontInfo]] = {
        script: [font for font in fonts if script in font.scripts]
        for script in SCRIPT_NAMES
    }

    for _script, pool in pools.items():
        if len(pool) < 2:
            continue
        rng.shuffle(pool)
        count = len(pool)
        n_test = max(1, round(count * test_fraction)) if test_fraction else 0
        n_val = max(1, round(count * val_fraction)) if val_fraction else 0
        while n_test + n_val >= count:
            if n_test > 0:
                n_test -= 1
            elif n_val > 0:
                n_val -= 1
            else:
                break
        for font in pool[:n_test]:
            _assign_split(assignment, font, "test")
        for font in pool[n_test : n_test + n_val]:
            _assign_split(assignment, font, "val")

    split: dict[str, list[FontInfo]] = {"train": [], "val": [], "test": []}
    for font in fonts:
        split[assignment[font.path]].append(font)

    for split_name in ("val", "test"):
        if not split[split_name]:
            continue
        for script, pool in pools.items():
            if len(pool) < 2:
                continue
            if not any(script in font.scripts for font in split[split_name]):
                message = (
                    f"The {split_name} split has no {script} font. Add more "
                    "fonts or lower the split fractions."
                )
                raise ValueError(message)
    if not split["train"]:
        message = "Font split left no training fonts."
        raise ValueError(message)
    return split


def make_background(
    size: int,
    rng: random.Random,
    np_rng: np.random.Generator,
    background_paths: Sequence[Path] = (),
) -> Image.Image:
    """Create a procedural or photo background."""
    if background_paths and rng.random() < 0.8:
        path = rng.choice(list(background_paths))
        with Image.open(path) as source:
            image = source.convert("RGB")
            width, height = image.size
            scale = rng.uniform(0.4, 1.0)
            crop_width = max(16, int(width * scale))
            crop_height = max(16, int(height * scale))
            x = rng.randint(0, max(0, width - crop_width))
            y = rng.randint(0, max(0, height - crop_height))
            crop = image.crop((x, y, x + crop_width, y + crop_height))
            return crop.resize((size, size), Image.Resampling.LANCZOS)

    base = float(rng.randint(180, 255))
    array = np.full((size, size, 3), base, dtype=np.float32)
    axis = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    array += axis[:, None, None] * rng.uniform(-28.0, 28.0)
    array += axis[None, :, None] * rng.uniform(-28.0, 28.0)
    array += np_rng.normal(0.0, rng.uniform(2.0, 12.0), array.shape)
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), mode="RGB")


def contrast_color(
    background: Image.Image,
    box: tuple[float, float, float, float],
    rng: random.Random,
) -> tuple[int, int, int]:
    """Pick ink color with strong contrast against the target region."""
    x0, y0, x1, y1 = (int(round(value)) for value in box)
    region = background.crop((x0, y0, x1, y1)).convert("L")
    luminance = float(np.asarray(region).mean()) if region.size else 128.0
    if luminance > 128.0:
        return (
            rng.randint(0, 80),
            rng.randint(0, 80),
            rng.randint(0, 80),
        )
    return (
        rng.randint(175, 255),
        rng.randint(175, 255),
        rng.randint(175, 255),
    )


def _normalize_unit_text(raw: str, script: str) -> str | None:
    normalized = normalize_script_name(script)
    if normalized == "arabic":
        clean = "".join(
            char
            for char in raw
            if char not in ARABIC_DIACRITICS and char != ARABIC_TATWEEL
        )
    else:
        clean = "".join(char for char in raw if char in string.printable)
    if not clean:
        return None
    if clean in CLASS_TO_ID:
        return clean
    if normalized == "arabic":
        return None
    return None


def shape_word(
    text: str,
    font_path: str | Path,
    font_size: int,
    script: str,
    padding: int = 4,
) -> RenderedWord | None:
    """Shape and render a word, returning a transparent alpha mask and boxes."""
    hb = _require_uharfbuzz()
    normalized = normalize_script_name(script)
    direction = "rtl" if normalized == "arabic" else "ltr"
    language = "ar" if normalized == "arabic" else "en"
    hb_features = {"liga": 0, "clig": 0} if normalized == "latin" else {}
    pil_features = ["-liga", "-clig"] if normalized == "latin" else None

    face, hb_font, pil_font, scale = _load_font(str(font_path), font_size)
    buffer = hb.Buffer()
    buffer.add_str(text)
    buffer.guess_segment_properties()
    buffer.direction = direction
    buffer.language = language
    hb.shape(hb_font, buffer, hb_features)

    infos = list(buffer.glyph_infos)
    positions = list(buffer.glyph_positions)
    if not infos:
        return None

    clusters = sorted({int(info.cluster) for info in infos})
    next_cluster = {
        cluster: clusters[index + 1] if index + 1 < len(clusters) else len(text)
        for index, cluster in enumerate(clusters)
    }

    unit_extents: dict[int, list[float]] = {}
    pen_x = 0.0
    for info, position in zip(infos, positions, strict=False):
        extents = hb_font.get_glyph_extents(info.codepoint)
        if extents is not None:
            x0 = pen_x + position.x_offset + extents.x_bearing
            x1 = x0 + extents.width
            top = position.y_offset + extents.y_bearing
            bottom = top + extents.height
            current = unit_extents.get(int(info.cluster))
            if current is None:
                unit_extents[int(info.cluster)] = [x0, x1, top, bottom]
            else:
                current[0] = min(current[0], x0)
                current[1] = max(current[1], x1)
                current[2] = max(current[2], top)
                current[3] = min(current[3], bottom)
        pen_x += position.x_advance

    if not unit_extents:
        return None

    x_min = min(values[0] for values in unit_extents.values()) * scale
    x_max = max(values[1] for values in unit_extents.values()) * scale
    y_top = max(values[2] for values in unit_extents.values()) * scale
    y_bottom = min(values[3] for values in unit_extents.values()) * scale
    width = int(math.ceil(x_max - x_min)) + 2 * padding
    height = int(math.ceil(y_top - y_bottom)) + 2 * padding
    if width <= 0 or height <= 0:
        return None

    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    origin_x = padding - x_min
    baseline = padding + y_top
    draw.text(
        (origin_x, baseline),
        text,
        font=pil_font,
        fill=(255, 255, 255, 255),
        anchor="ls",
        direction=direction,
        language=language,
        features=pil_features,
    )

    units: list[UnitBox] = []
    for cluster, (unit_x0, unit_x1, unit_top, unit_bottom) in sorted(
        unit_extents.items()
    ):
        unit_text = _normalize_unit_text(
            text[cluster : next_cluster[cluster]], normalized
        )
        if unit_text is None:
            continue
        box = (
            padding + unit_x0 * scale - x_min,
            padding + y_top - unit_top * scale,
            padding + unit_x1 * scale - x_min,
            padding + y_top - unit_bottom * scale,
        )
        clipped = _clip_box(box, width, height)
        if clipped is not None:
            units.append(UnitBox(text=unit_text, bbox=clipped))
    if not units:
        return None
    return RenderedWord(image=image, units=units, width=width, height=height)


def _clip_box(
    box: tuple[float, float, float, float], width: int, height: int
) -> tuple[float, float, float, float] | None:
    x0 = max(0.0, min(float(width), box[0]))
    y0 = max(0.0, min(float(height), box[1]))
    x1 = max(0.0, min(float(width), box[2]))
    y1 = max(0.0, min(float(height), box[3]))
    if x1 - x0 < 1.0 or y1 - y0 < 1.0:
        return None
    return x0, y0, x1, y1


def _colorize(mask: Image.Image, color: tuple[int, int, int]) -> Image.Image:
    colored = Image.new("RGBA", mask.size, (*color, 255))
    colored.putalpha(mask.getchannel("A"))
    return colored


def _rotate_word(
    rendered: RenderedWord, angle: float
) -> tuple[Image.Image, list[UnitBox], int, int]:
    if abs(angle) < 0.01:
        return rendered.image, rendered.units, rendered.width, rendered.height
    rotated = rendered.image.rotate(
        angle, expand=True, resample=Image.Resampling.BICUBIC
    )
    radians = math.radians(angle)
    cosine = math.cos(radians)
    sine = math.sin(radians)
    out_units: list[UnitBox] = []
    for unit in rendered.units:
        x0, y0, x1, y1 = unit.bbox
        xs: list[float] = []
        ys: list[float] = []
        for px, py in ((x0, y0), (x1, y0), (x1, y1), (x0, y1)):
            dx = px - rendered.width / 2.0
            dy = py - rendered.height / 2.0
            xs.append(dx * cosine + dy * sine + rotated.width / 2.0)
            ys.append(-dx * sine + dy * cosine + rotated.height / 2.0)
        box = _clip_box((min(xs), min(ys), max(xs), max(ys)), *rotated.size)
        if box is not None:
            out_units.append(UnitBox(text=unit.text, bbox=box))
    return rotated, out_units, rotated.width, rotated.height


def _scale_word(rendered: RenderedWord, factor: float) -> RenderedWord:
    width = max(1, int(round(rendered.width * factor)))
    height = max(1, int(round(rendered.height * factor)))
    image = rendered.image.resize((width, height), Image.Resampling.LANCZOS)
    units = [
        UnitBox(
            text=unit.text,
            bbox=(
                unit.bbox[0] * factor,
                unit.bbox[1] * factor,
                unit.bbox[2] * factor,
                unit.bbox[3] * factor,
            ),
        )
        for unit in rendered.units
    ]
    return RenderedWord(image=image, units=units, width=width, height=height)


def _labels_from_units(
    units: Sequence[UnitBox], x: int, y: int, image_size: int
) -> list[DatasetLabel]:
    labels: list[DatasetLabel] = []
    for unit in units:
        class_id = CLASS_TO_ID.get(unit.text)
        if class_id is None:
            continue
        box = _clip_box(
            (
                unit.bbox[0] + x,
                unit.bbox[1] + y,
                unit.bbox[2] + x,
                unit.bbox[3] + y,
            ),
            image_size,
            image_size,
        )
        if box is None:
            continue
        labels.append(DatasetLabel(class_id=class_id, text=unit.text, bbox=box))
    return labels


def _intersects(
    box: tuple[int, int, int, int], others: Sequence[tuple[int, int, int, int]]
) -> bool:
    x0, y0, x1, y1 = box
    return any(
        x0 < ox1 and x1 > ox0 and y0 < oy1 and y1 > oy0 for ox0, oy0, ox1, oy1 in others
    )


def _available_scripts(fonts: Sequence[FontInfo]) -> list[str]:
    scripts = [
        script
        for script in SCRIPT_NAMES
        if any(script in font.scripts for font in fonts)
    ]
    return scripts or list(SCRIPT_NAMES)


def _pick_font(
    fonts: Sequence[FontInfo], script: str, rng: random.Random
) -> FontInfo | None:
    pool = [font for font in fonts if script in font.scripts]
    return rng.choice(pool) if pool else None


def _sample_word(
    script: str, rng: random.Random, words: Mapping[str, Sequence[str]]
) -> str:
    pool = words.get(script)
    if pool and rng.random() < 0.7:
        word = rng.choice(list(pool))
        if word:
            return word
    if script == "arabic":
        return "".join(rng.choice(ARABIC_CLASSES) for _ in range(rng.randint(2, 6)))
    return "".join(rng.choice(ALPHANUMERIC_CLASSES) for _ in range(rng.randint(1, 10)))


def _make_collage(
    size: int,
    fonts: Sequence[FontInfo],
    rng: random.Random,
    np_rng: np.random.Generator,
    background_paths: Sequence[Path],
    words: Mapping[str, Sequence[str]],
    min_font_size: int,
    max_font_size: int,
    max_words: int,
) -> tuple[Image.Image, list[DatasetLabel]]:
    background = make_background(size, rng, np_rng, background_paths).convert("RGBA")
    labels: list[DatasetLabel] = []
    placed: list[tuple[int, int, int, int]] = []
    available_scripts = _available_scripts(fonts)
    target = rng.randint(1, max_words)
    for _ in range(target):
        for _attempt in range(30):
            script = rng.choice(available_scripts)
            font = _pick_font(fonts, script, rng)
            if font is None:
                continue
            font_size = rng.randint(min_font_size, max_font_size)
            text = _sample_word(script, rng, words)
            if not _font_supports_text(str(font.path), text):
                continue
            rendered = shape_word(text, font.path, font_size, script)
            if rendered is None or rendered.width >= size or rendered.height >= size:
                continue
            angle = rng.gauss(0.0, 1.5) if rng.random() < 0.3 else 0.0
            layer, units, width, height = _rotate_word(rendered, angle)
            if width >= size or height >= size:
                continue
            x = rng.randint(0, size - width)
            y = rng.randint(0, size - height)
            box = (x, y, x + width, y + height)
            if _intersects(box, placed):
                continue
            color = contrast_color(background, box, rng)
            background.alpha_composite(_colorize(layer, color), (x, y))
            labels.extend(_labels_from_units(units, x, y, size))
            placed.append(box)
            break
    return background.convert("RGB"), labels


def _make_line_layout(
    size: int,
    fonts: Sequence[FontInfo],
    rng: random.Random,
    np_rng: np.random.Generator,
    background_paths: Sequence[Path],
    words: Mapping[str, Sequence[str]],
    min_font_size: int,
    max_font_size: int,
    max_words: int,
    max_lines: int,
) -> tuple[Image.Image, list[DatasetLabel]]:
    background = make_background(size, rng, np_rng, background_paths).convert("RGBA")
    labels: list[DatasetLabel] = []
    margin = max(8, size // 40)
    line_count = rng.randint(min(2, max_lines), max_lines) if max_lines > 1 else 1
    available_scripts = _available_scripts(fonts)
    line_scripts = [
        available_scripts[index % len(available_scripts)] for index in range(line_count)
    ]
    rng.shuffle(line_scripts)
    available_height = size - 2 * margin
    for line_index, script in enumerate(line_scripts):
        line_top = margin + round(line_index * available_height / line_count)
        line_bottom = margin + round((line_index + 1) * available_height / line_count)
        slot_height = max(1, line_bottom - line_top)
        slot_font_max = min(max_font_size, max(min_font_size, int(slot_height * 0.70)))
        font = _pick_font(fonts, script, rng)
        if font is None:
            continue
        font_size = rng.randint(min_font_size, slot_font_max)
        rendered_words: list[RenderedWord] = []
        for _ in range(rng.randint(1, max_words)):
            text = _sample_word(script, rng, words)
            if not _font_supports_text(str(font.path), text):
                continue
            rendered = shape_word(text, font.path, font_size, script)
            if rendered is not None:
                rendered_words.append(rendered)
        if not rendered_words:
            continue

        gap = max(2, int(font_size * 0.25))
        available = size - 2 * margin
        while rendered_words:
            total_width = sum(word.width for word in rendered_words)
            total_width += gap * (len(rendered_words) - 1)
            if total_width <= available:
                break
            rendered_words.pop()
        if not rendered_words:
            continue
        total_width = sum(word.width for word in rendered_words)
        total_width += gap * (len(rendered_words) - 1)
        if total_width > available:
            factor = available / total_width
            if factor < 0.5:
                continue
            rendered_words = [_scale_word(word, factor) for word in rendered_words]
            gap = max(1, int(gap * factor))
            total_width = sum(word.width for word in rendered_words)
            total_width += gap * (len(rendered_words) - 1)

        line_height = max(word.height for word in rendered_words)
        if line_height > slot_height:
            factor = slot_height * 0.95 / line_height
            if factor < 0.5:
                continue
            rendered_words = [_scale_word(word, factor) for word in rendered_words]
            gap = max(1, int(gap * factor))
            line_height = max(word.height for word in rendered_words)
        y = line_top + rng.randint(0, max(0, slot_height - line_height))
        rtl = is_rtl_script(script)
        x = size - margin if rtl else margin
        for word in rendered_words:
            if rtl:
                x -= word.width
            px = int(round(x))
            py = int(round(y + (line_height - word.height) / 2.0))
            if px < 0 or px + word.width > size or py < 0:
                if rtl:
                    x -= gap
                else:
                    x += word.width + gap
                continue
            box = (px, py, px + word.width, py + word.height)
            color = contrast_color(background, box, rng)
            background.alpha_composite(_colorize(word.image, color), (px, py))
            labels.extend(_labels_from_units(word.units, px, py, size))
            if rtl:
                x -= gap
            else:
                x += word.width + gap
    return background.convert("RGB"), labels


def make_image(
    size: int,
    fonts: Sequence[FontInfo],
    rng: random.Random,
    np_rng: np.random.Generator,
    background_paths: Sequence[Path] = (),
    words: Mapping[str, Sequence[str]] | None = None,
    layout: str = "line",
    min_font_size: int = 18,
    max_font_size: int = 72,
    max_words: int = 6,
    max_lines: int = 4,
    empty_fraction: float = 0.0,
) -> tuple[Image.Image, list[DatasetLabel]]:
    """Generate one image and its pixel-coordinate labels."""
    if empty_fraction > 0.0 and rng.random() < empty_fraction:
        return (
            make_background(size, rng, np_rng, background_paths).convert("RGB"),
            [],
        )

    selected_layout = layout
    if layout == "mixed":
        selected_layout = rng.choice(["collage", "line"])
    word_lists = words or {}
    if selected_layout == "collage":
        image, labels = _make_collage(
            size,
            fonts,
            rng,
            np_rng,
            background_paths,
            word_lists,
            min_font_size,
            max_font_size,
            max_words,
        )
    else:
        image, labels = _make_line_layout(
            size,
            fonts,
            rng,
            np_rng,
            background_paths,
            word_lists,
            min_font_size,
            max_font_size,
            max_words,
            max_lines,
        )

    if rng.random() < 0.4:
        image = image.filter(ImageFilter.GaussianBlur(rng.uniform(0.2, 1.2)))
    if rng.random() < 0.5:
        array = np.asarray(image, dtype=np.float32)
        array += np_rng.normal(0.0, rng.uniform(2.0, 10.0), array.shape)
        image = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), mode="RGB")
    return image, labels


def _normalized_label(label: DatasetLabel, size: int) -> str:
    x0, y0, x1, y1 = label.bbox
    center_x = (x0 + x1) / 2.0 / size
    center_y = (y0 + y1) / 2.0 / size
    width = (x1 - x0) / size
    height = (y1 - y0) / size
    return f"{label.class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}"


def _write_preview(
    image: Image.Image,
    labels: Sequence[DatasetLabel],
    path: Path,
) -> None:
    preview = image.copy()
    draw = ImageDraw.Draw(preview)
    line_width = max(1, preview.width // 400)
    for label in labels:
        x0, y0, x1, y1 = label.bbox
        draw.rectangle((x0, y0, x1, y1), outline=(0, 220, 0), width=line_width)
        draw.text((x0 + 2, y0 + 2), str(label.class_id), fill=(255, 0, 0))
    preview.save(path, quality=90)


def _read_word_file(path: str | Path | None) -> list[str]:
    if path is None:
        return []
    return [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _split_word_lists(
    latin_path: str | Path | None,
    arabic_path: str | Path | None,
    mixed_path: str | Path | None,
) -> dict[str, list[str]]:
    latin = _read_word_file(latin_path) or list(DEFAULT_LATIN_WORDS)
    arabic = _read_word_file(arabic_path) or list(DEFAULT_ARABIC_WORDS)
    for word in _read_word_file(mixed_path):
        script = (
            "arabic" if any("\u0600" <= char <= "\u06ff" for char in word) else "latin"
        )
        if script == "arabic":
            arabic.append(word)
        else:
            latin.append(word)
    return {"latin": latin, "arabic": arabic}


def _split_counts(
    total: int, val_fraction: float, test_fraction: float, seed: int
) -> dict[str, int]:
    if total <= 0:
        return {"train": 0, "val": 0, "test": 0}
    n_val = max(1, round(total * val_fraction)) if val_fraction else 0
    n_test = max(1, round(total * test_fraction)) if test_fraction else 0
    while n_val + n_test >= total:
        if n_test > 0:
            n_test -= 1
        elif n_val > 0:
            n_val -= 1
        else:
            break
    n_train = total - n_val - n_test
    sequence = ["train"] * n_train + ["val"] * n_val + ["test"] * n_test
    random.Random(seed).shuffle(sequence)
    counts = {"train": 0, "val": 0, "test": 0}
    for split in sequence:
        counts[split] += 1
    return counts


def default_font_size_range(size: int) -> tuple[int, int]:
    """Return a font-size range that scales with the output image size."""
    return max(12, round(size * 0.035)), max(32, round(size * 0.12))


def generate_dataset(
    output_dir: str | Path,
    fonts: Sequence[FontInfo],
    n: int,
    size: int = 1280,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 42,
    background_paths: Sequence[Path] = (),
    words: Mapping[str, Sequence[str]] | None = None,
    layout: str = "line",
    min_font_size: int | None = None,
    max_font_size: int | None = None,
    max_words: int = 6,
    max_lines: int = 4,
    empty_fraction: float = 0.0,
    preview_count: int = 30,
) -> dict[str, Any]:
    """Generate a YOLO detection dataset and return its manifest."""
    if n <= 0:
        message = "n must be positive."
        raise ValueError(message)
    if size <= 0:
        message = "size must be positive."
        raise ValueError(message)
    if layout not in {"collage", "line", "mixed"}:
        message = f"Unknown layout: {layout!r}"
        raise ValueError(message)

    default_min, default_max = default_font_size_range(size)
    min_font_size = default_min if min_font_size is None else min_font_size
    max_font_size = default_max if max_font_size is None else max_font_size
    if min_font_size <= 0 or max_font_size < min_font_size:
        message = "Invalid font-size range."
        raise ValueError(message)

    split_fonts_map = split_fonts(fonts, val_fraction, test_fraction, seed)
    counts = _split_counts(n, val_fraction, test_fraction, seed)
    for split_name in ("val", "test"):
        if counts[split_name] and not split_fonts_map[split_name]:
            counts["train"] += counts[split_name]
            counts[split_name] = 0

    output = Path(output_dir)
    for split in ("train", "val", "test"):
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)
    (output / "previews").mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    sequence = (
        ["train"] * counts["train"]
        + ["val"] * counts["val"]
        + ["test"] * counts["test"]
    )
    random.Random(seed + 1).shuffle(sequence)
    word_lists = words or {}
    label_counts = {split: 0 for split in ("train", "val", "test")}
    image_counts = {split: 0 for split in ("train", "val", "test")}
    previews_written = 0

    for index, split in enumerate(sequence):
        split_fonts_list = split_fonts_map[split]
        if not split_fonts_list:
            split = "train"
            split_fonts_list = split_fonts_map["train"]
        image, labels = make_image(
            size=size,
            fonts=split_fonts_list,
            rng=rng,
            np_rng=np_rng,
            background_paths=background_paths,
            words=word_lists,
            layout=layout,
            min_font_size=min_font_size,
            max_font_size=max_font_size,
            max_words=max_words,
            max_lines=max_lines,
            empty_fraction=empty_fraction,
        )
        name = f"{index:06d}"
        image.save(
            output / "images" / split / f"{name}.jpg",
            quality=rng.randint(60, 95),
        )
        label_lines = [
            _normalized_label(label, size)
            for label in labels
            if label.bbox[2] > label.bbox[0] and label.bbox[3] > label.bbox[1]
        ]
        (output / "labels" / split / f"{name}.txt").write_text(
            "\n".join(label_lines) + ("\n" if label_lines else ""),
            encoding="utf-8",
        )
        image_counts[split] += 1
        label_counts[split] += len(label_lines)
        if previews_written < preview_count:
            _write_preview(image, labels, output / "previews" / f"{split}_{name}.jpg")
            previews_written += 1

    classes = list(DETECTION_CLASSES)
    data = {
        "path": str(output.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(classes),
        "names": {index: name for index, name in enumerate(classes)},
    }
    (output / "data.yaml").write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    (output / "classes.json").write_text(
        json.dumps(
            {
                "classes": classes,
                "class_to_id": CLASS_TO_ID,
                "arabic_ligatures": list(ARABIC_LIGATURES),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    manifest: dict[str, Any] = {
        "version": GENERATOR_VERSION,
        "seed": seed,
        "n": n,
        "size": size,
        "layout": layout,
        "val_fraction": val_fraction,
        "test_fraction": test_fraction,
        "min_font_size": min_font_size,
        "max_font_size": max_font_size,
        "max_words": max_words,
        "max_lines": max_lines,
        "empty_fraction": empty_fraction,
        "classes": classes,
        "image_counts": image_counts,
        "label_counts": label_counts,
        "splits": {
            split: [str(font.path) for font in split_fonts_map[split]]
            for split in ("train", "val", "test")
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def build_dataset(
    fonts_dir: str | Path,
    output_dir: str | Path,
    n: int,
    *,
    size: int = 1280,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 42,
    backgrounds_dir: str | Path | None = None,
    latin_words: str | Path | None = None,
    arabic_words: str | Path | None = None,
    mixed_words: str | Path | None = None,
    layout: str = "line",
    min_font_size: int | None = None,
    max_font_size: int | None = None,
    max_words: int = 6,
    max_lines: int = 4,
    empty_fraction: float = 0.0,
    preview_count: int = 30,
    max_fonts: int | None = None,
) -> dict[str, Any]:
    """Discover fonts, build word lists, and generate a dataset."""
    fonts = discover_fonts(fonts_dir)
    if max_fonts is not None:
        if max_fonts <= 0:
            message = "max_fonts must be positive."
            raise ValueError(message)
        fonts = fonts[:max_fonts]
    if not fonts:
        message = (
            f"No fonts covering the Latin or Arabic charset were found in "
            f"{fonts_dir}. Install fonts and re-run."
        )
        raise ValueError(message)
    background_paths = (
        sorted(
            path
            for path in Path(backgrounds_dir).rglob("*")
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        if backgrounds_dir is not None
        else []
    )
    words = _split_word_lists(latin_words, arabic_words, mixed_words)
    return generate_dataset(
        output_dir=output_dir,
        fonts=fonts,
        n=n,
        size=size,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        seed=seed,
        background_paths=background_paths,
        words=words,
        layout=layout,
        min_font_size=min_font_size,
        max_font_size=max_font_size,
        max_words=max_words,
        max_lines=max_lines,
        empty_fraction=empty_fraction,
        preview_count=preview_count,
    )
