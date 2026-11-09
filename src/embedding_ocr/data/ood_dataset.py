"""Deterministic out-of-distribution OCR dataset generation.

The dataset is intentionally rendered from fonts that are disjoint from the
prototype fonts used by EmbeddingOCR.  It contains Latin and Arabic glyphs,
words and phrases, with clean and degraded rendering conditions, so both GLM
OCR and our embedding recognizers are evaluated on identical crops.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from embedding_ocr.recognition.embedding.hierarchical.vocabulary import (
    render_prototype_image,
)

OOD_DATASET_VERSION = "ood_latin_arabic_v1"
OOD_CONDITIONS: tuple[str, ...] = ("clean", "mild", "hard")

OOD_LATIN_WORDS: tuple[str, ...] = (
    "fjord",
    "sphinx",
    "nymph",
    "rhythm",
    "syzygy",
    "quixotic",
    "jambalaya",
    "xylophone",
    "mnemonic",
    "labyrinth",
    "kayak",
    "crypt",
    "glyph",
    "zephyr",
    "braille",
    "typeset",
)

OOD_LATIN_PHRASES: tuple[str, ...] = (
    "Q3 revenue: $1,248.50",
    "HTTP/2 & TLS 1.3",
    "e-mail: a@b.co",
    "R&D budget - 2027",
    "x-height & cap-height",
    "C++ / Python 3.13",
    "ID: A1B2-C3D4",
    "50% faster, 2x smaller",
)

OOD_LATIN_GLYPHS: tuple[str, ...] = tuple("O0Il1S5Z2B8G6q9rpda")

OOD_ARABIC_WORDS: tuple[str, ...] = (
    "مكتبة",
    "الحاسوب",
    "الكتاب",
    "المدرسة",
    "الجامعة",
    "الطالب",
    "البرنامج",
    "الشبكة",
    "البيانات",
    "الخوارزمية",
    "المعالجة",
    "الترجمة",
    "الفاتورة",
    "الإجمالي",
    "العميل",
    "الخدمة",
)

OOD_ARABIC_PHRASES: tuple[str, ...] = (
    "السلام عليكم",
    "معالجة النصوص العربية",
    "الذكاء الاصطناعي",
    "التعلم الآلي",
    "رؤية الحاسوب",
    "الفاتورة الإجمالي 2026",
    "رقم الحساب 42",
    "شبكة عصبية عميقة",
)

OOD_ARABIC_DIACRITICS: tuple[str, ...] = (
    "مُحَمَّد",
    "اللُّغَةُ",
    "الذَّكَاءُ",
    "الْعَرَبِيَّةُ",
    "الْإِصْطِنَاعِيُّ",
)

OOD_ARABIC_LIGATURES: tuple[str, ...] = ("لا", "لأ", "لإ", "لآ")

OOD_ARABIC_GLYPHS: tuple[str, ...] = tuple("ابتثجحخدذرزسشصضطظعغفقكلمنهوي")

OOD_CORPORA: dict[str, dict[str, tuple[str, ...]]] = {
    "latin": {
        "word": OOD_LATIN_WORDS,
        "phrase": OOD_LATIN_PHRASES,
        "glyph": OOD_LATIN_GLYPHS,
    },
    "arabic": {
        "word": OOD_ARABIC_WORDS,
        "phrase": OOD_ARABIC_PHRASES,
        "diacritics": OOD_ARABIC_DIACRITICS,
        "ligature": OOD_ARABIC_LIGATURES,
        "glyph": OOD_ARABIC_GLYPHS,
    },
}

OOD_INDEX_FONTS: dict[str, tuple[str, ...]] = {
    "latin": (
        "/usr/share/fonts/liberation-sans-fonts/LiberationSans-Regular.ttf",
        "/usr/share/fonts/liberation-serif-fonts/LiberationSerif-Regular.ttf",
        "/usr/share/fonts/open-sans/OpenSans-Regular.ttf",
        "/usr/share/fonts/google-carlito-fonts/Carlito-Regular.ttf",
        "/usr/share/fonts/google-crosextra-caladea-fonts/Caladea-Regular.ttf",
    ),
    "arabic": (
        "/usr/share/fonts/google-noto-vf/NotoNaskhArabic[wght].ttf",
        "/usr/share/fonts/vazirmatn-vf-fonts/Vazirmatn[wght].ttf",
        "/usr/share/fonts/paktype-naskh-basic-fonts/PakTypeNaskhBasic.ttf",
    ),
}

OOD_TEST_FONTS: dict[str, tuple[str, ...]] = {
    "latin": (
        "/usr/share/fonts/julietaula-montserrat-fonts/Montserrat-Regular.otf",
        "/usr/share/fonts/aajohan-comfortaa-fonts/Comfortaa-Regular.otf",
        "/usr/share/fonts/urw-base35/URWGothic-Book.otf",
        "/usr/share/fonts/urw-base35/NimbusSansNarrow-Regular.otf",
        "/usr/share/fonts/abattis-cantarell-fonts/Cantarell-Regular.otf",
        "/usr/share/fonts/adobe-source-code-pro-fonts/SourceCodePro-Regular.otf",
        "/usr/share/fonts/adwaita-sans-fonts/AdwaitaSans-Regular.ttf",
        "/usr/share/fonts/stix-fonts/STIXTwoText-Regular.otf",
    ),
    "arabic": (
        "/usr/share/fonts/google-noto-vf/NotoSansArabic[wght].ttf",
        "/usr/share/fonts/google-droid-sans-fonts/DroidKufi-Regular.ttf",
    ),
}

DEFAULT_CONDITIONS: tuple[str, ...] = OOD_CONDITIONS

FONT_SEARCH_DIRS: tuple[Path, ...] = (
    Path("/usr/share/fonts"),
    Path("/usr/local/share/fonts"),
    Path.home() / ".fonts",
    Path.home() / ".local/share/fonts",
)
ARABIC_FONT_HINTS: tuple[str, ...] = (
    "arab",
    "naskh",
    "kufi",
    "vazir",
    "amiri",
    "scheherazade",
    "paktype",
    "droidkufi",
    "droidarabic",
)
NON_TEXT_FONT_HINTS: tuple[str, ...] = (
    "emoji",
    "symbol",
    "icon",
    "awesome",
    "material",
    "fa-solid",
    "fa-regular",
    "fontawesome",
)
FONT_SUFFIXES = {".ttf", ".otf"}


@dataclass(frozen=True)
class OODSample:
    """One image/reference pair from an OOD manifest."""

    sample_id: str
    image_path: Path
    text: str
    script: str
    category: str
    font: str
    condition: str

    @property
    def font_name(self) -> str:
        return Path(self.font).stem


def _existing_fonts(fonts: Sequence[str]) -> list[str]:
    return [font for font in fonts if Path(font).exists()]


def _classify_font(path: Path) -> str | None:
    name = path.name.lower()
    if any(hint in name for hint in NON_TEXT_FONT_HINTS):
        return None
    if any(hint in name for hint in ARABIC_FONT_HINTS):
        return "arabic"
    return "latin"


def discover_fonts(script: str, limit: int = 4) -> list[str]:
    """Find installed fonts for a script when configured paths are absent."""
    found: list[str] = []
    seen: set[str] = set()
    for root in FONT_SEARCH_DIRS:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if path.suffix.lower() not in FONT_SUFFIXES or not path.is_file():
                continue
            if _classify_font(path) != script:
                continue
            key = str(path)
            if key in seen:
                continue
            seen.add(key)
            found.append(key)
            if len(found) >= limit:
                return found
    return found


def resolve_font_split(
    index_fonts: dict[str, Sequence[str]] | None = None,
    test_fonts: dict[str, Sequence[str]] | None = None,
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Resolve configured fonts and fall back to installed fonts if needed."""
    configured_index = index_fonts or OOD_INDEX_FONTS
    configured_test = test_fonts or OOD_TEST_FONTS
    resolved_index: dict[str, list[str]] = {}
    resolved_test: dict[str, list[str]] = {}

    for script in ("latin", "arabic"):
        index = _existing_fonts(configured_index.get(script, ()))
        test = _existing_fonts(configured_test.get(script, ()))
        overlap = set(index) & set(test)
        if overlap:
            msg = f"index/test fonts overlap for {script!r}: {sorted(overlap)}"
            raise ValueError(msg)

        available = [
            font
            for font in discover_fonts(script)
            if font not in set(index) | set(test)
        ]
        if not index and available:
            take = max(1, len(available) // 2)
            index, available = available[:take], available[take:]
        if not test and available:
            test = available

        if not index:
            msg = f"no index fonts available for script {script!r}"
            raise ValueError(msg)
        if not test:
            msg = f"no held-out test fonts available for script {script!r}"
            raise ValueError(msg)
        resolved_index[script] = index
        resolved_test[script] = test

    return resolved_index, resolved_test


def _condition_image(
    image: np.ndarray, condition: str, rng: np.random.Generator
) -> np.ndarray:
    """Apply a deterministic degradation of the requested severity."""
    if condition == "clean" or condition not in DEFAULT_CONDITIONS:
        return image

    pil = Image.fromarray(image).convert("RGB")

    if condition == "mild":
        if rng.random() < 0.55:
            pil = pil.filter(ImageFilter.GaussianBlur(float(rng.uniform(0.3, 0.8))))
        if rng.random() < 0.55:
            pil = ImageEnhance.Contrast(pil).enhance(float(rng.uniform(0.8, 1.2)))
        if rng.random() < 0.35:
            angle = float(rng.uniform(-2.0, 2.0))
            pil = pil.rotate(
                angle, resample=Image.Resampling.BILINEAR, fillcolor=(255, 255, 255)
            )
        array = np.asarray(pil, dtype=np.float32)
        if rng.random() < 0.6:
            noise = rng.normal(0.0, float(rng.uniform(3.0, 7.0)), array.shape)
            array = np.clip(array + noise, 0, 255)
        return array.astype(np.uint8)

    # Hard: combine resolution loss, JPEG artifacts, perspective, noise and
    # contrast changes.  This is still intended to be human-readable.
    height, width = image.shape[:2]
    scale = float(rng.uniform(0.42, 0.68))
    small = cv2.resize(
        image,
        (max(8, int(width * scale)), max(8, int(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    image = cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)

    quality = int(rng.integers(18, 38))
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if ok:
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if decoded is not None:
            image = np.asarray(decoded, dtype=np.uint8)

    if rng.random() < 0.7:
        source: np.ndarray = np.array(
            [[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32
        )
        jitter: np.ndarray = rng.uniform(-0.035, 0.035, size=(4, 2)).astype(np.float32)
        destination: np.ndarray = source + jitter * np.array(
            [width, height], dtype=np.float32
        )
        matrix = cv2.getPerspectiveTransform(source, destination)
        image = np.asarray(
            cv2.warpPerspective(
                image,
                matrix,
                (width, height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(255, 255, 255),
            ),
            dtype=np.uint8,
        )

    pil = Image.fromarray(image).convert("RGB")
    if rng.random() < 0.7:
        pil = pil.filter(ImageFilter.GaussianBlur(float(rng.uniform(0.5, 1.2))))
    pil = ImageEnhance.Contrast(pil).enhance(float(rng.uniform(0.65, 1.35)))
    array = np.asarray(pil, dtype=np.float32)
    noise = rng.normal(0.0, float(rng.uniform(5.0, 12.0)), array.shape)
    array = np.clip(array + noise, 0, 255)
    return array.astype(np.uint8)


def _sample_id(
    script: str, category: str, font: str, condition: str, index: int
) -> str:
    digest = hashlib.sha1(
        f"{script}|{category}|{font}|{condition}|{index}".encode()
    ).hexdigest()[:10]
    return f"{script}_{category}_{Path(font).stem}_{condition}_{digest}"


def build_ood_dataset(
    out_dir: str | Path,
    *,
    seed: int = 20261108,
    conditions: Sequence[str] = DEFAULT_CONDITIONS,
    counts: dict[str, dict[str, int]] | None = None,
    index_fonts: dict[str, Sequence[str]] | None = None,
    test_fonts: dict[str, Sequence[str]] | None = None,
    font_size: int = 44,
    overwrite: bool = False,
) -> Path:
    """Render the OOD dataset and write ``manifest.jsonl``.

    Args:
        out_dir: Output directory. Images are written to ``images/``.
        seed: Deterministic seed for degradation choices.
        conditions: Degradation conditions to generate.
        counts: Per-script/per-category sample counts. Defaults to all corpus
            entries for ``word``/``phrase``/``glyph`` and capped extras.
        index_fonts: Fonts used to build prototypes. Must be disjoint from
            ``test_fonts``.
        test_fonts: Fonts used to render the evaluation crops.
        font_size: Rendering size in points.
        overwrite: Replace an existing manifest/images directory.

    Returns:
        Path to the written manifest.
    """
    out = Path(out_dir)
    manifest_path = out / "manifest.jsonl"
    if manifest_path.exists() and not overwrite:
        msg = f"{manifest_path} already exists; pass overwrite=True to regenerate"
        raise FileExistsError(msg)

    invalid = sorted(set(conditions) - set(DEFAULT_CONDITIONS))
    if invalid:
        msg = f"unknown conditions: {invalid}; choose from {DEFAULT_CONDITIONS}"
        raise ValueError(msg)

    index_font_map, test_font_map = resolve_font_split(index_fonts, test_fonts)

    default_counts = {
        "latin": {"word": 16, "phrase": 8, "glyph": 16},
        "arabic": {
            "word": 16,
            "phrase": 8,
            "diacritics": 5,
            "ligature": 4,
            "glyph": 28,
        },
    }
    count_map = counts or default_counts
    images_dir = out / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    index = 0
    for script in ("latin", "arabic"):
        corpora = OOD_CORPORA[script]
        script_counts = count_map.get(script, default_counts[script])
        for category, corpus in corpora.items():
            requested = int(script_counts.get(category, len(corpus)))
            texts = list(corpus)[: max(0, requested)]
            for text in texts:
                for font_path in test_font_map[script]:
                    for condition in conditions:
                        rng = np.random.default_rng(seed + index)
                        image = render_prototype_image(
                            text,
                            font_path,
                            font_size=font_size,
                            pad=16,
                            script=script,
                        )
                        image = _condition_image(image, condition, rng)
                        sample_id = _sample_id(
                            script, category, font_path, condition, index
                        )
                        image_path = images_dir / f"{sample_id}.png"
                        Image.fromarray(image).save(image_path)
                        records.append(
                            {
                                "id": sample_id,
                                "image": str(image_path.relative_to(out)),
                                "text": text,
                                "script": script,
                                "category": category,
                                "font": font_path,
                                "condition": condition,
                            }
                        )
                        index += 1

    with manifest_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    meta = {
        "version": OOD_DATASET_VERSION,
        "seed": seed,
        "font_size": font_size,
        "conditions": list(conditions),
        "index_fonts": index_font_map,
        "test_fonts": test_font_map,
        "counts": count_map,
        "num_samples": len(records),
    }
    (out / "dataset.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest_path


def load_ood_dataset(manifest_path: str | Path) -> list[OODSample]:
    """Load an OOD manifest, resolving image paths relative to its directory."""
    manifest = Path(manifest_path)
    base = manifest.parent
    samples: list[OODSample] = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        image_path = Path(record["image"])
        if not image_path.is_absolute():
            image_path = base / image_path
        samples.append(
            OODSample(
                sample_id=str(record["id"]),
                image_path=image_path,
                text=str(record["text"]),
                script=str(record["script"]),
                category=str(record["category"]),
                font=str(record["font"]),
                condition=str(record["condition"]),
            )
        )
    if not samples:
        msg = f"manifest contains no samples: {manifest}"
        raise ValueError(msg)
    return samples
