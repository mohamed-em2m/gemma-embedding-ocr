"""Adapters for real OCR word crops used in contrastive fine-tuning."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from embedding_ocr.training.short_texts import TextSample


@dataclass(frozen=True)
class RealCrop:
    """A real word crop with its ground-truth text."""

    image_path: Path
    text: str
    source: str = "real"

    @property
    def text_id(self) -> str:
        return hashlib.sha1(self.text.encode("utf-8")).hexdigest()


def load_real_crops(path: str | Path) -> list[RealCrop]:
    """Load a JSONL manifest with ``image`` and ``text`` fields."""
    crops: list[RealCrop] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            image = payload.get("image", payload.get("image_path"))
            text = payload.get("text")
            if not isinstance(image, str) or not isinstance(text, str):
                continue
            sample = TextSample.from_text(
                text, source=str(payload.get("source", "real"))
            )
            if sample is None:
                continue
            image_path = Path(image)
            if not image_path.is_absolute():
                image_path = Path(path).parent / image_path
            if not image_path.exists():
                continue
            crops.append(
                RealCrop(
                    image_path=image_path,
                    text=sample.text,
                    source=str(payload.get("source", "real")),
                )
            )
    return crops


def save_real_crops(
    crops: Iterable[tuple[Image.Image, str]],
    output_dir: str | Path,
    *,
    source: str = "real",
    prefix: str = "crop",
) -> Path:
    """Save PIL crops and return a JSONL manifest path."""
    root = Path(output_dir).expanduser().resolve()
    image_dir = root / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    manifest = root / f"{prefix}.jsonl"
    with manifest.open("w", encoding="utf-8") as handle:
        for index, (image, text) in enumerate(crops):
            sample = TextSample.from_text(text, source=source)
            if sample is None:
                continue
            image_path = image_dir / f"{prefix}_{index:07d}.png"
            image.convert("RGB").save(image_path)
            handle.write(
                json.dumps(
                    {
                        "image": str(image_path),
                        "text": sample.text,
                        "source": source,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return manifest
