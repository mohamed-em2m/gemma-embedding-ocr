"""Google Fonts discovery and family-disjoint splits for fine-tuning."""

from __future__ import annotations

import hashlib
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

FONT_SUFFIXES = {".ttf", ".otf"}


@dataclass(frozen=True)
class FontFamily:
    """One font family and its available font files."""

    name: str
    paths: tuple[Path, ...]

    @property
    def primary(self) -> Path:
        """Return the regular-looking file, falling back to the first file."""
        for path in self.paths:
            stem = path.stem.lower()
            if any(tag in stem for tag in ("regular", "roman", "book", "normal")):
                return path
        return self.paths[0]


def discover_font_families(root: str | Path) -> list[FontFamily]:
    """Discover font families under a Google Fonts checkout or font directory.

    Google Fonts stores files as ``ofl/<family>/<Font>.ttf``. For flat
    directories, the family is inferred from the file name before the first
    dash or underscore.
    """
    root_path = Path(root)
    if not root_path.exists():
        msg = f"Font directory not found: {root_path}"
        raise FileNotFoundError(msg)

    grouped: dict[str, list[Path]] = {}
    for path in sorted(root_path.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in FONT_SUFFIXES:
            continue
        family = _family_name(path)
        grouped.setdefault(family, []).append(path)

    return [
        FontFamily(name=name, paths=tuple(sorted(paths)))
        for name, paths in sorted(grouped.items())
    ]


def split_font_families(
    families: Sequence[FontFamily],
    *,
    holdout_fraction: float = 0.15,
    seed: int = 42,
) -> tuple[list[FontFamily], list[FontFamily]]:
    """Split families into train and held-out sets without leakage."""
    if not families:
        msg = "No usable font families found."
        raise ValueError(msg)
    if not 0.0 < holdout_fraction < 1.0:
        msg = f"holdout_fraction must be in (0, 1), got {holdout_fraction}"
        raise ValueError(msg)
    if len(families) < 2:
        msg = "At least two font families are required for a held-out split."
        raise ValueError(msg)

    ordered = sorted(families, key=lambda family: family.name)
    rng = random.Random(seed)
    rng.shuffle(ordered)
    holdout_count = max(1, round(len(ordered) * holdout_fraction))
    holdout_count = min(holdout_count, len(ordered) - 1)
    heldout = ordered[:holdout_count]
    train = ordered[holdout_count:]
    return (
        sorted(train, key=lambda family: family.name),
        sorted(heldout, key=lambda family: family.name),
    )


def family_id(name: str) -> str:
    """Return a short stable identifier for cache keys and metadata."""
    return hashlib.sha256(name.encode("utf-8")).hexdigest()[:12]


def iter_font_paths(families: Iterable[FontFamily]) -> list[Path]:
    """Flatten families into a sorted list of font file paths."""
    return [path for family in families for path in family.paths]


def _family_name(path: Path) -> str:
    parent = path.parent.name
    if parent.lower() in {"ofl", "apache", "ufl"}:
        return _clean_stem(path.stem)
    if parent and parent not in {".", "/"}:
        return parent
    return _clean_stem(path.stem)


def _clean_stem(stem: str) -> str:
    for separator in ("-", "_", "["):
        if separator in stem:
            stem = stem.split(separator, 1)[0]
    return stem.strip() or "unknown"
