#!/usr/bin/env python
"""Prepare line-level OCR datasets for the CTC recognizer.

Downloads line-image datasets from the Hugging Face Hub, writes PNG crops,
normalizes the labels, builds the character vocabulary from the train split
only, and records everything in ``manifest.jsonl`` + ``stats.json``.

Examples::

    # Full preparation (needs the 'finetune' extra for `datasets`)
    uv run --no-sync python scripts/prepare_line_datasets.py

    # Quick smoke run: 200 rows per split, no writes
    uv run --no-sync python scripts/prepare_line_datasets.py \
        --limit-per-split 200 --dry-run

    # Inspect an unknown local export (e.g. the Kaggle KHATT download)
    uv run --no-sync python scripts/prepare_line_datasets.py \
        --inspect-local artifacts/raw/khatt_kaggle

    # Download and inspect a Kaggle dataset in one step
    uv run --no-sync python scripts/prepare_line_datasets.py \
        --kaggle-download nizarcharrada/khattarabic \
        --kaggle-dest artifacts/raw/khatt_kaggle
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
from pathlib import Path
from typing import Any

import yaml

from embedding_ocr.training.line_dataset import (
    FilterConfig,
    LineSource,
    inspect_local_dataset,
    prepare_line_datasets,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
LOGGER = logging.getLogger("prepare_line_datasets")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--config", type=Path, default=_REPO_ROOT / "configs/line_datasets.yaml"
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--sources",
        nargs="*",
        default=None,
        help="Only prepare these source names (default: every source in the config).",
    )
    parser.add_argument("--max-per-source", type=int, default=None)
    parser.add_argument("--limit-per-split", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-diacritics", action="store_true")
    parser.add_argument("--no-alef-folding", action="store_true")
    parser.add_argument(
        "--inspect-local",
        type=Path,
        default=None,
        help="Print the tree and label-file samples of a local export, then exit.",
    )
    parser.add_argument(
        "--kaggle-download",
        default=None,
        help="Kaggle dataset id to download before inspecting, e.g. owner/slug.",
    )
    parser.add_argument("--kaggle-dest", type=Path, default=None)
    return parser.parse_args()


def _load_config(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        message = f"invalid dataset config (expected a mapping): {path}"
        raise ValueError(message)
    return loaded


def _sources(config: dict[str, Any], selected: list[str] | None) -> list[LineSource]:
    entries = config.get("sources") or []
    if not isinstance(entries, list) or not entries:
        message = "dataset config has no 'sources' list"
        raise ValueError(message)
    wanted = set(selected) if selected else None
    sources: list[LineSource] = []
    for entry in entries:
        if not isinstance(entry, dict) or "name" not in entry or "hf_id" not in entry:
            continue
        name = str(entry["name"])
        if wanted is not None and name not in wanted:
            continue
        splits = entry.get("splits")
        sources.append(
            LineSource(
                name=name,
                hf_id=str(entry["hf_id"]),
                script=str(entry["script"]) if entry.get("script") else None,
                splits=tuple(str(item) for item in splits)
                if isinstance(splits, list)
                else ("train", "val", "test"),
            )
        )
    if not sources:
        message = "no matching sources to prepare"
        raise ValueError(message)
    return sources


def _kaggle_download(dataset_id: str, destination: Path) -> Path:
    destination = destination.expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    command = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        dataset_id,
        "-p",
        str(destination),
        "--unzip",
    ]
    LOGGER.info("running: %s", " ".join(command))
    try:
        subprocess.run(command, check=True)  # noqa: S603
    except FileNotFoundError as exc:
        message = (
            "the kaggle CLI is not installed; install it and place credentials "
            "in ~/.kaggle/kaggle.json"
        )
        raise RuntimeError(message) from exc
    except subprocess.CalledProcessError as exc:
        message = (
            f"kaggle download failed for {dataset_id!r}. Check that "
            "~/.kaggle/kaggle.json exists and that you accepted the dataset terms."
        )
        raise RuntimeError(message) from exc
    return destination


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()

    if args.kaggle_download:
        if args.kaggle_dest is None:
            message = "--kaggle-dest is required together with --kaggle-download"
            raise SystemExit(message)
        _kaggle_download(args.kaggle_download, args.kaggle_dest)
        print(
            json.dumps(
                inspect_local_dataset(args.kaggle_dest), ensure_ascii=False, indent=2
            )
        )
        return

    if args.inspect_local is not None:
        print(
            json.dumps(
                inspect_local_dataset(args.inspect_local),
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    config = _load_config(args.config)
    filters = FilterConfig(**dict(config.get("filters") or {}))
    normalization = dict(config.get("normalization") or {})
    output = args.output or Path(str(config.get("output", "artifacts/lines")))
    if not output.is_absolute():
        output = _REPO_ROOT / output

    stats = prepare_line_datasets(
        _sources(config, args.sources),
        output,
        filters=filters,
        max_per_source=args.max_per_source,
        limit_per_split=args.limit_per_split,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
        strip_diacritics=(
            not args.keep_diacritics
            and bool(normalization.get("strip_diacritics", True))
        ),
        fold_alefs=(
            not args.no_alef_folding and bool(normalization.get("fold_alefs", True))
        ),
        log=LOGGER.info,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
