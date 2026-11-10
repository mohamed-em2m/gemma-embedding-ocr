"""Download Roboflow Universe datasets described in a source config.

The Roboflow SDK needs a free API key (``ROBOFLOW_API_KEY`` or ``--api-key``).
Each entry in the config must have a ``roboflow`` block::

    sources:
      - name: devanagari_words
        path: artifacts/real_sources/devanagari-word-detection
        roboflow:
          workspace: aaa-8kwq1
          project: devanagari-word-detection
          version: 1          # verify on the Universe page
          format: yolov8      # use 'folder' for classification projects

Usage:
    export ROBOFLOW_API_KEY=...
    uv run --no-sync python scripts/download_roboflow_sources.py \
        --config configs/real_word_sources.yaml

Then build the merged dataset::

    uv run --no-sync python scripts/build_real_word_dataset.py \
        --config configs/real_word_sources.yaml \
        --out artifacts/yolo_words_real
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download Roboflow Universe datasets from a source config."
    )
    parser.add_argument(
        "--config", default=Path("configs/real_word_sources.yaml"), type=Path
    )
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        help="Download only these source names (repeatable).",
    )
    return parser


def _load_entries(path: Path) -> list[Mapping[str, Any]]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, Mapping):
        message = f"Invalid config (expected a mapping): {path}"
        raise ValueError(message)
    entries = loaded.get("sources", [])
    if not isinstance(entries, list):
        message = f"Invalid 'sources' list in {path}"
        raise ValueError(message)
    return [entry for entry in entries if isinstance(entry, Mapping)]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.config.is_file():
        print(f"error: config not found: {args.config}", file=sys.stderr)
        return 2

    api_key = args.api_key or os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        print(
            "error: set ROBOFLOW_API_KEY or pass --api-key.\n"
            "Create a free key at https://app.roboflow.com/settings/api",
            file=sys.stderr,
        )
        return 2

    try:
        from roboflow import Roboflow  # type: ignore[import-untyped]
    except ImportError:
        print(
            "error: the roboflow package is not installed.\n"
            "Install it with: uv pip install --python .venv/bin/python roboflow\n"
            "or add the 'roboflow' extra.",
            file=sys.stderr,
        )
        return 2

    try:
        entries = _load_entries(args.config)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    only = set(args.only)
    rf = Roboflow(api_key=api_key)
    downloaded = 0
    for entry in entries:
        name = str(entry.get("name") or entry.get("path") or "dataset")
        if only and name not in only:
            continue
        roboflow_cfg = entry.get("roboflow")
        if not isinstance(roboflow_cfg, Mapping):
            print(f"skip {name}: no 'roboflow' block")
            continue
        try:
            workspace = str(roboflow_cfg["workspace"])
            project = str(roboflow_cfg["project"])
            version = int(roboflow_cfg["version"])
        except (KeyError, TypeError, ValueError):
            print(
                f"error: {name} needs roboflow.workspace, project and version",
                file=sys.stderr,
            )
            return 2
        model_format = str(roboflow_cfg.get("format", "yolov8"))
        location = Path(str(entry["path"])).expanduser().resolve()
        if location.exists() and any(location.iterdir()):
            if not args.overwrite:
                print(f"skip {name}: {location} is not empty")
                continue
        elif location.exists():
            # The Roboflow SDK treats any existing path as "already downloaded",
            # even when it is empty. Remove the placeholder directory first.
            location.rmdir()
        location.parent.mkdir(parents=True, exist_ok=True)
        print(
            f"downloading {name}: {workspace}/{project} v{version} "
            f"({model_format}) -> {location}"
        )
        project_obj = rf.workspace(workspace).project(project)
        project_obj.version(version).download(
            model_format,
            location=str(location),
            overwrite=args.overwrite,
        )
        downloaded += 1

    print(f"Downloaded {downloaded} dataset(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
