#!/usr/bin/env python3
"""Upload a YOLO dataset folder to the Hugging Face Hub.

Run ``--dry-run`` first: it validates the layout and prints exactly what would
be uploaded (file counts and sizes per directory) without touching the network.

Requires the ``hf`` extra: ``uv pip install --python .venv/bin/python -e '.[hf]'``
and an ``HF_TOKEN`` with write access.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

_CONSOLE = Console()

_IGNORE = [
    "*.cache",
    "*.log",
    "*.tmp",
    "__pycache__/*",
    ".ipynb_checkpoints/*",
    ".DS_Store",
]

_REQUIRED = ("data.yaml", "README.md")


def _summarize(folder: Path) -> tuple[Table, int, int]:
    table = Table(title=f"upload plan — {folder}")
    table.add_column("directory", style="bold")
    table.add_column("files", justify="right")
    table.add_column("size", justify="right")

    total_files = 0
    total_bytes = 0
    groups: dict[str, list[Path]] = {}
    for path in sorted(folder.rglob("*")):
        if not path.is_file() or path.name.endswith(".cache"):
            continue
        relative = path.relative_to(folder)
        key = str(relative.parent) if str(relative.parent) != "." else "(root)"
        groups.setdefault(key, []).append(path)

    for key, files in groups.items():
        size = sum(f.stat().st_size for f in files)
        total_files += len(files)
        total_bytes += size
        table.add_row(key, str(len(files)), f"{size / 1e6:,.1f} MB")
    table.add_row(
        "[bold]total[/]",
        f"[bold]{total_files}[/]",
        f"[bold]{total_bytes / 1e6:,.1f} MB[/]",
    )
    return table, total_files, total_bytes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Upload a YOLO dataset to the Hugging Face Hub."
    )
    parser.add_argument(
        "--folder", type=Path, default=Path("artifacts/yolo_words_real")
    )
    parser.add_argument("--repo-id", default="elemam/gemma-ocr-words-real")
    parser.add_argument(
        "--card",
        type=Path,
        default=Path("docs/datasets/yolo_words_real.md"),
        help="Canonical dataset card, copied to <folder>/README.md before upload.",
    )
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--commit-message", default="Upload YOLO word dataset v1.0.0")
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the folder and print the upload plan without uploading.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not args.folder.is_dir():
        print(f"error: folder not found: {args.folder}", file=sys.stderr)
        return 2
    missing = [name for name in _REQUIRED if not (args.folder / name).is_file()]
    if missing:
        print(f"error: missing required file(s): {', '.join(missing)}", file=sys.stderr)
        return 2

    if args.card is not None and args.card.is_file():
        target = args.folder / "README.md"
        canonical = args.card.read_text(encoding="utf-8")
        if not target.is_file() or target.read_text(encoding="utf-8") != canonical:
            if args.dry_run:
                _CONSOLE.print(
                    f"[yellow]card differs from {args.card}; "
                    "a real upload would refresh README.md[/]"
                )
            else:
                target.write_text(canonical, encoding="utf-8")
                _CONSOLE.print(f"refreshed {target} from {args.card}")

    table, total_files, total_bytes = _summarize(args.folder)
    _CONSOLE.print(table)
    _CONSOLE.print(
        f"repo: [bold]{args.repo_id}[/] "
        f"({'private' if args.private else 'public'}, dataset)"
    )
    _CONSOLE.print(f"ignored patterns: {', '.join(_IGNORE)}")

    if args.dry_run:
        _CONSOLE.print("[yellow]dry run: nothing was uploaded[/]")
        return 0

    try:
        from huggingface_hub import HfApi  # type: ignore[import-untyped]
    except ImportError:
        print(
            "error: huggingface_hub is not installed. "
            "Run `uv pip install --python .venv/bin/python -e '.[hf]'`.",
            file=sys.stderr,
        )
        return 2

    if not args.token:
        print("error: no token. Set HF_TOKEN or pass --token.", file=sys.stderr)
        return 2
    if total_bytes > 2_000_000_000:
        _CONSOLE.print(
            "[yellow]warning: folder exceeds 2 GB; prefer `huggingface-cli upload` "
            "with Xet/LFS for resumable transfers[/]"
        )

    api = HfApi(token=args.token)
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        private=args.private,
        exist_ok=True,
    )
    _CONSOLE.print(f"created/verified dataset repo {args.repo_id}")

    api.upload_folder(
        repo_id=args.repo_id,
        repo_type="dataset",
        folder_path=str(args.folder),
        commit_message=args.commit_message,
        ignore_patterns=_IGNORE,
        max_workers=args.max_workers,
    )
    _CONSOLE.print(f"[green]uploaded {total_files} files to {args.repo_id}[/]")
    _CONSOLE.print(f"https://huggingface.co/datasets/{args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
