#!/usr/bin/env python3
"""Download a YOLO word-detection dataset from the Hugging Face Hub.

Two transfer methods are supported:

``hub`` (default)
    ``huggingface_hub.snapshot_download``. Resumable, parallel, and needs no
    ``git-lfs``. Requires the ``hf`` extra.

``git``
    A plain ``git clone`` of the dataset repo. **Requires git-lfs**, otherwise
    every image arrives as a ~130-byte pointer file. The script detects that
    case and fails instead of letting you train on corrupt data.

Example::

    uv run --no-sync python scripts/download_hf_dataset.py \\
        --repo-id mohamedemam/words-detection --dest artifacts/words-detection
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

from rich.console import Console
from rich.table import Table

_CONSOLE = Console()
_LFS_PREFIX = b"version https://git-lfs.github.com/spec/v1"
_SPLITS = ("train", "val", "test")


def _looks_like_lfs_pointer(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            return handle.read(len(_LFS_PREFIX)) == _LFS_PREFIX
    except OSError:
        return False


def _validate(root: Path) -> tuple[dict[str, int], list[str]]:
    problems: list[str] = []
    counts: dict[str, int] = {}
    for name in ("data.yaml", "README.md"):
        if not (root / name).is_file():
            problems.append(f"missing {name}")
    for split in _SPLITS:
        image_dir = root / "images" / split
        label_dir = root / "labels" / split
        if not image_dir.is_dir():
            continue
        images = [
            p for p in image_dir.iterdir() if p.is_file() and not p.name.startswith(".")
        ]
        labels = (
            [p for p in label_dir.iterdir() if p.suffix == ".txt"]
            if label_dir.is_dir()
            else []
        )
        counts[split] = len(images)
        if not images:
            problems.append(f"no images in images/{split}")
            continue
        if len(labels) < len(images):
            problems.append(
                f"labels/{split} has {len(labels)} files for {len(images)} images"
            )
        sample = next((p for p in images if p.stat().st_size > 0), None)
        if sample is not None and _looks_like_lfs_pointer(sample):
            problems.append(
                f"{sample.name} is a git-lfs pointer, not an image "
                "(install git-lfs or use --method hub)"
            )
    if not counts:
        problems.append("no images/train, images/val or images/test directories")
    return counts, problems


def _download_hub(repo_id: str, dest: Path, revision: str, token: str | None) -> None:
    try:
        from huggingface_hub import snapshot_download  # type: ignore[import-untyped]
    except ImportError:
        raise SystemExit(
            "error: huggingface_hub is not installed. "
            "Run `uv pip install --python .venv/bin/python -e '.[hf]'` "
            "or use --method git."
        ) from None
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        local_dir=str(dest),
        token=token,
        ignore_patterns=["*.cache", "__pycache__/*"],
    )


def _download_git(repo_id: str, dest: Path, revision: str) -> None:
    if shutil.which("git") is None:
        raise SystemExit("error: git is not installed; use --method hub")
    if shutil.which("git-lfs") is None:
        raise SystemExit(
            "error: git-lfs is not installed. A plain `git clone` would download "
            "LFS pointer files instead of images.\n"
            "Install it (`apt-get install git-lfs && git lfs install`) or use "
            "--method hub."
        )
    url = f"https://huggingface.co/datasets/{repo_id}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        shutil.rmtree(dest)
    subprocess.run(
        ["git", "clone", "--depth", "1", "--branch", revision, url, str(dest)],
        check=True,
    )
    subprocess.run(["git", "lfs", "pull"], cwd=dest, check=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download a YOLO dataset from the Hugging Face Hub."
    )
    parser.add_argument("--repo-id", default="mohamedemam/words-detection")
    parser.add_argument("--dest", type=Path, default=Path("artifacts/words-detection"))
    parser.add_argument("--revision", default="main")
    parser.add_argument(
        "--method",
        choices=("auto", "hub", "git"),
        default="auto",
        help="auto uses the Hub downloader and falls back to git clone.",
    )
    parser.add_argument("--token", default=None, help="Only needed for private repos.")
    parser.add_argument(
        "--force", action="store_true", help="Re-download even if dest looks valid."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dest: Path = args.dest

    if dest.exists() and not args.force:
        counts, problems = _validate(dest)
        if not problems:
            _CONSOLE.print(
                f"[green]{dest} already looks complete, skipping download[/]"
            )
            _report(counts)
            return 0
        _CONSOLE.print(f"[yellow]{dest} exists but is incomplete: {problems[0]}[/]")

    if args.force and dest.exists():
        shutil.rmtree(dest)

    method = args.method
    if method in ("auto", "hub"):
        try:
            _CONSOLE.print(
                f"downloading [bold]{args.repo_id}[/] via the Hub client → {dest}"
            )
            _download_hub(args.repo_id, dest, args.revision, args.token)
        except (SystemExit, Exception) as error:  # noqa: BLE001
            if method == "hub":
                raise
            _CONSOLE.print(
                f"[yellow]hub download failed ({error}); falling back to git clone[/]"
            )
            _download_git(args.repo_id, dest, args.revision)
    else:
        _CONSOLE.print(f"cloning [bold]{args.repo_id}[/] with git-lfs → {dest}")
        _download_git(args.repo_id, dest, args.revision)

    counts, problems = _validate(dest)
    if problems:
        _CONSOLE.print("[red]dataset validation failed:[/]")
        for problem in problems:
            _CONSOLE.print(f"  - {problem}")
        return 1

    _report(counts)
    _CONSOLE.print(
        f"\ntrain with:\n  uv run --no-sync python scripts/train_yolo_words.py "
        f"--data {dest / 'data.yaml'}"
    )
    return 0


def _report(counts: dict[str, int]) -> None:
    table = Table(title="dataset")
    table.add_column("split", style="bold")
    table.add_column("images", justify="right")
    for split, count in counts.items():
        table.add_row(split, str(count))
    table.add_row("[bold]total[/]", f"[bold]{sum(counts.values())}[/]")
    _CONSOLE.print(table)


if __name__ == "__main__":
    raise SystemExit(main())
