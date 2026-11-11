#!/usr/bin/env python
"""Download the Google Fonts OFL collection with a sparse git checkout."""

from __future__ import annotations

import argparse
import logging
import subprocess
from pathlib import Path

from embedding_ocr.training.fonts import discover_font_families

LOGGER = logging.getLogger("download_google_fonts")
REPO_URL = "https://github.com/google/fonts.git"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/google_fonts"),
        help="Destination for the sparse Google Fonts checkout.",
    )
    parser.add_argument(
        "--repo",
        default=REPO_URL,
        help="Google Fonts git URL.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reuse an existing checkout and refresh the sparse subset.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    destination = args.output.expanduser().resolve()

    if not (destination / ".git").exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        command = [
            "git",
            "clone",
            "--depth",
            "1",
            "--filter=blob:none",
            "--sparse",
            args.repo,
            str(destination),
        ]
        LOGGER.info("Cloning %s", args.repo)
        subprocess.run(command, check=True)
    elif not args.force:
        LOGGER.info("Reusing existing checkout at %s", destination)
    else:
        LOGGER.info("Refreshing sparse checkout at %s", destination)
        subprocess.run(
            ["git", "-C", str(destination), "pull", "--ff-only"],
            check=True,
        )

    subprocess.run(
        ["git", "-C", str(destination), "sparse-checkout", "set", "ofl"],
        check=True,
    )
    families = discover_font_families(destination / "ofl")
    files = sum(len(family.paths) for family in families)
    if not families:
        msg = f"No OFL font families found under {destination / 'ofl'}"
        raise RuntimeError(msg)
    LOGGER.info("Found %d families / %d font files", len(families), files)
    LOGGER.info("Licenses are kept in each family's OFL.txt")


if __name__ == "__main__":
    main()
