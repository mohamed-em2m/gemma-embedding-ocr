"""Run metadata and filesystem helpers."""

from __future__ import annotations

import subprocess
from pathlib import Path


def git_hash(repo: Path | None = None) -> str | None:
    """Return the current git commit hash, or ``None`` outside a repository."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            capture_output=True,
            check=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None
