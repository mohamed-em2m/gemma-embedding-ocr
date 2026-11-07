"""Smoke tests for the Typer CLI."""

from __future__ import annotations

import numpy as np
from PIL import Image
from typer.testing import CliRunner

from embedding_ocr import __version__
from embedding_ocr.cli import app

runner = CliRunner()


def test_version_command() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_show_config_command() -> None:
    result = runner.invoke(app, ["show-config", "--config", "configs/default.yaml"])
    assert result.exit_code == 0
    assert "embedder:" in result.stdout
    assert "recognition:" in result.stdout


def test_help_lists_core_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("recognize", "build-prototypes", "build-vocabulary", "benchmark"):
        assert command in result.stdout


def test_recognize_reports_unknown_backend(tmp_path) -> None:
    image_path = tmp_path / "sample.png"
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(image_path)
    result = runner.invoke(
        app,
        ["recognize", str(image_path), "--recognizer", "does-not-exist"],
    )
    assert result.exit_code == 1
    assert "Unknown recognizer" in result.output
