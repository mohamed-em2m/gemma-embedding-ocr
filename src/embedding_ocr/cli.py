"""Typer CLI for EmbeddingOCR."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import typer
import yaml
from PIL import Image

from embedding_ocr import __version__
from embedding_ocr.config import AppConfig, load_config, save_config
from embedding_ocr.eval.benchmark import read_manifest, run_benchmark, write_report
from embedding_ocr.pipeline.pipeline import OCRPipeline
from embedding_ocr.recognition.factory import (
    build_embedder,
    build_pipeline,
    build_prototypes,
    build_vocabulary,
)
from embedding_ocr.utils.io import git_hash

app = typer.Typer(add_completion=False, help="EmbeddingOCR command line interface.")


def _apply_overrides(
    config: AppConfig,
    device: str | None = None,
    recognizer: str | None = None,
    detector: str | None = None,
    script: str | None = None,
    dim: int | None = None,
) -> AppConfig:
    if device is not None:
        config.embedder.device = device
    if recognizer is not None:
        config.recognition.backend = recognizer
    if detector is not None:
        config.detector.backend = detector
    if script is not None:
        config.recognition.script = script
    if dim is not None:
        config.embedder.dim = dim
        config.index.dim = dim
    return config


def _build_pipeline_or_exit(config: AppConfig) -> OCRPipeline:
    try:
        return build_pipeline(config)
    except (ImportError, FileNotFoundError, ValueError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc


def _read_image(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(__version__)


@app.command()
def ui(
    config_dir: Path = typer.Option(
        Path("configs"), "--config-dir", help="Directory containing YAML profiles."
    ),
    host: str = typer.Option("127.0.0.1", help="Server bind address."),
    port: int = typer.Option(7860, help="Server port."),
    share: bool = typer.Option(False, help="Create a public Gradio share link."),
) -> None:
    """Launch the Gradio web interface."""
    try:
        from embedding_ocr.webui import launch_ui
    except ImportError as exc:
        typer.echo(
            "error: Gradio is not installed; run `uv sync --extra ui`.",
            err=True,
        )
        raise typer.Exit(code=1) from exc
    launch_ui(config_dir=config_dir, host=host, port=port, share=share)


@app.command("show-config")
def show_config(
    config: Path = typer.Option(Path("configs/default.yaml"), "--config", "-c"),
) -> None:
    """Print the resolved configuration as YAML."""
    resolved = load_config(config)
    typer.echo(yaml.safe_dump(resolved.model_dump(mode="json"), sort_keys=False))


@app.command()
def recognize(
    image: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    config: Path = typer.Option(Path("configs/default.yaml"), "--config", "-c"),
    device: str | None = typer.Option(None, help="Override embedder device."),
    recognizer: str | None = typer.Option(None, help="Override recognizer backend."),
    detector: str | None = typer.Option(None, help="Override detector backend."),
    script: str | None = typer.Option(None, help="Override script."),
    as_json: bool = typer.Option(False, "--json", help="Emit structured JSON."),
) -> None:
    """Recognize text in a single image."""
    resolved = _apply_overrides(
        load_config(config),
        device=device,
        recognizer=recognizer,
        detector=detector,
        script=script,
    )
    pipeline = _build_pipeline_or_exit(resolved)
    result = pipeline.recognize(_read_image(image))
    if as_json:
        payload = dataclasses.asdict(result)
        payload["metadata"] = {
            **result.metadata,
            "git_hash": git_hash(),
            "config": resolved.model_dump(mode="json"),
        }
        typer.echo(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        typer.echo(result.text)


@app.command("build-prototypes")
def build_prototypes_command(
    config: Path = typer.Option(Path("configs/default.yaml"), "--config", "-c"),
    output: Path = typer.Option(Path("artifacts/prototypes/v1"), "--output", "-o"),
    fonts: list[str] | None = typer.Option(
        None, "--font", help="Override prototype fonts."
    ),
    font_catalog: Path | None = typer.Option(
        None, help="Optional text file with one font path per line."
    ),
    num_fonts: int | None = typer.Option(None, help="Limit fonts from catalog."),
    script: str | None = typer.Option(None, help="Override script."),
    dim: int | None = typer.Option(None, help="Override embedding dimension."),
    device: str | None = typer.Option(None, help="Override embedder device."),
) -> None:
    """Build and persist a character prototype database."""
    resolved = _apply_overrides(
        load_config(config), device=device, script=script, dim=dim
    )
    font_list = list(fonts) if fonts else []
    if font_catalog is not None:
        catalog = [
            line.strip()
            for line in font_catalog.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]
        font_list.extend(catalog)
    if num_fonts is not None:
        font_list = font_list[:num_fonts]
    db = build_prototypes(resolved, fonts=font_list or None)
    db.save(output)
    typer.echo(
        f"Saved {len(db.labels)} prototypes ({len(db.classes)} classes, "
        f"{len(db.fonts)} fonts) to {output}"
    )


@app.command("build-vocabulary")
def build_vocabulary_command(
    config: Path = typer.Option(Path("configs/default.yaml"), "--config", "-c"),
    device: str | None = typer.Option(None, help="Override embedder device."),
    script: str | None = typer.Option(None, help="Override script."),
    dim: int | None = typer.Option(None, help="Override embedding dimension."),
) -> None:
    """Build hierarchical word/subword/char vocabulary indexes."""
    resolved = _apply_overrides(
        load_config(config), device=device, script=script, dim=dim
    )
    embedder = build_embedder(resolved.embedder)
    vocabulary = build_vocabulary(resolved, embedder)
    typer.echo(
        f"Built vocabulary: {len(vocabulary.word_tokens)} words, "
        f"{len(vocabulary.subword_tokens)} subwords, "
        f"{len(vocabulary.char_tokens)} chars"
    )


@app.command()
def benchmark(
    manifest: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    config: Path = typer.Option(Path("configs/default.yaml"), "--config", "-c"),
    output: Path = typer.Option(Path("docs/benchmark_report.json"), "--output", "-o"),
    device: str | None = typer.Option(None, help="Override embedder device."),
    recognizer: str | None = typer.Option(None, help="Override recognizer backend."),
    detector: str | None = typer.Option(None, help="Override detector backend."),
    script: str | None = typer.Option(None, help="Override script."),
) -> None:
    """Run a manifest-driven CER/WER benchmark."""
    resolved = _apply_overrides(
        load_config(config),
        device=device,
        recognizer=recognizer,
        detector=detector,
        script=script,
    )
    pipeline = _build_pipeline_or_exit(resolved)
    samples = read_manifest(manifest)
    report = run_benchmark(pipeline, samples)
    report["metadata"] = {
        "git_hash": git_hash(),
        "config": resolved.model_dump(mode="json"),
    }
    write_report(report, output)
    save_config(resolved, output.with_suffix(".config.yaml"))
    metrics = report["metrics"]
    typer.echo(json.dumps(metrics, indent=2))
    typer.echo(f"Report written to {output}")


if __name__ == "__main__":
    app()
