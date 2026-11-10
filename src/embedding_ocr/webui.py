"""Gradio web interface for the EmbeddingOCR pipeline.

The module deliberately imports Gradio lazily so the CLI and tests keep working
when the optional ``ui`` extra is not installed.
"""

from __future__ import annotations

import argparse
import gc
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from embedding_ocr.config import AppConfig, load_config
from embedding_ocr.pipeline.pipeline import OCRPipeline
from embedding_ocr.pipeline.types import OCRResult
from embedding_ocr.recognition.factory import build_pipeline

DEFAULT_CONFIG_DIR = Path("configs")
DEFAULT_PROFILE = "default.yaml"
DEFAULT_DEVICE = "auto"
DEFAULT_RECOGNIZER = "config default"
DEFAULT_DETECTOR = "config default"
DEFAULT_SCRIPT = "config default"
RECOGNIZER_CHOICES = (
    DEFAULT_RECOGNIZER,
    "hierarchical",
    "embedding",
    "yolo_units",
    "paddle",
    "tesseract",
    "easyocr",
)
DETECTOR_CHOICES = (
    DEFAULT_DETECTOR,
    "auto",
    "paddle",
    "craft",
    "yolo_words",
    "lines",
    "visual",
    "full",
)
SCRIPT_CHOICES = (DEFAULT_SCRIPT, "latin", "arabic")
DEVICE_CHOICES = ("auto", "cpu", "cuda")

MODE_OCR = "OCR (detect + read)"
MODE_WORDS = "Words only (YOLO)"
MODE_CHOICES = (MODE_OCR, MODE_WORDS)
YOLO_IMGSZ_CHOICES = ("640", "960", "1280", "1600")
DEFAULT_YOLO_CONF = 0.25
DEFAULT_YOLO_IMGSZ = "960"

_PIPELINE_CACHE: dict[tuple[Any, ...], OCRPipeline] = {}
_CACHE_LOCK = threading.Lock()
_YOLO_CACHE: dict[tuple[str, float], Any] = {}


@dataclass(frozen=True)
class UiOverrides:
    """Runtime choices exposed by the web UI."""

    device: str = DEFAULT_DEVICE
    recognizer: str = DEFAULT_RECOGNIZER
    detector: str = DEFAULT_DETECTOR
    script: str = DEFAULT_SCRIPT
    quick: bool = True
    yolo_model: str | None = None
    yolo_conf: float = DEFAULT_YOLO_CONF
    yolo_imgsz: int = int(DEFAULT_YOLO_IMGSZ)


def list_example_images() -> list[str]:
    """Return bundled example images that can be loaded into the UI."""
    candidates = (
        Path.cwd() / "assets" / "examples",
        Path(__file__).resolve().parents[2] / "assets" / "examples",
    )
    for directory in candidates:
        if directory.is_dir():
            images = sorted(
                path for path in directory.iterdir() if path.suffix.lower() == ".png"
            )
            if images:
                return [str(path) for path in images]
    return []


def example_rows(profiles: list[str]) -> list[list[Any]]:
    """Return Gradio example rows with the matching profile and script."""
    default_profile = DEFAULT_PROFILE if DEFAULT_PROFILE in profiles else profiles[0]
    arabic_profile = (
        "embedding_align.yaml"
        if "embedding_align.yaml" in profiles
        else default_profile
    )
    rows: list[list[Any]] = []
    for path in list_example_images():
        is_arabic = Path(path).name.lower().startswith("arabic")
        rows.append(
            [
                path,
                arabic_profile if is_arabic else default_profile,
                DEFAULT_DEVICE,
                "hierarchical" if is_arabic else DEFAULT_RECOGNIZER,
                DEFAULT_DETECTOR,
                "arabic" if is_arabic else "latin",
                True,
            ]
        )
    return rows


def list_config_profiles(config_dir: str | Path = DEFAULT_CONFIG_DIR) -> list[str]:
    """Return available YAML profile names, sorted."""
    directory = Path(config_dir)
    if not directory.exists():
        return []
    return sorted(path.name for path in directory.glob("*.yaml") if path.is_file())


def resolve_config_path(
    profile: str | None, config_dir: str | Path = DEFAULT_CONFIG_DIR
) -> Path:
    """Resolve a profile name while preventing path traversal."""
    directory = Path(config_dir).resolve()
    profiles = list_config_profiles(directory)
    if not profiles:
        msg = f"no YAML profiles found in {directory}"
        raise FileNotFoundError(msg)

    name = profile or (DEFAULT_PROFILE if DEFAULT_PROFILE in profiles else profiles[0])
    if Path(name).name != name:
        msg = f"invalid profile name: {name!r}"
        raise ValueError(msg)
    if name not in profiles:
        msg = f"unknown profile {name!r}; choose from {profiles}"
        raise ValueError(msg)
    return directory / name


def list_yolo_checkpoints(root: str | Path = "artifacts") -> list[str]:
    """Return trained word-detection checkpoints, newest first.

    Only ``best.pt``/``last.pt`` files under the artifacts tree are listed, so
    generic COCO checkpoints (which do not detect words) are not offered.
    """
    base = Path(root)
    if not base.is_dir():
        return []
    found: dict[Path, float] = {}
    for pattern in ("**/weights/best.pt", "**/weights/last.pt", "**/best.pt"):
        for path in base.glob(pattern):
            if path.is_file():
                try:
                    found[path.resolve()] = path.stat().st_mtime
                except OSError:
                    continue
    ordered = sorted(found.items(), key=lambda item: item[1], reverse=True)
    return [str(path) for path, _ in ordered]


def _clean_choice(value: str | None) -> str | None:
    if value is None or value in {
        "",
        DEFAULT_RECOGNIZER,
        DEFAULT_DETECTOR,
        DEFAULT_SCRIPT,
    }:
        return None
    return value


def apply_ui_overrides(config: AppConfig, overrides: UiOverrides) -> AppConfig:
    """Apply validated UI choices to a loaded config in place."""
    config.embedder.device = overrides.device

    recognizer = _clean_choice(overrides.recognizer)
    if recognizer is not None:
        config.recognition.backend = recognizer

    detector = _clean_choice(overrides.detector)
    if detector is not None:
        config.detector.backend = detector

    if overrides.yolo_model:
        # Picking a checkpoint implies the YOLO detector backend.
        config.detector.backend = "yolo_words"
        config.detector.yolo_model = Path(overrides.yolo_model).expanduser()
        config.detector.yolo_conf = float(overrides.yolo_conf)
        config.detector.yolo_imgsz = int(overrides.yolo_imgsz)

    script = _clean_choice(overrides.script)
    if script is not None:
        config.recognition.script = script
        config.recognition.rtl = None
        config.detector.rtl = None

    if overrides.quick and config.recognition.backend.lower() == "hierarchical":
        vocabulary = config.recognition.vocabulary
        vocabulary.max_words = min(vocabulary.max_words or 4000, 4000)
        vocabulary.max_subwords = min(vocabulary.max_subwords or 3000, 3000)

    return config


def _cache_key(config_path: Path, overrides: UiOverrides) -> tuple[Any, ...]:
    stat = config_path.stat()
    return (
        str(config_path),
        stat.st_mtime_ns,
        overrides.device,
        overrides.recognizer,
        overrides.detector,
        overrides.script,
        overrides.quick,
        overrides.yolo_model,
        overrides.yolo_conf,
        overrides.yolo_imgsz,
    )


def _is_cuda_oom(exc: BaseException) -> bool:
    """Return True when an exception is a CUDA out-of-memory error."""
    return type(exc).__name__ == "OutOfMemoryError" and "CUDA" in str(exc)


def _release_cuda_memory() -> None:
    """Run Python/CUDA garbage collection after dropping a pipeline."""
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _release_pipelines(pipelines: list[OCRPipeline]) -> None:
    for pipeline in pipelines:
        del pipeline
    pipelines.clear()
    _release_cuda_memory()


def get_pipeline(
    profile: str | None,
    overrides: UiOverrides,
    config_dir: str | Path = DEFAULT_CONFIG_DIR,
) -> OCRPipeline:
    """Build or fetch the one cached pipeline for the selected profile."""
    config_path = resolve_config_path(profile, config_dir)
    key = _cache_key(config_path, overrides)
    with _CACHE_LOCK:
        cached = _PIPELINE_CACHE.get(key)
        if cached is not None:
            return cached
        stale = list(_PIPELINE_CACHE.values())
        _PIPELINE_CACHE.clear()
    # Free the previous model(s) before loading another one onto the GPU.
    _release_pipelines(stale)
    config = apply_ui_overrides(load_config(config_path), overrides)
    pipeline = build_pipeline(config)
    with _CACHE_LOCK:
        _PIPELINE_CACHE[key] = pipeline
    return pipeline


def clear_pipeline_cache() -> None:
    """Drop cached pipelines and release their GPU memory."""
    with _CACHE_LOCK:
        pipelines = list(_PIPELINE_CACHE.values())
        _PIPELINE_CACHE.clear()
    _release_pipelines(pipelines)


def get_yolo_detector(
    model_path: str | Path,
    conf: float = DEFAULT_YOLO_CONF,
    imgsz: int = int(DEFAULT_YOLO_IMGSZ),
    device: str | None = None,
) -> Any:
    """Load (and cache) a YOLO word detector for the given checkpoint."""
    from embedding_ocr.pipeline.detect import YoloWordDetector

    path = Path(str(model_path)).expanduser()
    if not path.is_file():
        msg = f"YOLO checkpoint not found: {path}"
        raise FileNotFoundError(msg)
    key = (str(path.resolve()), path.stat().st_mtime)
    with _CACHE_LOCK:
        detector = _YOLO_CACHE.get(key)
        if detector is None:
            detector = YoloWordDetector(
                model_path=path,
                conf=float(conf),
                iou=0.5,
                imgsz=int(imgsz),
                device=device or None,
                max_det=1000,
            )
            _YOLO_CACHE[key] = detector
    # Cheap attributes: keep the cached model but honour the current controls.
    detector.conf = float(conf)
    detector.imgsz = int(imgsz)
    detector.device = device or None
    return detector


def clear_yolo_cache() -> None:
    """Drop cached YOLO detectors and release their GPU memory."""
    with _CACHE_LOCK:
        _YOLO_CACHE.clear()
    _release_cuda_memory()


def _to_rgb(image: Any) -> np.ndarray:
    if image is None:
        msg = "no image supplied"
        raise ValueError(msg)
    array = np.asarray(image)
    if array.size == 0:
        msg = "image is empty"
        raise ValueError(msg)
    if array.ndim == 2:
        return np.repeat(array[:, :, None], 3, axis=2).astype(np.uint8, copy=False)
    if array.ndim != 3 or array.shape[2] not in (3, 4):
        msg = f"expected HxWx3 or HxWx4 image, got shape {array.shape}"
        raise ValueError(msg)
    return np.ascontiguousarray(array[:, :, :3]).astype(np.uint8, copy=False)


def annotate_word_boxes(image: np.ndarray, result: OCRResult) -> np.ndarray:
    """Draw recognizer word boxes over the input image for debugging."""
    canvas = Image.fromarray(_to_rgb(image)).convert("RGB")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default()
    except OSError:  # pragma: no cover - Pillow always ships a default font
        font = None

    raw_boxes = result.metadata.get("word_boxes")
    if not isinstance(raw_boxes, list):
        return np.asarray(canvas, dtype=np.uint8)

    color = (37, 99, 235)
    for index, item in enumerate(raw_boxes, start=1):
        if not isinstance(item, dict):
            continue
        bbox = item.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        try:
            x, y, width, height = (int(value) for value in bbox)
        except (TypeError, ValueError):
            continue
        x1 = x + max(1, width)
        y1 = y + max(1, height)
        draw.rectangle((x, y, x1, y1), outline=color, width=2)
        text = str(item.get("text", "")).strip() or "?"
        label = f"{index}: {text}"
        if font is not None:
            draw.text((x + 2, max(0, y - 12)), label, fill=color, font=font)
        else:  # pragma: no cover
            draw.text((x + 2, max(0, y - 12)), label, fill=color)

    return np.asarray(canvas, dtype=np.uint8)


def annotate_result(image: np.ndarray, result: OCRResult) -> np.ndarray:
    """Draw detected region boxes and labels over the input image."""
    canvas = Image.fromarray(_to_rgb(image)).convert("RGB")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default()
    except OSError:  # pragma: no cover - Pillow always ships a default font
        font = None

    for index, region in enumerate(result.regions, start=1):
        x, y, width, height = region.region.bbox
        x1 = x + max(1, width)
        y1 = y + max(1, height)
        color = (16, 185, 129)
        draw.rectangle((x, y, x1, y1), outline=color, width=2)
        label = f"{index}: {region.text}"
        if font is not None:
            draw.text((x + 2, max(0, y - 12)), label, fill=color, font=font)
        else:  # pragma: no cover
            draw.text((x + 2, max(0, y - 12)), label, fill=color)

    return np.asarray(canvas, dtype=np.uint8)


def annotate_word_regions(image: np.ndarray, regions: list[Any]) -> np.ndarray:
    """Draw detected word boxes with their confidence over the image."""
    canvas = Image.fromarray(_to_rgb(image)).convert("RGB")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default()
    except OSError:  # pragma: no cover - Pillow always ships a default font
        font = None

    color = (220, 38, 38)
    for index, region in enumerate(regions, start=1):
        x, y, width, height = (int(value) for value in region.bbox)
        draw.rectangle(
            (x, y, x + max(1, width), y + max(1, height)), outline=color, width=2
        )
        label = f"{index}: {float(region.confidence):.2f}"
        if font is not None:
            draw.text((x + 2, max(0, y - 12)), label, fill=color, font=font)
        else:  # pragma: no cover
            draw.text((x + 2, max(0, y - 12)), label, fill=color)

    return np.asarray(canvas, dtype=np.uint8)


def run_word_detection(
    image: Any,
    model_path: str,
    conf: float = DEFAULT_YOLO_CONF,
    imgsz: int = int(DEFAULT_YOLO_IMGSZ),
    device: str | None = None,
    progress: Any = None,
) -> tuple[
    str,
    float,
    dict[str, Any],
    np.ndarray | None,
    np.ndarray | None,
    str,
]:
    """Run a YOLO checkpoint on its own and return word boxes only."""
    try:
        array = _to_rgb(image)
        _notify(progress, 0.2, "Loading YOLO word detector...")
        detector = get_yolo_detector(model_path, conf=conf, imgsz=imgsz, device=device)
        _notify(progress, 0.6, "Detecting words...")
        started = time.perf_counter()
        regions = detector.detect(array)
        latency_ms = (time.perf_counter() - started) * 1000.0
        annotated = annotate_word_regions(array, regions)
        scores = [float(region.confidence) for region in regions]
        mean_conf = sum(scores) / len(scores) if scores else 0.0
        details = {
            "mode": MODE_WORDS,
            "checkpoint": str(model_path),
            "conf": float(conf),
            "imgsz": int(imgsz),
            "device": device or "auto",
            "word_count": len(regions),
            "latency_ms": round(latency_ms, 2),
            "words": [
                {
                    "index": index,
                    "bbox": list(region.bbox),
                    "confidence": round(float(region.confidence), 4),
                }
                for index, region in enumerate(regions, start=1)
            ],
        }
        status = (
            "### Done\n"
            f"- Mode: `{MODE_WORDS}`\n"
            f"- Checkpoint: `{Path(str(model_path)).name}`\n"
            f"- Words: {len(regions)}\n"
            f"- Mean confidence: {mean_conf:.3f}\n"
            f"- Latency: {latency_ms:.1f} ms"
        )
        return (
            f"{len(regions)} words detected",
            mean_conf,
            details,
            annotated,
            annotated,
            status,
        )
    except Exception as exc:  # noqa: BLE001 - surface the error in the UI
        if _is_cuda_oom(exc):
            clear_yolo_cache()
        else:
            _release_cuda_memory()
        error = f"{type(exc).__name__}: {exc}"
        details = {"mode": MODE_WORDS, "checkpoint": str(model_path), "error": error}
        fallback = _to_rgb(image) if image is not None else None
        return "", 0.0, details, fallback, fallback, f"### Error\n`{error}`"


def format_details(
    profile: str | None,
    overrides: UiOverrides,
    result: OCRResult,
) -> dict[str, Any]:
    """Build the JSON payload shown in the details panel."""
    return {
        "profile": profile,
        "overrides": {
            "device": overrides.device,
            "recognizer": overrides.recognizer,
            "detector": overrides.detector,
            "script": overrides.script,
            "quick": overrides.quick,
            "yolo_model": overrides.yolo_model,
            "yolo_conf": overrides.yolo_conf,
            "yolo_imgsz": overrides.yolo_imgsz,
        },
        "text": result.text,
        "confidence": result.confidence,
        "metadata": result.metadata,
        "regions": [
            {
                "bbox": list(region.region.bbox),
                "text": region.text,
                "confidence": region.confidence,
            }
            for region in result.regions
        ],
    }


def _notify(progress: Any, value: float, description: str) -> None:
    if progress is None:
        return
    try:
        progress(value, desc=description)
    except TypeError:  # pragma: no cover - older Gradio signatures
        progress(value)


def run_ocr(
    image: Any,
    profile: str | None,
    device: str,
    recognizer: str,
    detector: str,
    script: str,
    quick: bool,
    progress: Any = None,
    *,
    mode: str = MODE_OCR,
    yolo_model: str | None = None,
    yolo_conf: float = DEFAULT_YOLO_CONF,
    yolo_imgsz: int = int(DEFAULT_YOLO_IMGSZ),
    yolo_device: str | None = None,
) -> tuple[
    str,
    float,
    dict[str, Any],
    np.ndarray | None,
    np.ndarray | None,
    str,
]:
    """Run the selected pipeline and return Gradio-compatible outputs."""
    if mode == MODE_WORDS:
        if not yolo_model:
            message = "### Error\n`no YOLO checkpoint selected`"
            return (
                "",
                0.0,
                {"mode": MODE_WORDS, "error": "no checkpoint"},
                None,
                None,
                message,
            )
        return run_word_detection(
            image,
            yolo_model,
            conf=yolo_conf,
            imgsz=yolo_imgsz,
            device=yolo_device or device,
            progress=progress,
        )

    overrides = UiOverrides(
        device=device,
        recognizer=recognizer,
        detector=detector,
        script=script,
        quick=quick,
        yolo_model=yolo_model or None,
        yolo_conf=yolo_conf,
        yolo_imgsz=yolo_imgsz,
    )
    try:
        array = _to_rgb(image)
        _notify(progress, 0.1, "Loading pipeline (first run may download models)...")
        pipeline = get_pipeline(profile, overrides)
        _notify(progress, 0.55, "Running detection and recognition...")
        result = pipeline.recognize(array)
        annotated = annotate_result(array, result)
        word_debug = annotate_word_boxes(array, result)
        details = format_details(profile, overrides, result)
        latency = result.metadata.get("latency_ms")
        latency_text = (
            f"{float(latency):.1f} ms" if isinstance(latency, (int, float)) else "n/a"
        )
        raw_word_boxes = result.metadata.get("word_boxes")
        word_box_count = len(raw_word_boxes) if isinstance(raw_word_boxes, list) else 0
        status = (
            "### Done\n"
            f"- Profile: `{profile or 'default'}`\n"
            f"- Recognizer: `{result.metadata.get('recognizer', 'unknown')}`\n"
            f"- Detector: `{result.metadata.get('detector', 'unknown')}`\n"
            f"- Regions: {len(result.regions)}\n"
            f"- Word boxes: {word_box_count}\n"
            f"- Latency: {latency_text}"
        )
        return (
            result.text,
            float(result.confidence),
            details,
            annotated,
            word_debug,
            status,
        )
    except Exception as exc:  # noqa: BLE001 - surface the error in the UI
        if _is_cuda_oom(exc):
            clear_pipeline_cache()
        else:
            _release_cuda_memory()
        error = f"{type(exc).__name__}: {exc}"
        details = {
            "profile": profile,
            "overrides": asdict(overrides),
            "error": error,
        }
        fallback = _to_rgb(image) if image is not None else None
        return "", 0.0, details, fallback, fallback, f"### Error\n`{error}`"


def _clear() -> tuple[None, str, float, None, None, dict[str, Any], str]:
    return None, "", 0.0, None, None, {}, ""


def build_demo(
    config_dir: str | Path = DEFAULT_CONFIG_DIR,
    default_checkpoint: str | None = None,
) -> Any:
    """Create the Gradio Blocks application."""
    try:
        import gradio as gr
    except ImportError as exc:  # pragma: no cover - optional dependency
        msg = (
            "Gradio is not installed. Install the UI extra with "
            "`uv pip install --python .venv/bin/python -e '.[ui]'` or `uv pip install gradio`."
        )
        raise ImportError(msg) from exc

    profiles = list_config_profiles(config_dir)
    if not profiles:
        msg = f"no YAML profiles found in {config_dir}"
        raise FileNotFoundError(msg)
    default_profile = DEFAULT_PROFILE if DEFAULT_PROFILE in profiles else profiles[0]

    with gr.Blocks(title="EmbeddingOCR") as demo:
        gr.Markdown(
            "# EmbeddingOCR\n"
            "Upload an image, choose a profile, and run visual-embedding OCR. "
            "The first run can take a while because models and vocabulary "
            "indexes may need to be built.\n\n"
            "Switch **Mode** to *Words only (YOLO)* to run a trained `.pt` "
            "checkpoint on its own and get word boxes without recognition."
        )
        with gr.Row():
            with gr.Column(scale=1):
                image = gr.Image(type="numpy", label="Input image")
                mode = gr.Radio(
                    choices=list(MODE_CHOICES),
                    value=MODE_OCR,
                    label="Mode",
                )
                profile = gr.Dropdown(
                    choices=profiles,
                    value=default_profile,
                    label="Config profile",
                )
                with gr.Accordion("Overrides", open=False):
                    device = gr.Dropdown(
                        choices=list(DEVICE_CHOICES),
                        value=DEFAULT_DEVICE,
                        label="Embedder device",
                    )
                    recognizer = gr.Dropdown(
                        choices=list(RECOGNIZER_CHOICES),
                        value=DEFAULT_RECOGNIZER,
                        label="Recognizer",
                    )
                    detector = gr.Dropdown(
                        choices=list(DETECTOR_CHOICES),
                        value=DEFAULT_DETECTOR,
                        label="Detector",
                    )
                    script = gr.Dropdown(
                        choices=list(SCRIPT_CHOICES),
                        value=DEFAULT_SCRIPT,
                        label="Script",
                    )
                    quick = gr.Checkbox(
                        value=True,
                        label="Quick vocabulary (4k words / 3k subwords; uncheck for full)",
                    )
                with gr.Accordion("YOLO word detector", open=True):
                    checkpoint_choices = list_yolo_checkpoints()
                    initial_checkpoint = default_checkpoint or (
                        checkpoint_choices[0] if checkpoint_choices else None
                    )
                    if (
                        initial_checkpoint
                        and initial_checkpoint not in checkpoint_choices
                    ):
                        checkpoint_choices = [initial_checkpoint, *checkpoint_choices]
                    yolo_model = gr.Dropdown(
                        choices=checkpoint_choices,
                        value=initial_checkpoint,
                        label="Checkpoint (.pt)",
                        allow_custom_value=True,
                        info="Trained word checkpoints under artifacts/ (a path also works).",
                    )
                    yolo_conf = gr.Slider(
                        minimum=0.05,
                        maximum=0.95,
                        value=DEFAULT_YOLO_CONF,
                        step=0.05,
                        label="Confidence threshold",
                    )
                    yolo_imgsz = gr.Dropdown(
                        choices=list(YOLO_IMGSZ_CHOICES),
                        value=DEFAULT_YOLO_IMGSZ,
                        label="Inference size",
                    )
                examples = example_rows(profiles)
                if examples:
                    gr.Examples(
                        examples=examples,
                        inputs=[
                            image,
                            profile,
                            device,
                            recognizer,
                            detector,
                            script,
                            quick,
                        ],
                        label="Example images",
                    )
                run_button = gr.Button("Run OCR", variant="primary")
                clear_button = gr.Button("Clear")
            with gr.Column(scale=1):
                text = gr.Textbox(
                    label="Recognized text",
                    lines=8,
                )
                confidence = gr.Number(label="Confidence", precision=4)
                annotated = gr.Image(label="Detected regions", type="numpy")
                word_debug = gr.Image(
                    label="Word boxes / segmentation",
                    type="numpy",
                )
                details = gr.JSON(label="Details")
                status = gr.Markdown()

        def _run(
            image_value: Any,
            profile_value: str | None,
            device_value: str,
            recognizer_value: str,
            detector_value: str,
            script_value: str,
            quick_value: bool,
            mode_value: str,
            yolo_model_value: str | None,
            yolo_conf_value: float,
            yolo_imgsz_value: str,
            progress: Any = gr.Progress(),  # noqa: B008 - required by Gradio
        ) -> tuple[str, float, dict[str, Any], Any, Any, str]:
            return run_ocr(
                image_value,
                profile_value,
                device_value,
                recognizer_value,
                detector_value,
                script_value,
                quick_value,
                progress,
                mode=mode_value,
                yolo_model=yolo_model_value,
                yolo_conf=float(
                    yolo_conf_value
                    if yolo_conf_value is not None
                    else DEFAULT_YOLO_CONF
                ),
                yolo_imgsz=int(yolo_imgsz_value or DEFAULT_YOLO_IMGSZ),
            )

        run_button.click(
            fn=_run,
            inputs=[
                image,
                profile,
                device,
                recognizer,
                detector,
                script,
                quick,
                mode,
                yolo_model,
                yolo_conf,
                yolo_imgsz,
            ],
            outputs=[text, confidence, details, annotated, word_debug, status],
        )
        clear_button.click(
            fn=_clear,
            inputs=[],
            outputs=[image, text, confidence, annotated, word_debug, details, status],
        )

    return demo


def launch_ui(
    config_dir: str | Path = DEFAULT_CONFIG_DIR,
    host: str = "127.0.0.1",
    port: int = 7860,
    share: bool = True,
    yolo_model: str | None = None,
) -> None:
    """Launch the Gradio app."""
    demo = build_demo(config_dir, default_checkpoint=yolo_model)
    demo.launch(server_name=host, server_port=port, share=share, show_error=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the EmbeddingOCR web UI.")
    parser.add_argument("--config-dir", default=str(DEFAULT_CONFIG_DIR))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument(
        "--share",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Create a public Gradio share link (default).",
    )
    parser.add_argument(
        "--yolo-model",
        default=None,
        help="Default YOLO word checkpoint preselected in the UI.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    launch_ui(
        config_dir=args.config_dir,
        host=args.host,
        port=args.port,
        share=args.share,
        yolo_model=args.yolo_model,
    )


if __name__ == "__main__":
    main()
