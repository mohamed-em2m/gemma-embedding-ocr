#!/usr/bin/env python
"""Train the CTC line recognizer on a prepared line manifest.

The model is a CRNN encoder with a CTC head plus a small transformer decoder,
trained with ``CTC + cross-entropy``. Validation reports character error rate
for greedy CTC decoding and, optionally, for the autoregressive decoder.

Examples::

    # Full run (CPU works; a GPU is much faster)
    uv run --no-sync python scripts/train_ctc_ocr.py \
        --config configs/ctc_line.yaml

    # Memorization sanity check before a long run
    uv run --no-sync python scripts/train_ctc_ocr.py --overfit 32 --epochs 40

    # Resume
    uv run --no-sync python scripts/train_ctc_ocr.py --resume artifacts/ctc_ocr/last.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import jiwer
import torch
import torch.nn.functional as F  # noqa: N812
import yaml
from torch import Tensor
from torch.utils.data import DataLoader

from embedding_ocr.recognition.ctc.charset import CharSet
from embedding_ocr.recognition.ctc.checkpoint import (
    load_checkpoint,
    load_optimizer_state,
    save_checkpoint,
)
from embedding_ocr.recognition.ctc.decode import (
    greedy_decode_batch,
    log_softmax,
    mask_padded,
)
from embedding_ocr.recognition.ctc.model import CTCSeq2Seq, ModelConfig
from embedding_ocr.recognition.ctc.normalize import normalize_for_cer
from embedding_ocr.training.ctc_dataset import (
    BucketBatchSampler,
    LineBatch,
    LineCropDataset,
    LineDatasetConfig,
    collate_lines,
)
from embedding_ocr.training.line_dataset import LineRecord, read_manifest

_REPO_ROOT = Path(__file__).resolve().parents[1]
LOGGER = logging.getLogger("train_ctc_ocr")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--config", type=Path, default=_REPO_ROOT / "configs/ctc_line.yaml"
    )
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--charset", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--epochs", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument(
        "--ctc-weight",
        type=float,
        default=None,
        help="Weight on the CTC loss; the decoder gets 1 - w (default: from config).",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--scheduler",
        choices=("cosine", "constant"),
        default=None,
        help="LR schedule after warmup (default: from config, cosine).",
    )
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument(
        "--overfit",
        type=int,
        default=0,
        help="Train and validate on the same N samples (memorization check).",
    )
    parser.add_argument("--limit-batches", type=int, default=0)
    parser.add_argument("--decoder-eval-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def _load_config(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        message = f"invalid training config: {path}"
        raise ValueError(message)
    return loaded


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else _REPO_ROOT / path


def _device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _split(records: list[LineRecord], name: str) -> list[LineRecord]:
    return [record for record in records if record.split == name]


def _loaders(
    train: list[LineRecord],
    val: list[LineRecord],
    charset: CharSet,
    data_config: LineDatasetConfig,
    model_config: ModelConfig,
    *,
    batch_size: int,
    eval_batch_size: int,
    workers: int,
    seed: int,
) -> tuple[DataLoader[Any], DataLoader[Any]]:
    train_dataset = LineCropDataset(
        train,
        charset,
        data_config,
        max_decoder_len=model_config.max_decoder_len,
    )
    val_dataset = LineCropDataset(
        val,
        charset,
        LineDatasetConfig(
            image_height=data_config.image_height,
            max_width=data_config.max_width,
            augment=False,
        ),
        max_decoder_len=model_config.max_decoder_len,
    )
    train_sampler = BucketBatchSampler(
        [train_dataset.sort_key(index) for index in range(len(train_dataset))],
        batch_size,
        shuffle=True,
        seed=seed,
        drop_last=len(train_dataset) > batch_size,
    )
    val_sampler = BucketBatchSampler(
        [val_dataset.sort_key(index) for index in range(len(val_dataset))],
        eval_batch_size,
        shuffle=False,
    )
    collate = lambda batch: collate_lines(batch, eos_index=charset.size)  # noqa: E731
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        collate_fn=collate,
        num_workers=workers,
        persistent_workers=workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_sampler=val_sampler,
        collate_fn=collate,
        num_workers=workers,
        persistent_workers=workers > 0,
    )
    return train_loader, val_loader


def _to_device(batch: LineBatch, device: torch.device) -> LineBatch:
    return LineBatch(
        images=batch.images.to(device),
        widths=batch.widths.to(device),
        ctc_targets=batch.ctc_targets.to(device),
        target_lengths=batch.target_lengths.to(device),
        decoder_inputs=batch.decoder_inputs.to(device),
        decoder_targets=batch.decoder_targets.to(device),
        texts=batch.texts,
    )


def _loss(
    model: CTCSeq2Seq,
    batch: LineBatch,
    ctc_weight: float,
    label_smoothing: float,
) -> tuple[Tensor, Tensor, Tensor]:
    ctc_logits, decoder_logits, lengths = model(
        batch.images, batch.widths, batch.decoder_inputs
    )
    log_probs = ctc_logits.log_softmax(dim=-1).transpose(0, 1)  # (T, B, V)
    ctc_loss = F.ctc_loss(
        log_probs,
        batch.ctc_targets,
        lengths,
        batch.target_lengths,
        blank=0,
        zero_infinity=True,
    )
    if decoder_logits is None:  # pragma: no cover - always present in training
        return ctc_loss, ctc_loss.detach(), torch.zeros((), device=ctc_loss.device)
    flat_logits = decoder_logits.reshape(-1, decoder_logits.shape[-1])
    flat_targets = batch.decoder_targets.reshape(-1)
    ce_loss = F.cross_entropy(
        flat_logits, flat_targets, ignore_index=-100, label_smoothing=label_smoothing
    )
    total = ctc_weight * ctc_loss + (1.0 - ctc_weight) * ce_loss
    return total, ctc_loss.detach(), ce_loss.detach()


def _character_error_rate(references: list[str], hypotheses: list[str]) -> float:
    if not references:
        return 0.0
    return float(
        jiwer.cer(
            [normalize_for_cer(text) for text in references],
            [normalize_for_cer(text) for text in hypotheses],
        )
    )


@torch.no_grad()
def _evaluate(
    model: CTCSeq2Seq,
    loader: DataLoader[LineBatch],
    charset: CharSet,
    device: torch.device,
    decoder_samples: int,
) -> dict[str, Any]:
    model.eval()
    references: list[str] = []
    ctc_predictions: list[str] = []
    decoder_predictions: list[str] = []
    decoder_seen = 0

    for batch in loader:
        batch = _to_device(batch, device)
        ctc_logits, _, lengths = model(batch.images, batch.widths)
        log_probs = mask_padded(
            log_softmax(ctc_logits.float().cpu().numpy()), lengths.cpu().tolist()
        )
        decoded = greedy_decode_batch(log_probs)
        references.extend(batch.texts)
        ctc_predictions.extend(charset.decode(item.indices) for item in decoded)

        if decoder_samples and decoder_seen < decoder_samples:
            take = min(decoder_samples - decoder_seen, batch.images.shape[0])
            generated = model.greedy_decode(batch.images[:take], batch.widths[:take])
            decoder_predictions.extend(charset.decode(ids) for ids in generated)
            decoder_seen += take

    results: dict[str, Any] = {
        "ctc_cer": _character_error_rate(references, ctc_predictions),
        "samples": len(references),
        "examples": [
            {
                "reference": reference,
                "ctc": ctc_predictions[index],
                "decoder": decoder_predictions[index]
                if index < len(decoder_predictions)
                else "",
            }
            for index, reference in enumerate(references[:8])
        ],
    }
    if decoder_predictions:
        results["decoder_cer"] = _character_error_rate(
            references[: len(decoder_predictions)], decoder_predictions
        )
    model.train()
    return results


def _scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    *,
    kind: str = "cosine",
    min_ratio: float = 0.05,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup followed by a cosine decay to ``min_ratio`` of the peak LR.

    ``constant`` keeps the LR at its peak after warmup, which is what short
    memorization checks need: a cosine decay that reaches zero over a few
    hundred steps starves CTC before it converges.
    """
    if kind not in {"cosine", "constant"}:
        message = f"unknown scheduler {kind!r}; expected 'cosine' or 'constant'"
        raise ValueError(message)

    def scale(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        if kind == "constant":
            return 1.0
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return min_ratio + (1.0 - min_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    config = _load_config(args.config)

    data_settings = dict(config.get("data") or {})
    train_settings = dict(config.get("train") or {})
    model_settings = dict(config.get("model") or {})
    manifest_path = _resolve(args.manifest or Path(str(data_settings["manifest"])))
    charset_path = _resolve(
        args.charset
        or Path(
            str(data_settings.get("charset", manifest_path.parent / "charset.json"))
        )
    )
    output = _resolve(
        args.output or Path(str(config.get("output", "artifacts/ctc_ocr")))
    )
    output.mkdir(parents=True, exist_ok=True)

    seed = args.seed if args.seed is not None else int(train_settings.get("seed", 42))
    random.seed(seed)
    torch.manual_seed(seed)

    charset = CharSet.load(charset_path)
    records = read_manifest(manifest_path)
    train_records = _split(records, "train")
    val_records = _split(records, "val") or _split(records, "test")
    if not train_records:
        message = f"no train records in {manifest_path}"
        raise ValueError(message)
    if not val_records:
        message = f"no val or test records in {manifest_path}"
        raise ValueError(message)
    if args.overfit:
        train_records = train_records[: args.overfit]
        val_records = train_records
        LOGGER.info(
            "overfit mode: %d samples used for train and val", len(train_records)
        )

    image_height = int(data_settings.get("image_height", 32))
    model_config = ModelConfig(
        vocab_size=charset.size,
        image_height=image_height,
        **{
            key: value
            for key, value in model_settings.items()
            if key in ModelConfig.__dataclass_fields__
        },
    )
    data_config = LineDatasetConfig(
        image_height=image_height,
        max_width=int(data_settings.get("max_width", 1280)),
        augment=bool(data_settings.get("augment", True)),
    )
    device = _device(args.device or str(train_settings.get("device", "auto")))
    amp = args.amp or bool(train_settings.get("amp", False))
    if amp and device.type != "cuda":
        LOGGER.warning("AMP requested without CUDA; disabling it")
        amp = False

    batch_size = args.batch_size or int(train_settings.get("batch_size", 32))
    eval_batch_size = int(train_settings.get("eval_batch_size", batch_size))
    workers = (
        args.num_workers
        if args.num_workers is not None
        else int(train_settings.get("num_workers", 2))
    )
    train_loader, val_loader = _loaders(
        train_records,
        val_records,
        charset,
        data_config,
        model_config,
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        workers=workers,
        seed=seed,
    )

    model = CTCSeq2Seq(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr or float(train_settings.get("lr", 3e-4)),
        weight_decay=float(train_settings.get("weight_decay", 1e-4)),
    )
    epochs = int(args.epochs or train_settings.get("epochs", 25))
    steps_per_epoch = args.limit_batches or len(train_loader)
    total_steps = max(1, epochs * steps_per_epoch)
    warmup = int(total_steps * float(train_settings.get("warmup_ratio", 0.05)))
    scheduler_kind = args.scheduler or str(train_settings.get("scheduler", "cosine"))
    scheduler = _scheduler(
        optimizer,
        warmup,
        total_steps,
        kind=scheduler_kind,
        min_ratio=float(train_settings.get("min_lr_ratio", 0.05)),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    grad_clip = float(train_settings.get("grad_clip", 5.0))
    ctc_weight = float(
        args.ctc_weight
        if args.ctc_weight is not None
        else train_settings.get("ctc_weight", 0.3)
    )
    label_smoothing = float(train_settings.get("label_smoothing", 0.1))
    decoder_samples = (
        args.decoder_eval_samples
        if args.decoder_eval_samples is not None
        else int(train_settings.get("decoder_eval_samples", 200))
    )

    start_epoch = 0
    best_cer = float("inf")
    global_step = 0
    if args.resume is not None:
        payload = load_checkpoint(args.resume, map_location="cpu")
        model.load_state_dict(payload["state_dict"])
        optimizer_state = load_optimizer_state(payload)
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
        start_epoch = int(payload.get("epoch", 0))
        global_step = int(payload.get("step", 0))
        best_cer = float(payload.get("metrics", {}).get("ctc_cer", best_cer))
        LOGGER.info("resumed from %s at epoch %d", args.resume, start_epoch)

    normalization = {}
    stats_path = manifest_path.parent / "stats.json"
    if stats_path.is_file():
        normalization = json.loads(stats_path.read_text(encoding="utf-8")).get(
            "normalization", {}
        )
    (output / "config.json").write_text(
        json.dumps(
            {
                "model": asdict(model_config),
                "data": asdict(data_config),
                "train": train_settings,
                "manifest": str(manifest_path),
                "charset": str(charset_path),
                "parameters": model.parameter_count(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    LOGGER.info(
        "model %.2fM parameters, %d train / %d val records, device=%s",
        model.parameter_count() / 1e6,
        len(train_records),
        len(val_records),
        device,
    )

    log_path = output / "train_log.jsonl"
    for epoch in range(start_epoch, epochs):
        model.train()
        started = time.perf_counter()
        running = 0.0
        running_ctc = 0.0
        running_ce = 0.0
        batches = 0
        for batch in train_loader:
            batch = _to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=amp):
                loss, ctc_loss, ce_loss = _loss(
                    model, batch, ctc_weight, label_smoothing
                )
            if amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            scheduler.step()
            global_step += 1
            batches += 1
            running += float(loss.detach())
            running_ctc += float(ctc_loss)
            running_ce += float(ce_loss)
            if global_step % int(train_settings.get("log_every", 50)) == 0:
                LOGGER.info(
                    "epoch %d step %d loss %.4f ctc %.4f ce %.4f lr %.2e",
                    epoch,
                    global_step,
                    running / batches,
                    running_ctc / batches,
                    running_ce / batches,
                    scheduler.get_last_lr()[0],
                )
            if args.limit_batches and batches >= args.limit_batches:
                break

        metrics = _evaluate(
            model, val_loader, charset, device, decoder_samples=decoder_samples
        )
        elapsed = time.perf_counter() - started
        entry = {
            "epoch": epoch,
            "step": global_step,
            "loss": running / max(1, batches),
            "ctc_loss": running_ctc / max(1, batches),
            "ce_loss": running_ce / max(1, batches),
            "lr": scheduler.get_last_lr()[0],
            "seconds": round(elapsed, 1),
            **{key: value for key, value in metrics.items() if key != "examples"},
        }
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")
        LOGGER.info(
            "epoch %d done in %.0fs: ctc_cer=%.4f decoder_cer=%s",
            epoch,
            elapsed,
            float(metrics["ctc_cer"]),
            f"{float(metrics['decoder_cer']):.4f}"
            if "decoder_cer" in metrics
            else "n/a",
        )
        for example in metrics["examples"][:3]:
            LOGGER.info(
                "  ref=%r ctc=%r decoder=%r",
                example["reference"],
                example["ctc"],
                example["decoder"],
            )

        save_checkpoint(
            output / "last.pt",
            model,
            charset=charset,
            optimizer=optimizer,
            epoch=epoch + 1,
            step=global_step,
            metrics={"ctc_cer": float(metrics["ctc_cer"])},
            normalization=normalization,
        )
        if float(metrics["ctc_cer"]) < best_cer:
            best_cer = float(metrics["ctc_cer"])
            save_checkpoint(
                output / "best.pt",
                model,
                charset=charset,
                optimizer=optimizer,
                epoch=epoch + 1,
                step=global_step,
                metrics={"ctc_cer": best_cer},
                normalization=normalization,
            )
            LOGGER.info("new best ctc_cer=%.4f -> %s", best_cer, output / "best.pt")


if __name__ == "__main__":
    main()
