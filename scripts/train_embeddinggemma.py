#!/usr/bin/env python
"""Fine-tune EmbeddingGemma 2 on rendered text and real OCR word crops."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import torch
import yaml
from datasets import Dataset, DatasetDict
from PIL import Image
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.base.sampler import MultiDatasetBatchSamplers
from sentence_transformers.sentence_transformer.losses import (
    CachedMultipleNegativesRankingLoss,
)

from embedding_ocr.training.batch_probe import (
    DEFAULT_BATCH_TUNING_PATH,
    probe_training_batch,
)
from embedding_ocr.training.embedding_dataset import load_manifest
from embedding_ocr.training.sampler import (
    OcrDataCollator,
    UniqueTextBatchSampler,
    split_sentence_features,
)
from embedding_ocr.utils.gpu import resolve_device

_REPO_ROOT = Path(__file__).resolve().parents[1]
LOGGER = logging.getLogger("train_embeddinggemma")
MODES = ("text_image", "image_image")


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path, default=None)
    known, _ = config_parser.parse_known_args()
    defaults = _load_defaults(known.config)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.set_defaults(**defaults)
    parser.add_argument("--config", type=Path, default=known.config)
    parser.add_argument("--model", default="google/embeddinggemma-2")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=_REPO_ROOT / "artifacts/embedding_dataset",
        help="Directory produced by scripts/build_embedding_dataset.py.",
    )
    parser.add_argument(
        "--output", type=Path, default=_REPO_ROOT / "artifacts/embeddinggemma-ocr"
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--precision", choices=("auto", "float32", "bfloat16"), default="float32"
    )
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--mini-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-seq-length", type=int, default=64)
    parser.add_argument("--canvas-size", type=int, default=128)
    parser.add_argument("--prompt-name", default="Document")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--gather-across-devices", action="store_true")
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--save-steps", type=int, default=1000)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--eval-steps", type=int, default=1000)
    parser.add_argument("--dataloader-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--probe-batch", action="store_true")
    parser.add_argument("--retune-batch", action="store_true")
    parser.add_argument(
        "--batch-cache",
        type=Path,
        default=DEFAULT_BATCH_TUNING_PATH,
    )
    return parser.parse_args()


def _load_defaults(path: Path | None) -> dict[str, object]:
    if path is None:
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        msg = f"Config must be a YAML mapping: {path}"
        raise ValueError(msg)
    key_map = {
        "model_id": "model",
        "dataset_dir": "dataset",
        "output_dir": "output",
        "per_device_train_batch_size": "batch_size",
        "per_device_eval_batch_size": "eval_batch_size",
        "num_train_epochs": "epochs",
    }
    defaults: dict[str, object] = {}
    for key, value in payload.items():
        target = key_map.get(str(key), str(key))
        defaults[target] = value
    return defaults


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    device = resolve_device(args.device)
    dtype = _resolve_dtype(args.precision, device)
    LOGGER.info("Device=%s dtype=%s", device, dtype)

    train_dataset, eval_dataset = _load_datasets(args.dataset)
    model = SentenceTransformer(
        args.model,
        device=device,
        model_kwargs={"torch_dtype": dtype},
        config_kwargs={"audio_config": None},
    )
    if args.max_seq_length > 0:
        model.max_seq_length = args.max_seq_length
    prompt = _resolve_prompt(model, args.prompt_name)
    if prompt is None:
        LOGGER.warning(
            "Prompt %r is not registered; text anchors use no prompt", args.prompt_name
        )

    train_collator = OcrDataCollator(
        preprocess_fn=model.preprocess,
        prompts={
            "text_image": {"anchor": prompt} if prompt else {},
            "image_image": {},
        },
    )
    loss = CachedMultipleNegativesRankingLoss(
        model=model,
        mini_batch_size=args.mini_batch_size,
        gather_across_devices=args.gather_across_devices,
        hardness_mode="hard_negatives",
        hardness_strength=5.0,
    )

    batch_size = args.batch_size
    if args.probe_batch:
        batch_size = _probe_batch_size(
            model=model,
            dataset=train_dataset["image_image"],
            collator=train_collator,
            loss=loss,
            args=args,
            device=device,
            dtype=dtype,
        )
        LOGGER.info("Using probed per-device batch size: %d", batch_size)

    training_args = SentenceTransformerTrainingArguments(
        output_dir=str(args.output),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        bf16=dtype == torch.bfloat16,
        fp16=False,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=args.eval_steps if eval_dataset is not None else None,
        dataloader_num_workers=args.dataloader_workers,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        report_to="none",
        seed=args.seed,
        batch_sampler=UniqueTextBatchSampler,
        multi_dataset_batch_sampler=MultiDatasetBatchSamplers.PROPORTIONAL,
    )
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
        data_collator=train_collator,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(args.output / "final"))
    model.save_pretrained(str(args.output / "final"))
    _write_run_metadata(args, device, dtype, batch_size)
    LOGGER.info("Saved fine-tuned model to %s", args.output / "final")


def _load_datasets(root: Path) -> tuple[DatasetDict, DatasetDict | None]:
    train = _load_split(root / "train")
    val = _load_split(root / "val")
    eval_dataset = val if len(val) and all(len(ds) for ds in val.values()) else None
    if eval_dataset is None:
        LOGGER.warning("No validation rows; running without evaluation")
    return train, eval_dataset


def _load_split(root: Path) -> DatasetDict:
    datasets: dict[str, Dataset] = {}
    for mode in MODES:
        path = root / f"{mode}.jsonl"
        if not path.exists():
            LOGGER.warning("Missing manifest %s", path)
            continue
        rows = load_manifest(path)
        if not rows:
            continue
        datasets[mode] = _dataset_from_rows(rows, mode)
    if not datasets:
        msg = f"No manifests found under {root}"
        raise FileNotFoundError(msg)
    return DatasetDict(datasets)


def _dataset_from_rows(rows: list[dict[str, object]], mode: str) -> Dataset:
    anchors: list[object] = []
    positives: list[Image.Image] = []
    negatives: list[Image.Image] = []
    text_ids: list[str] = []
    texts: list[str] = []
    for row in rows:
        text = str(row.get("text", ""))
        text_id = str(row.get("text_id", ""))
        positive = row.get("positive")
        negative = row.get("negative")
        if not isinstance(positive, str) or not isinstance(negative, str):
            continue
        if mode == "text_image":
            anchors.append(text)
        else:
            anchor_path = row.get("anchor_path")
            if not isinstance(anchor_path, str):
                continue
            anchors.append(Image.open(anchor_path).convert("RGB"))
        positives.append(Image.open(positive).convert("RGB"))
        negatives.append(Image.open(negative).convert("RGB"))
        text_ids.append(text_id)
        texts.append(text)
    return Dataset.from_dict(
        {
            "anchor": anchors,
            "positive": positives,
            "negative": negatives,
            "text_id": text_ids,
            "text": texts,
        }
    )


def _resolve_dtype(precision: str, device: str) -> torch.dtype:
    if precision == "bfloat16":
        if not device.startswith("cuda") or not torch.cuda.is_bf16_supported():
            msg = (
                "bfloat16 requires a CUDA device with bf16 support; use float32 on T4."
            )
            raise ValueError(msg)
        return torch.bfloat16
    if precision == "float32":
        return torch.float32
    if device.startswith("cuda") and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32


def _resolve_prompt(model: SentenceTransformer, name: str) -> str | None:
    prompts = getattr(model, "prompts", {})
    if isinstance(prompts, dict):
        value = prompts.get(name)
        return value if isinstance(value, str) else None
    return None


def _probe_batch_size(
    *,
    model: SentenceTransformer,
    dataset: Dataset,
    collator: OcrDataCollator,
    loss: CachedMultipleNegativesRankingLoss,
    args: argparse.Namespace,
    device: str,
    dtype: torch.dtype,
) -> int:
    probe_collator = OcrDataCollator(
        preprocess_fn=model.preprocess,
        prompts={},
    )

    def step(batch_size: int) -> None:
        items = [dataset[index] for index in range(min(batch_size, len(dataset)))]
        if len(items) < batch_size:
            items = items * (batch_size // max(1, len(items)) + 1)
            items = items[:batch_size]
        batch = probe_collator(items)
        batch = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        features = split_sentence_features(batch)
        labels = torch.empty(0, dtype=torch.long, device=device)
        output = loss(features, labels)
        output.backward()
        model.zero_grad(set_to_none=True)

    key = {
        "model_id": args.model,
        "precision": str(dtype).removeprefix("torch."),
        "image_size": args.canvas_size,
        "loss": "cached_mnrl_hard",
        "max_seq_length": args.max_seq_length,
        "gradient_checkpointing": args.gradient_checkpointing,
        "dataset": "image_image",
        "world_size": int(os.environ.get("WORLD_SIZE", "1")),
    }
    rank = int(os.environ.get("RANK", "0"))
    cache_path = args.batch_cache
    if rank:
        cache_path = cache_path.with_name(
            f"{cache_path.stem}.rank{rank}{cache_path.suffix}"
        )
    result = probe_training_batch(
        step,
        key=key,
        device=device,
        cache_path=cache_path,
        retune=args.retune_batch,
        world_size=int(os.environ.get("WORLD_SIZE", "1")),
    )
    LOGGER.info("Batch probe: %s", json.dumps(result.as_dict(), indent=2))

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        tensor = torch.tensor([result.batch_size], device=device)
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.MIN)
        return int(tensor.item())
    return result.batch_size


def _write_run_metadata(
    args: argparse.Namespace,
    device: str,
    dtype: torch.dtype,
    batch_size: int,
) -> None:
    payload = vars(args).copy()
    payload["dataset"] = str(args.dataset)
    payload["output"] = str(args.output)
    payload["device_resolved"] = device
    payload["dtype"] = str(dtype).removeprefix("torch.")
    payload["per_device_train_batch_size"] = batch_size
    try:
        import subprocess

        payload["git_hash"] = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
    except OSError:
        payload["git_hash"] = None
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "run_metadata.json").write_text(
        json.dumps(payload, indent=2, default=str)
    )


if __name__ == "__main__":
    main()
