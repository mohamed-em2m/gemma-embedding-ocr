#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root on a Kaggle 2x T4 session.
# This script never runs `uv sync`; it installs into the existing environment.

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VENV_PY="${VENV_PY:-$REPO_ROOT/.venv/bin/python}"
TEXTS="${TEXTS:-$REPO_ROOT/data/short_texts.jsonl}"
FONTS="${FONTS:-$REPO_ROOT/data/google_fonts/ofl}"
DATASET="${DATASET:-$REPO_ROOT/artifacts/embedding_dataset}"
REAL_CROPS="${REAL_CROPS:-}"

cd "$REPO_ROOT"

if [[ ! -x "$VENV_PY" ]]; then
  uv venv --python 3.11 "$REPO_ROOT/.venv"
fi

uv pip install --python "$VENV_PY" -e ".[finetune]"

if [[ ! -d "$(dirname "$FONTS")/.git" ]]; then
  uv run --no-sync python scripts/download_google_fonts.py --output "$(dirname "$FONTS")"
fi

if [[ ! -f "$TEXTS" ]]; then
  uv run --no-sync python scripts/sample_short_texts.py \
    --per-source 10000 \
    --output "$TEXTS"
fi

REAL_ARGS=()
if [[ -n "$REAL_CROPS" ]]; then
  REAL_ARGS=(--real-crops "$REAL_CROPS" --real-rows 20000)
fi

uv run --no-sync python scripts/build_embedding_dataset.py \
  --fonts-dir "$FONTS" \
  --texts "$TEXTS" \
  --output "$DATASET" \
  --text-image-rows 60000 \
  --image-image-rows 20000 \
  "${REAL_ARGS[@]}"

if ! uv run --no-sync python -c "import torch; raise SystemExit(0 if torch.cuda.is_bf16_supported() else 1)"; then
  echo "T4 detected: using float32 (EmbeddingGemma 2 does not support fp16)."
fi

uv run --no-sync torchrun --standalone --nproc_per_node=2 scripts/train_embeddinggemma.py \
  --config configs/embedding_finetune.yaml \
  --dataset "$DATASET" \
  --precision float32 \
  --device cuda \
  --probe-batch \
  --retune-batch \
  --gather-across-devices \
  --batch-size 4 \
  --eval-batch-size 4 \
  --gradient-accumulation-steps 4 \
  --dataloader-workers 2

uv run --no-sync python scripts/eval_embedding_retrieval.py \
  --model-id "$REPO_ROOT/artifacts/embeddinggemma-ocr/final" \
  --dataset "$DATASET" \
  --device cuda \
  --dtype float32 \
  --output "$REPO_ROOT/artifacts/retrieval_metrics.json"
