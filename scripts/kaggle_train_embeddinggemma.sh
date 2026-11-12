#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root on a Kaggle T4 session (1 or 2 GPUs).
# This script never runs `uv sync`; it installs into the existing environment.
#
# Useful overrides (all optional):
#   NPROC=1|2|auto              DDP process count (default: auto = GPU count)
#   PRECISION=float32|bfloat16|auto
#   EPOCHS=3.0 BATCH_SIZE=4 EVAL_BATCH_SIZE=4 GRAD_ACCUM=4 MINI_BATCH_SIZE=8
#   TEXT_IMAGE_ROWS=60000 IMAGE_IMAGE_ROWS=20000 REAL_ROWS=20000
#   SAMPLES_PER_SOURCE=10000 DATALOADER_WORKERS=2
#   REAL_CROPS=/path/to/real.jsonl   (or a directory containing real.jsonl)
#   REBUILD=1                   Rebuild the rendered dataset even if it exists
#   MAX_STEPS=400               Stop after N optimizer steps (-1 = use EPOCHS)
#   SAVE_STEPS=1000             Checkpoint interval
#   MODEL_ID=...                Base or checkpoint model to start from
#   RESUME=/path/to/checkpoint  Resume training from a checkpoint directory
#   OUTPUT=/path/to/run         Fine-tuned model directory

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VENV_PY="${VENV_PY:-$REPO_ROOT/.venv/bin/python}"
TEXTS="${TEXTS:-$REPO_ROOT/data/short_texts.jsonl}"
FONTS_ROOT="${FONTS_ROOT:-$REPO_ROOT/data/google_fonts}"
FONTS="${FONTS:-$FONTS_ROOT/ofl}"
DATASET="${DATASET:-$REPO_ROOT/artifacts/embedding_dataset}"
REAL_CROPS="${REAL_CROPS:-}"
OUTPUT="${OUTPUT:-$REPO_ROOT/artifacts/embeddinggemma-ocr}"

NPROC="${NPROC:-auto}"
PRECISION="${PRECISION:-float32}"
EPOCHS="${EPOCHS:-3.0}"
BATCH_SIZE="${BATCH_SIZE:-4}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-8}"
DATALOADER_WORKERS="${DATALOADER_WORKERS:-2}"
TEXT_IMAGE_ROWS="${TEXT_IMAGE_ROWS:-60000}"
IMAGE_IMAGE_ROWS="${IMAGE_IMAGE_ROWS:-20000}"
REAL_ROWS="${REAL_ROWS:-20000}"
SAMPLES_PER_SOURCE="${SAMPLES_PER_SOURCE:-10000}"
REBUILD="${REBUILD:-0}"
MAX_STEPS="${MAX_STEPS:--1}"
SAVE_STEPS="${SAVE_STEPS:-1000}"
MODEL_ID="${MODEL_ID:-}"
RESUME="${RESUME:-}"

cd "$REPO_ROOT"

if [[ ! -x "$VENV_PY" ]]; then
  uv venv --python 3.11 "$REPO_ROOT/.venv"
fi

uv pip install --python "$VENV_PY" -e ".[finetune]"

if [[ ! -d "$FONTS" ]]; then
  uv run --no-sync python scripts/download_google_fonts.py --output "$FONTS_ROOT"
fi

if [[ ! -f "$TEXTS" ]]; then
  uv run --no-sync python scripts/sample_short_texts.py \
    --per-source "$SAMPLES_PER_SOURCE" \
    --output "$TEXTS"
fi

if [[ -n "$REAL_CROPS" && ! -e "$REAL_CROPS" ]]; then
  echo "REAL_CROPS does not exist: $REAL_CROPS" >&2
  exit 1
fi

if [[ "$REBUILD" == "1" || ! -f "$DATASET/summary.json" ]]; then
  REAL_ARGS=()
  if [[ -n "$REAL_CROPS" ]]; then
    REAL_ARGS=(--real-crops "$REAL_CROPS" --real-rows "$REAL_ROWS")
  fi

  uv run --no-sync python scripts/build_embedding_dataset.py \
    --fonts-dir "$FONTS" \
    --texts "$TEXTS" \
    --output "$DATASET" \
    --text-image-rows "$TEXT_IMAGE_ROWS" \
    --image-image-rows "$IMAGE_IMAGE_ROWS" \
    "${REAL_ARGS[@]}"
else
  echo "Reusing existing dataset at $DATASET (set REBUILD=1 to regenerate)."
fi

BF16_CHECK=(
  python -c "import torch; raise SystemExit(0 if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8 and torch.cuda.is_bf16_supported() else 1)"
)

if [[ "$PRECISION" == "auto" ]]; then
  if uv run --no-sync "${BF16_CHECK[@]}"; then
    PRECISION="bfloat16"
  else
    PRECISION="float32"
  fi
fi
if [[ "$PRECISION" != "float32" && "$PRECISION" != "bfloat16" ]]; then
  echo "PRECISION must be float32, bfloat16 or auto (got '$PRECISION')" >&2
  exit 1
fi
if [[ "$PRECISION" == "bfloat16" ]] && ! uv run --no-sync "${BF16_CHECK[@]}"; then
  echo "bfloat16 is not supported on this GPU (T4); falling back to float32." >&2
  PRECISION="float32"
fi

if ! uv run --no-sync python -c \
  "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)"; then
  echo "No CUDA device found; this launcher targets Kaggle T4 GPUs." >&2
  exit 1
fi

if [[ "$NPROC" == "auto" ]]; then
  NPROC="$(uv run --no-sync python -c \
    "import torch; print(max(1, torch.cuda.device_count()))")"
fi
if ! [[ "$NPROC" =~ ^[0-9]+$ ]] || [[ "$NPROC" -lt 1 ]]; then
  echo "NPROC must be a positive integer or auto (got '$NPROC')" >&2
  exit 1
fi

TRAIN_CMD=(python scripts/train_embeddinggemma.py)
DDP_ARGS=()
if [[ "$NPROC" -gt 1 ]]; then
  TRAIN_CMD=(torchrun --standalone --nproc_per_node="$NPROC" scripts/train_embeddinggemma.py)
  DDP_ARGS=(--gather-across-devices)
fi

echo "Launching with NPROC=$NPROC PRECISION=$PRECISION DEVICE=cuda"
MODEL_ARGS=()
if [[ -n "$MODEL_ID" ]]; then
  MODEL_ARGS=(--model "$MODEL_ID")
fi
RESUME_ARGS=()
if [[ -n "$RESUME" ]]; then
  if [[ ! -d "$RESUME" ]]; then
    echo "RESUME checkpoint not found: $RESUME" >&2
    exit 1
  fi
  RESUME_ARGS=(--resume-from-checkpoint "$RESUME")
fi
uv run --no-sync "${TRAIN_CMD[@]}" \
  --config configs/embedding_finetune.yaml \
  --dataset "$DATASET" \
  --output "$OUTPUT" \
  --precision "$PRECISION" \
  --device cuda \
  --epochs "$EPOCHS" \
  --max-steps "$MAX_STEPS" \
  --save-steps "$SAVE_STEPS" \
  --probe-batch \
  --retune-batch \
  --batch-size "$BATCH_SIZE" \
  --eval-batch-size "$EVAL_BATCH_SIZE" \
  --mini-batch-size "$MINI_BATCH_SIZE" \
  --gradient-accumulation-steps "$GRAD_ACCUM" \
  --dataloader-workers "$DATALOADER_WORKERS" \
  "${DDP_ARGS[@]}" \
  "${RESUME_ARGS[@]}" \
  "${MODEL_ARGS[@]}"

uv run --no-sync python scripts/eval_embedding_retrieval.py \
  --model-id "$OUTPUT/final" \
  --dataset "$DATASET" \
  --device cuda \
  --dtype "$PRECISION" \
  --output "$REPO_ROOT/artifacts/retrieval_metrics.json"
