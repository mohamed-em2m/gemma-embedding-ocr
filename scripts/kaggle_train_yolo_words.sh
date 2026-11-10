#!/usr/bin/env bash
# Train the one-class YOLO word detector on Kaggle's 2x T4 GPU runtime.
#
# Run from a Kaggle notebook cell:
#   !bash scripts/kaggle_train_yolo_words.sh
#
# Override any default with an environment variable, e.g.:
#   !EPOCHS=60 BATCH=24 bash scripts/kaggle_train_yolo_words.sh

set -Eeuo pipefail
trap 'echo "ERROR: Kaggle training failed at line $LINENO" >&2' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

WORK_DIR="${WORK_DIR:-/kaggle/working/artifacts}"
FONTS="${FONTS:-/usr/share/fonts}"
N="${N:-30000}"
IMGSZ="${IMGSZ:-1280}"
EPOCHS="${EPOCHS:-100}"
MODEL="${MODEL:-yolo26s.pt}"
BATCH="${BATCH:-32}"
DEVICES="${DEVICES:-0,1}"
WORKERS="${WORKERS:-8}"
CACHE="${CACHE:-disk}"
NAME="${NAME:-yolo_words}"
DATASET="${DATASET:-$WORK_DIR/yolo_words}"
RUNS="${RUNS:-$WORK_DIR/yolo_runs}"

if [[ ! -x .venv/bin/python ]]; then
    echo "Preparing the training environment (--train, no system packages)..."
    bash scripts/setup_env.sh \
        --train --no-system --no-ui --no-shim --no-catalogs --no-hooks
fi

echo "Generating word dataset at $DATASET ..."
uv run --no-sync python scripts/build_word_dataset.py \
    --fonts "$FONTS" \
    --out "$DATASET" \
    --n "$N" \
    --size "$IMGSZ" \
    --layout mixed

echo "Training on GPU devices $DEVICES ..."
uv run --no-sync python scripts/train_yolo_words.py \
    --data "$DATASET/data.yaml" \
    --model "$MODEL" \
    --epochs "$EPOCHS" \
    --imgsz "$IMGSZ" \
    --batch "$BATCH" \
    --device "$DEVICES" \
    --workers "$WORKERS" \
    --cache "$CACHE" \
    --project "$RUNS" \
    --name "$NAME"

echo "Best checkpoint: $RUNS/$NAME/weights/best.pt"
echo "Copy it to a Kaggle dataset or download it before the session ends."
