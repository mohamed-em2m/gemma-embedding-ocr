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

if ! command -v uv >/dev/null 2>&1; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

if [[ ! -x .venv/bin/python ]]; then
    echo "Creating .venv with Python 3.11..."
    uv venv --python 3.11 .venv
fi

echo "Ensuring the training extra is installed into .venv..."
uv pip install --python "$PROJECT_ROOT/.venv/bin/python" -e "$PROJECT_ROOT[train]"
if ! uv run --no-sync python -c "import ultralytics, uharfbuzz" >/dev/null 2>&1; then
    echo "ERROR: ultralytics/uharfbuzz are not importable from .venv." >&2
    exit 1
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
