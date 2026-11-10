#!/usr/bin/env bash
# Download the published word-detection dataset from the Hugging Face Hub and
# train the YOLO word detector on it. Works locally, on Colab and on Kaggle.
#
#   bash scripts/train_from_hf.sh
#   EPOCHS=40 IMGSZ=960 BATCH=16 DEVICE=0 bash scripts/train_from_hf.sh
#   HF_REPO=someone/other-dataset bash scripts/train_from_hf.sh
#
# Any setting can be overridden with an environment variable:
#   HF_REPO     dataset repo id          (mohamedemam/words-detection)
#   HF_DEST     local download dir       (artifacts/words-detection)
#   HF_METHOD   auto | hub | git         (auto)
#   HF_TOKEN    token for a private repo (empty)
#   HF_FORCE    1 = re-download          (0)
#   MODEL       base checkpoint          (yolo26s.pt)
#   EPOCHS      training epochs          (40)
#   IMGSZ       image size               (960)
#   BATCH       batch size               (16)
#   DEVICE      cuda index or auto/cpu   (auto)
#   WORKERS     dataloader workers       (4)
#   NAME        run name                 (yolo_words_hf)
#   RUNS        output directory         (artifacts/yolo_runs)
#
# A plain `git clone` needs git-lfs; without it the images arrive as pointer
# files. The downloader detects that and tells you to use the Hub client.

set -Eeuo pipefail
trap 'echo "ERROR: training failed at line $LINENO" >&2' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

HF_REPO="${HF_REPO:-mohamedemam/words-detection}"
HF_DEST="${HF_DEST:-artifacts/words-detection}"
HF_METHOD="${HF_METHOD:-auto}"
HF_TOKEN="${HF_TOKEN:-}"
HF_FORCE="${HF_FORCE:-0}"

MODEL="${MODEL:-yolo26s.pt}"
EPOCHS="${EPOCHS:-40}"
IMGSZ="${IMGSZ:-960}"
BATCH="${BATCH:-16}"
DEVICE="${DEVICE:-auto}"
WORKERS="${WORKERS:-4}"
NAME="${NAME:-yolo_words_hf}"
RUNS="${RUNS:-artifacts/yolo_runs}"

if ! command -v uv >/dev/null 2>&1; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

if [[ ! -x .venv/bin/python ]]; then
    echo "Creating .venv ..."
    uv venv --python 3.11 .venv
fi

echo "Installing the train and hf extras into .venv ..."
uv pip install --python "$PROJECT_ROOT/.venv/bin/python" -e "$PROJECT_ROOT[train,hf]"

download_args=(--repo-id "$HF_REPO" --dest "$HF_DEST" --method "$HF_METHOD")
if [[ -n "$HF_TOKEN" ]]; then
    download_args+=(--token "$HF_TOKEN")
fi
if [[ "$HF_FORCE" == "1" ]]; then
    download_args+=(--force)
fi

echo "Downloading $HF_REPO from the Hugging Face Hub ..."
uv run --no-sync python scripts/download_hf_dataset.py "${download_args[@]}"

if [[ ! -f "$HF_DEST/data.yaml" ]]; then
    echo "ERROR: $HF_DEST/data.yaml not found after download." >&2
    exit 1
fi

echo "Training on $HF_DEST/data.yaml (device=$DEVICE, imgsz=$IMGSZ, epochs=$EPOCHS) ..."
uv run --no-sync python scripts/train_yolo_words.py \
    --data "$HF_DEST/data.yaml" \
    --model "$MODEL" \
    --epochs "$EPOCHS" \
    --imgsz "$IMGSZ" \
    --batch "$BATCH" \
    --device "$DEVICE" \
    --workers "$WORKERS" \
    --project "$RUNS" \
    --name "$NAME"

echo "Best checkpoint: $RUNS/$NAME/weights/best.pt"
echo "Evaluate it with:"
echo "  uv run --no-sync python scripts/eval_yolo_words.py \\"
echo "      --weights $RUNS/$NAME/weights/best.pt --data $HF_DEST/data.yaml --per-source"
