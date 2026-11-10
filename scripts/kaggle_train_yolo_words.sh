#!/usr/bin/env bash
# Train the one-class YOLO word detector on Kaggle's 2x T4 GPU runtime.
#
# Run from a Kaggle notebook cell:
#   !bash scripts/kaggle_train_yolo_words.sh
#
# Dataset modes:
#   DATASET_MODE=synthetic  render synthetic words (default)
#   DATASET_MODE=real       merge only real Roboflow/YOLO sources
#   DATASET_MODE=mixed      merge real sources + synthetic words
#   DATASET_MODE=hf         download the published dataset from the Hugging Face
#                           Hub (default: mohamedemam/words-detection) and train
#
# Hugging Face mode needs no Roboflow key and no local data:
#   !DATASET_MODE=hf bash scripts/kaggle_train_yolo_words.sh
#   !DATASET_MODE=hf HF_REPO=other-user/other-dataset EPOCHS=60 \
#    bash scripts/kaggle_train_yolo_words.sh
#
# Real sources can come from configs/real_word_sources.yaml and/or a
# colon-separated REAL_SOURCES list of extracted exports or .zip files:
#   !DATASET_MODE=mixed \
#    REAL_SOURCES=/kaggle/input/words-a:/kaggle/input/words-b \
#    bash scripts/kaggle_train_yolo_words.sh
#
# Download Roboflow sources first (needs a free API key):
#   !DATASET_MODE=real DOWNLOAD=1 ROBOFLOW_API_KEY=xxxx \
#    bash scripts/kaggle_train_yolo_words.sh
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

# Dataset construction.
DATASET_MODE="${DATASET_MODE:-synthetic}"          # synthetic | real | mixed | hf
REAL_CONFIG="${REAL_CONFIG:-}"                     # YAML source list
REAL_SOURCES="${REAL_SOURCES:-}"                   # colon-separated dirs/zips
DOWNLOAD="${DOWNLOAD:-0}"                          # 1 = download Roboflow sources
ROBOFLOW_API_KEY="${ROBOFLOW_API_KEY:-}"
SYNTHETIC_N="${SYNTHETIC_N:-${N:-30000}}"          # N kept as a legacy alias
OVERWRITE="${OVERWRITE:-1}"                        # rebuild the merged dataset
SYNTH_DIR="${SYNTH_DIR:-$WORK_DIR/yolo_words_synthetic}"

# Hugging Face dataset source (DATASET_MODE=hf).
HF_REPO="${HF_REPO:-mohamedemam/words-detection}"
HF_DEST="${HF_DEST:-$WORK_DIR/words-detection}"
HF_METHOD="${HF_METHOD:-auto}"                     # auto | hub | git
HF_TOKEN="${HF_TOKEN:-}"
HF_FORCE="${HF_FORCE:-0}"                          # 1 = re-download the dataset

# Use the repo's source list only when the caller did not provide sources.
if [[ -z "$REAL_CONFIG" && -z "$REAL_SOURCES" ]]; then
    REAL_CONFIG="configs/real_word_sources.yaml"
fi

if [[ "$DATASET_MODE" != "synthetic" && "$DATASET_MODE" != "real" && "$DATASET_MODE" != "mixed" && "$DATASET_MODE" != "hf" ]]; then
    echo "ERROR: DATASET_MODE must be synthetic, real, mixed or hf (got $DATASET_MODE)" >&2
    exit 2
fi

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
uv pip install --python "$PROJECT_ROOT/.venv/bin/python" -e "$PROJECT_ROOT[train,hf]"
if ! uv run --no-sync python -c "import ultralytics, uharfbuzz" >/dev/null 2>&1; then
    echo "ERROR: ultralytics/uharfbuzz are not importable from .venv." >&2
    exit 1
fi

if [[ "$DOWNLOAD" == "1" ]]; then
    if [[ -z "$ROBOFLOW_API_KEY" ]]; then
        echo "ERROR: DOWNLOAD=1 needs ROBOFLOW_API_KEY." >&2
        exit 2
    fi
    if [[ -z "$REAL_CONFIG" || ! -f "$REAL_CONFIG" ]]; then
        echo "ERROR: DOWNLOAD=1 needs a REAL_CONFIG file." >&2
        exit 2
    fi
    echo "Installing the Roboflow SDK..."
    uv pip install --python "$PROJECT_ROOT/.venv/bin/python" roboflow
    download_args=(--config "$REAL_CONFIG" --api-key "$ROBOFLOW_API_KEY")
    if [[ "$OVERWRITE" == "1" ]]; then
        download_args+=(--overwrite)
    fi
    uv run --no-sync python scripts/download_roboflow_sources.py "${download_args[@]}"
fi

merge_args=(--out "$DATASET")
if [[ -f "$REAL_CONFIG" ]]; then
    merge_args+=(--config "$REAL_CONFIG")
fi
if [[ -n "$REAL_SOURCES" ]]; then
    IFS=':' read -r -a source_parts <<< "$REAL_SOURCES"
    for source in "${source_parts[@]}"; do
        if [[ -n "$source" ]]; then
            merge_args+=(--source "$source")
        fi
    done
fi
if [[ "$OVERWRITE" == "1" ]]; then
    merge_args+=(--overwrite)
fi

build_synthetic() {
    echo "Generating synthetic word dataset at $SYNTH_DIR ..."
    uv run --no-sync python scripts/build_word_dataset.py \
        --fonts "$FONTS" \
        --out "$SYNTH_DIR" \
        --n "$SYNTHETIC_N" \
        --size "$IMGSZ" \
        --layout mixed
}

case "$DATASET_MODE" in
    synthetic)
        echo "Generating synthetic word dataset at $DATASET ..."
        uv run --no-sync python scripts/build_word_dataset.py \
            --fonts "$FONTS" \
            --out "$DATASET" \
            --n "$SYNTHETIC_N" \
            --size "$IMGSZ" \
            --layout mixed
        ;;
    real)
        if [[ ! -f "$REAL_CONFIG" && -z "$REAL_SOURCES" ]]; then
            echo "ERROR: DATASET_MODE=real needs REAL_CONFIG or REAL_SOURCES." >&2
            exit 2
        fi
        echo "Merging real word datasets into $DATASET ..."
        uv run --no-sync python scripts/build_real_word_dataset.py "${merge_args[@]}"
        ;;
    mixed)
        build_synthetic
        merge_args+=(--source "$SYNTH_DIR")
        echo "Merging real + synthetic word datasets into $DATASET ..."
        uv run --no-sync python scripts/build_real_word_dataset.py "${merge_args[@]}"
        ;;
    hf)
        hf_args=(--repo-id "$HF_REPO" --dest "$HF_DEST" --method "$HF_METHOD")
        if [[ -n "$HF_TOKEN" ]]; then
            hf_args+=(--token "$HF_TOKEN")
        fi
        if [[ "${HF_FORCE:-0}" == "1" ]]; then
            hf_args+=(--force)
        fi
        echo "Downloading $HF_REPO from the Hugging Face Hub ..."
        uv run --no-sync python scripts/download_hf_dataset.py "${hf_args[@]}"
        DATASET="$HF_DEST"
        ;;
esac

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
