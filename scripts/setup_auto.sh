#!/usr/bin/env bash
# ==============================================================================
# EmbeddingOCR: automatic Linux / Kaggle / Colab setup
#
# Installs the system libraries and Python extras needed for the Gradio UI,
# Arabic text shaping, and the reproducible benchmark scripts. Safe to re-run.
#
# Usage:
#   bash scripts/setup_auto.sh
#   bash scripts/setup_auto.sh --launch          # start the public Gradio UI
#   bash scripts/setup_auto.sh --launch --no-share
#   bash scripts/setup_auto.sh --with-paddle     # also install the PaddleOCR baseline
#   bash scripts/setup_auto.sh --no-system       # skip apt/dnf packages
#   bash scripts/setup_auto.sh --check           # run ruff/mypy/pytest afterwards
# ==============================================================================

set -Eeuo pipefail
trap 'echo "ERROR: setup failed at line $LINENO" >&2' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

INSTALL_SYSTEM=1
INSTALL_PADDLE=0
BUILD_PROTOTYPES=0
RUN_CHECKS=0
LAUNCH=0
SHARE=1

usage() {
    cat <<'EOF'
Usage: bash scripts/setup_auto.sh [OPTIONS]

Options:
  --launch            Start the Gradio UI when setup finishes.
  --no-share          Keep the UI local (default is a public Gradio link).
  --with-paddle       Also install the optional PaddleOCR baseline.
  --build-prototypes  Build the Latin character prototype cache.
  --no-system         Skip apt/dnf system packages.
  --check             Run ruff, mypy, and pytest after setup.
  -h, --help          Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --launch) LAUNCH=1 ;;
        --no-share) SHARE=0 ;;
        --with-paddle) INSTALL_PADDLE=1 ;;
        --build-prototypes) BUILD_PROTOTYPES=1 ;;
        --no-system) INSTALL_SYSTEM=0 ;;
        --check) RUN_CHECKS=1 ;;
        -h|--help) usage; exit 0 ;;
        *)
            echo "ERROR: unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
    shift
done

step=0
run_step() {
    step=$((step + 1))
    echo ""
    echo "[$step] $*"
}

SUDO=()
if [[ ${EUID} -ne 0 ]] && command -v sudo >/dev/null 2>&1; then
    SUDO=(sudo)
fi

echo "=========================================================="
echo " EmbeddingOCR automatic setup"
echo " Project: $PROJECT_ROOT"
echo "=========================================================="

# ---------------------------------------------------------------------------
# 1. System libraries: OpenCV runtime, Arabic shaping (libraqm), fonts
# ---------------------------------------------------------------------------
run_step "Installing system libraries..."
if [[ $INSTALL_SYSTEM -eq 1 ]]; then
    if command -v apt-get >/dev/null 2>&1; then
        export DEBIAN_FRONTEND=noninteractive
        "${SUDO[@]}" apt-get update -qq || true

        CORE_PACKAGES=(libgl1 libglib2.0-0 fontconfig)
        for package in "${CORE_PACKAGES[@]}"; do
            "${SUDO[@]}" apt-get install -y -qq "$package" >/dev/null 2>&1 \
                || echo "  - optional package unavailable: $package"
        done

        if ! "${SUDO[@]}" apt-get install -y -qq libraqm0 >/dev/null 2>&1; then
            "${SUDO[@]}" apt-get install -y -qq libraqm0t64 >/dev/null 2>&1 \
                || echo "  - libraqm unavailable; Pillow will use the shaping fallback"
        fi

        FONT_PACKAGES=(
            fonts-dejavu-core fonts-liberation fonts-noto-core
            fonts-noto-naskh-arabic fonts-noto-sans-arabic
            fonts-vazirmatn fonts-sil-scheherazade
        )
        for package in "${FONT_PACKAGES[@]}"; do
            "${SUDO[@]}" apt-get install -y -qq "$package" >/dev/null 2>&1 \
                || echo "  - optional package unavailable: $package"
        done
        fc-cache -f >/dev/null 2>&1 || true
        echo "✓ System libraries and fonts checked."
    elif command -v dnf >/dev/null 2>&1; then
        "${SUDO[@]}" dnf install -y -q libraqm fontconfig \
            liberation-sans-fonts google-noto-vf \
            google-noto-naskh-arabic-vf-fonts vazirmatn-vf-fonts \
            >/dev/null 2>&1 || true
        fc-cache -f >/dev/null 2>&1 || true
        echo "✓ System libraries and fonts checked."
    else
        echo "WARNING: no apt-get or dnf found; skipping system packages."
    fi
else
    echo "Skipped (--no-system)."
fi

# ---------------------------------------------------------------------------
# 2. uv
# ---------------------------------------------------------------------------
run_step "Ensuring uv is installed..."
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi
command -v uv >/dev/null 2>&1 || {
    echo "ERROR: uv is not on PATH." >&2
    echo 'Add: export PATH="$HOME/.local/bin:$PATH"' >&2
    exit 1
}
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
echo "✓ $(uv --version)"

# ---------------------------------------------------------------------------
# 3. Python environment: one exact sync with every UI/benchmark extra
# ---------------------------------------------------------------------------
run_step "Syncing Python dependencies..."
EXTRAS=(tesseract easyocr experiments arabic ui)
if [[ $INSTALL_PADDLE -eq 1 ]]; then
    EXTRAS+=(baseline)
fi

SYNC_ARGS=(sync --locked --group dev)
for extra in "${EXTRAS[@]}"; do
    SYNC_ARGS+=(--extra "$extra")
done
echo "Extras: ${EXTRAS[*]}"
uv "${SYNC_ARGS[@]}"
echo "✓ Python environment ready."

# ---------------------------------------------------------------------------
# 4. Verify / repair imports that commonly break after a uv sync
# ---------------------------------------------------------------------------
run_step "Verifying runtime imports..."
PYTHON="$PROJECT_ROOT/.venv/bin/python"
[[ -x "$PYTHON" ]] || {
    echo "ERROR: $PYTHON not found after sync." >&2
    exit 1
}

if ! "$PYTHON" -c "import cv2" >/dev/null 2>&1; then
    echo "  - OpenCV import failed; forcing a clean headless reinstall..."
    uv pip uninstall --python "$PYTHON" opencv-python opencv-contrib-python         >/dev/null 2>&1 || true
    uv pip install --python "$PYTHON" --reinstall "opencv-python-headless>=4.11"
fi
"$PYTHON" -c "import cv2" >/dev/null 2>&1 || {
    echo "ERROR: cv2 is still unavailable after reinstall." >&2
    echo "Check that opencv-python and opencv-contrib-python are not installed." >&2
    exit 1
}

if ! "$PYTHON" -c "import gradio" >/dev/null 2>&1; then
    echo "  - Gradio import failed; forcing a reinstall..."
    uv pip install --python "$PYTHON" --reinstall "gradio>=5.0"
fi
"$PYTHON" -c "import gradio" >/dev/null 2>&1 || {
    echo "ERROR: Gradio is still unavailable after reinstall." >&2
    exit 1
}

if ! "$PYTHON" -c "import arabic_reshaper, bidi" >/dev/null 2>&1; then
    echo "  - Arabic shaping helpers missing; installing them..."
    uv pip install --python "$PYTHON" --reinstall arabic-reshaper python-bidi
fi

PYTHONPATH="$PROJECT_ROOT/src" "$PYTHON" - <<'PY'
import importlib.metadata

import cv2
from PIL import features

print(f"✓ cv2 {cv2.__version__}")
print(f"✓ gradio {importlib.metadata.version('gradio')}")
if features.check("raqm"):
    print("✓ Pillow libraqm shaping enabled")
else:
    print("! Pillow libraqm unavailable; using arabic-reshaper/python-bidi fallback")
PY

# ---------------------------------------------------------------------------
# 5. Optional prototype cache
# ---------------------------------------------------------------------------
if [[ $BUILD_PROTOTYPES -eq 1 ]]; then
    run_step "Building Latin prototype cache..."
    PYTHONPATH="$PROJECT_ROOT/src" "$PYTHON" scripts/build_prototype_cache.py \
        --num-fonts 25 --dim 768 --output-dir artifacts/prototypes
    echo "✓ Prototype cache written to artifacts/prototypes/clean_alnum_768d"
else
    run_step "Prototype cache..."
    echo "Skipped; pass --build-prototypes for embedding_seg.yaml."
fi

# ---------------------------------------------------------------------------
# 6. Optional checks
# ---------------------------------------------------------------------------
if [[ $RUN_CHECKS -eq 1 ]]; then
    run_step "Running validation..."
    "$PROJECT_ROOT/.venv/bin/ruff" check .
    "$PROJECT_ROOT/.venv/bin/ruff" format --check .
    "$PROJECT_ROOT/.venv/bin/mypy" src
    PYTHONPATH="$PROJECT_ROOT/src" "$PROJECT_ROOT/.venv/bin/pytest" -q
    echo "✓ Validation passed."
fi

# ---------------------------------------------------------------------------
# 7. Launch
# ---------------------------------------------------------------------------
if [[ $LAUNCH -eq 1 ]]; then
    run_step "Launching Gradio UI..."
    UI_ARGS=(ui)
    if [[ $SHARE -eq 1 ]]; then
        UI_ARGS+=(--share)
    else
        UI_ARGS+=(--no-share)
    fi
    exec "$PROJECT_ROOT/.venv/bin/embedding-ocr" "${UI_ARGS[@]}"
fi

echo ""
echo "=========================================================="
echo " Setup complete."
echo ""
echo " Launch UI:          bash scripts/setup_auto.sh --launch"
echo " Arabic profile:     embedding_align.yaml"
echo " Arabic recognizer:  hierarchical"
echo " Arabic detector:    full   (use visual for multi-word lines)"
echo " Arabic script:      arabic"
echo "=========================================================="
