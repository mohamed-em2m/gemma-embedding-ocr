#!/usr/bin/env bash
# Launch the Gradio UI from the repo's virtual environment.
#
# The script auto-detects ``.venv`` (preferred) or ``venv`` and repairs common
# Kaggle/Colab breakages without running ``uv sync`` (which can prune extras).
#
# Usage:
#   bash scripts/run_ui.sh
#   bash scripts/run_ui.sh --no-share
#   VENV=venv bash scripts/run_ui.sh --share
#   make ui UI_ARGS="--no-share"

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

find_venv() {
    if [[ -n "${VENV:-}" ]]; then
        if [[ "$VENV" = /* ]]; then
            printf '%s\n' "$VENV"
        else
            printf '%s\n' "$PROJECT_ROOT/$VENV"
        fi
        return 0
    fi

    local candidate
    for candidate in "$PROJECT_ROOT/.venv" "$PROJECT_ROOT/venv"; do
        if [[ -x "$candidate/bin/python" ]]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done
    return 1
}

venv_path="$(find_venv || true)"
if [[ -z "$venv_path" || ! -x "$venv_path/bin/python" ]]; then
    echo "ERROR: no virtual environment found." >&2
    echo "Expected $PROJECT_ROOT/.venv (or $PROJECT_ROOT/venv)." >&2
    echo "Run: bash scripts/setup_auto.sh" >&2
    exit 1
fi

python="$venv_path/bin/python"
cli="$venv_path/bin/embedding-ocr"

launcher=()
if [[ -x "$cli" ]]; then
    launcher=("$cli")
else
    # Fall back to the source tree when the console script was not generated
    # (for example after a minimal/temporary uv environment).
    export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
    launcher=("$python" -m embedding_ocr.cli)
fi

has_module() {
    "$python" -c "import $1" >/dev/null 2>&1
}

require_uv() {
    if ! command -v uv >/dev/null 2>&1; then
        local candidate
        for candidate in "${UV_ENV_FILE:-}" "${HOME:-}/.local/bin/env" "${HOME:-}/.cargo/env"; do
            if [[ -n "$candidate" && -f "$candidate" ]]; then
                # shellcheck disable=SC1090
                source "$candidate" >/dev/null 2>&1 || true
                command -v uv >/dev/null 2>&1 && break
            fi
        done
    fi
    if ! command -v uv >/dev/null 2>&1; then
        export PATH="${HOME:-}/.local/bin:${HOME:-}/.cargo/bin:$PATH"
    fi
    if ! command -v uv >/dev/null 2>&1; then
        echo "ERROR: uv is required to repair missing runtime packages." >&2
        echo "Install uv or run: bash scripts/setup_auto.sh" >&2
        exit 1
    fi
}

if ! has_module cv2; then
    echo "OpenCV import failed; repairing opencv-python-headless..."
    require_uv
    uv pip uninstall --python "$python" \
        opencv-python opencv-contrib-python >/dev/null 2>&1 || true
    uv pip install --python "$python" --reinstall \
        "opencv-python-headless>=4.11"
fi
if ! has_module cv2; then
    echo "ERROR: cv2 is still unavailable after repair." >&2
    echo "Check: uv pip list --python \"$python\" | grep -i opencv" >&2
    exit 1
fi

if ! has_module gradio; then
    echo "Gradio import failed; installing gradio..."
    require_uv
    uv pip install --python "$python" --reinstall "gradio>=5.0"
fi
if ! has_module gradio; then
    echo "ERROR: gradio is still unavailable after repair." >&2
    exit 1
fi

echo "Using environment: $venv_path"
exec "${launcher[@]}" ui "$@"
