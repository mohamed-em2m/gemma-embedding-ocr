#!/usr/bin/env bash
# ==============================================================================
# EmbeddingOCR: Environment Setup Script
# Reproduces the full development and runtime environment.
# Supports Fedora/RHEL (dnf) and Debian/Ubuntu (apt-get).
# Uses uv exclusively as specified in AGENTS.md.
# ==============================================================================

set -euo pipefail

echo "=========================================================="
echo " Starting EmbeddingOCR Environment Setup..."
echo "=========================================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}") && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# ---------------------------------------------------------------------------
# Detect package manager
# ---------------------------------------------------------------------------
if command -v dnf &>/dev/null; then
    PKG_MANAGER="dnf"
elif command -v apt-get &>/dev/null; then
    PKG_MANAGER="apt"
else
    echo "ERROR: Neither dnf nor apt-get found. Install system packages manually." >&2
    exit 1
fi
echo "Detected package manager: $PKG_MANAGER"

# ---------------------------------------------------------------------------
# 1. System packages: Tesseract OCR + clean Latin fonts
# ---------------------------------------------------------------------------
echo ""
echo "[1/4] Installing system dependencies, Tesseract OCR, and fonts..."

if [[ "$PKG_MANAGER" == "dnf" ]]; then
    # Fedora/RHEL/CentOS package names
    dnf install -y -q \
        git curl wget \
        fontconfig \
        tesseract \
        tesseract-langpack-eng \
        liberation-sans-fonts \
        liberation-serif-fonts \
        liberation-mono-fonts \
        google-carlito-fonts \
        open-sans \
        abattis-cantarell-fonts \
        abattis-cantarell-vf-fonts \
        aajohan-comfortaa-fonts \
        adobe-source-code-pro-fonts \
        google-droid-sans-fonts \
        julietaula-montserrat-fonts \
        google-noto \
        google-noto-vf \
        google-noto-sans-mono-vf-fonts \
        2>/dev/null || true

    # Optional: urw-base35 provides URW Nimbus (Helvetica/Times/Courier equivalents)
    dnf install -y -q urw-base35-fonts 2>/dev/null || true

else
    # Debian/Ubuntu package names
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq
    apt-get install -y -qq \
        git curl wget \
        fontconfig \
        tesseract-ocr \
        tesseract-ocr-eng \
        fonts-liberation \
        fonts-liberation2 \
        fonts-dejavu-core \
        fonts-dejavu-extra \
        fonts-noto-core \
        fonts-noto-mono \
        fonts-open-sans \
        fonts-cantarell \
        fonts-comfortaa \
        fonts-inconsolata \
        fonts-ubuntu \
        fonts-roboto \
        fonts-freefont-ttf
fi

fc-cache -f -v >/dev/null 2>&1
echo "✓ System dependencies, Tesseract, and fonts installed."

# ---------------------------------------------------------------------------
# 2. Install / ensure uv is present
# ---------------------------------------------------------------------------
echo ""
echo "[2/4] Setting up uv package manager..."
if ! command -v uv &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Make uv available in this session
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi

# Verify uv is now reachable (error clearly if still missing)
if ! command -v uv &>/dev/null; then
    echo "ERROR: uv installation succeeded but 'uv' is not on PATH." >&2
    echo "  Add one of these to your shell profile and re-run:" >&2
    echo "    export PATH=\"\$HOME/.local/bin:\$PATH\"" >&2
    echo "    export PATH=\"\$HOME/.cargo/bin:\$PATH\"" >&2
    exit 1
fi
echo "✓ uv is ready: $(uv --version)"

# ---------------------------------------------------------------------------
# 3. Synchronize Python environment
# ---------------------------------------------------------------------------
echo ""
echo "[3/4] Installing Python dependencies via uv sync..."
uv sync --group dev
echo "✓ Python environment ready."

# ---------------------------------------------------------------------------
# 4. Install flash_attn compatibility shim (on-disk fallback)
#    NOTE: dots_mocr.py also installs this shim in-process at runtime.
#    The on-disk version covers other code paths that import flash_attn
#    before DotsMOCRRecognizer.__init__ has had a chance to run.
# ---------------------------------------------------------------------------
echo ""
echo "[4/4] Setting up flash_attn SDPA compatibility shim..."

# Write the shim via a here-doc so quoting is unambiguous
uv run python - <<'PYEOF'
import importlib.util
import sys
from pathlib import Path

spec = importlib.util.find_spec("flash_attn")
if spec is not None and spec.origin is not None:
    print("✓ flash_attn already resolved:", spec.origin)
    sys.exit(0)

# Locate the active venv site-packages directory
site_dirs = [p for p in sys.path if "site-packages" in p]
if not site_dirs:
    print("WARNING: Could not locate site-packages; skipping on-disk flash_attn shim.")
    sys.exit(0)

target_dir = Path(site_dirs[0]) / "flash_attn"
target_dir.mkdir(parents=True, exist_ok=True)

shim_code = '''\
"""flash_attn compatibility shim.

This stub satisfies ``check_imports("flash_attn")`` / ``is_flash_attn_available()``
checks in HuggingFace remote code (e.g. dots-studio/dots.mocr).
The actual attention backend is overridden to ``sdpa`` or ``eager`` at load time,
so none of these stub functions are ever called.
"""
__version__ = "2.0.0.stub"


def flash_attn_varlen_func(*args, **kwargs):
    raise NotImplementedError("flash_attn CUDA kernels are not compiled; use SDPA attention.")


def flash_attn_func(*args, **kwargs):
    raise NotImplementedError("flash_attn CUDA kernels are not compiled; use SDPA attention.")
'''

(target_dir / "__init__.py").write_text(shim_code)
print("✓ flash_attn compatibility shim installed to", target_dir)
PYEOF

# ---------------------------------------------------------------------------
# 5. Build clean Latin-font catalog for the benchmark scripts
# ---------------------------------------------------------------------------
echo ""
echo "Indexing clean document fonts..."
mkdir -p data

uv run python - <<'PYEOF'
from pathlib import Path

font_root = Path("/usr/share/fonts")
all_fonts = sorted(set(
    str(p)
    for p in (*font_root.rglob("*.ttf"), *font_root.rglob("*.otf"))
))

# Keywords that identify non-Latin / symbol / emoji fonts to exclude
_EXCLUDE = {
    "tifinagh", "taitham", "armenian", "rohingya", "cherokee",
    "balinese", "sinhala", "carian", "hebrew", "arabic", "siddham",
    "bengali", "devanagari", "ethiopic", "gurmukhi", "takri", "thai",
    "khmer", "lao", "myanmar", "cham", "sogdian", "multani", "linear",
    "sundanese", "meetei", "emoji", "symbol", "cjk", "japanese",
    "chinese", "korean", "tibetan", "georgian", "tamil", "kannada",
    "telugu", "oriya", "sinhala", "mongolian", "buginese",
}

# Keywords that identify clean Latin document fonts to keep
_KEEP = {
    "liberation", "dejavu", "roboto", "noto.?sans[^-]", "notosans-",
    "noto.?serif[^-]", "notoserif-", "ubuntu", "lato", "opensans",
    "open.?sans", "cascadia", "firacode", "inter", "inconsolata",
    "cantarell", "courier", "carlito", "source.?code", "montserrat",
    "droid.?sans[^a-z]", "comfortaa", "caladea", "adwaita",
    "noto.?sans.?mono", "noto.?mono",
}

import re

keep_rx = re.compile("|".join(_KEEP), re.IGNORECASE)
excl_rx = re.compile("|".join(_EXCLUDE), re.IGNORECASE)

filtered = [
    p for p in all_fonts
    if not excl_rx.search(p) and keep_rx.search(p)
]

out = Path("data/standard_clean_fonts.txt")
out.write_text("\n".join(filtered) + "\n")
print(f"Successfully cataloged {len(filtered)} verified clean document fonts in {out}")
PYEOF

echo ""
echo "=========================================================="
echo " EmbeddingOCR environment successfully configured!"
echo ""
echo " Run unit tests:   uv run pytest"
echo " Run linter:       uv run ruff check ."
echo " Run type checker: uv run mypy src"
echo " Run benchmark:    uv run python scripts/compare_new_ocr_models.py"
echo "=========================================================="
