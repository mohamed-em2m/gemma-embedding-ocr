#!/usr/bin/env bash
# ==============================================================================
# EmbeddingOCR: Environment Setup Script
# Reproduces the full development and runtime environment on Debian/Ubuntu/Colab
# Uses uv exclusively as specified in AGENTS.md
# ==============================================================================

set -euo pipefail

echo "=========================================================="
echo " Starting EmbeddingOCR Environment Setup..."
echo "=========================================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

# 1. System packages, tesseract OCR, & clean typography font families
echo "[1/4] Installing system dependencies, Tesseract OCR, and clean fonts..."
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq \
    git curl wget \
    fontconfig \
    tesseract-ocr \
    fonts-dejavu-core fonts-dejavu-extra \
    fonts-noto-core fonts-noto-mono \
    fonts-roboto \
    fonts-liberation fonts-liberation2 \
    fonts-freefont-ttf \
    fonts-ubuntu \
    fonts-lato \
    fonts-open-sans \
    fonts-cantarell \
    fonts-inconsolata \
    fonts-comfortaa \
    fonts-cascadia-code \
    fonts-firacode \
    fonts-courier-prime \
    fonts-ebgaramond \
    fonts-cabin \
    fonts-quicksand \
    fonts-inter

fc-cache -f -v > /dev/null 2>&1
echo "✓ System dependencies, Tesseract, & fonts installed."

# 2. Install / ensure uv is present
echo "[2/4] Setting up uv package manager..."
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi
echo "✓ uv is ready: $(uv --version)"

# 3. Synchronize environment with uv (dev group included)
echo "[3/4] Installing Python dependencies via uv sync..."
uv sync --group dev

# 4. Setup flash_attn compatibility shim for Turing/T4 & environments without CUDA kernels
echo "[4/4] Setting up flash_attn SDPA compatibility shim..."
uv run python -c "
import importlib.util
from pathlib import Path
import site

spec = importlib.util.find_spec('flash_attn')
if spec is None or spec.origin is None:
    site_dirs = [p for p in site.getsitepackages() if 'site-packages' in p]
    target_dir = Path(site_dirs[0]) / 'flash_attn' if site_dirs else None
    if target_dir:
        target_dir.mkdir(parents=True, exist_ok=True)
        init_file = target_dir / '__init__.py'
        init_file.write_text(
            '__version__ = \"0.0.1\"\n\n'
            'def flash_attn_varlen_func(*args, **kwargs):\n'
            '    raise NotImplementedError(\"flash_attn CUDA kernels not compiled; use SDPA attention.\")\n'
        )
        print('✓ flash_attn compatibility shim installed to', target_dir)
else:
    print('✓ flash_attn already resolved:', spec.origin)
"

# 5. Build clean font catalog
echo "Indexing clean document fonts..."
mkdir -p data
uv run python -c "
from pathlib import Path

all_fonts = list(Path('/usr/share/fonts').rglob('*.ttf')) + list(Path('/usr/share/fonts').rglob('*.otf'))
standard_keywords = ['liberation', 'dejavu', 'roboto', 'notosans-', 'notoserif-', 'ubuntu', 'lato', 'opensans', 'cascadia', 'firacode', 'inter', 'inconsolata', 'cantarell', 'courier']

filtered = []
for p in all_fonts:
    name = str(p).lower()
    if any(x in name for x in ['tifinagh', 'taitham', 'armenian', 'rohingya', 'cherokee', 'balinese', 'sinhala', 'carian', 'hebrew', 'arabic', 'siddham', 'bengali', 'devanagari', 'ethiopic', 'gurmukhi', 'takri', 'thai', 'khmer', 'lao', 'myanmar', 'cham', 'sogdian', 'multani', 'linear', 'sundanese', 'meetei']):
        continue
    if any(k in name for k in standard_keywords):
        filtered.append(str(p))

filtered = sorted(list(set(filtered)))
with open('data/standard_clean_fonts.txt', 'w') as out:
    for f_path in filtered:
        out.write(f_path + '\n')

print(f'Successfully cataloged {len(filtered)} verified clean document fonts in data/standard_clean_fonts.txt')
"

echo "=========================================================="
echo " EmbeddingOCR environment successfully configured!"
echo " Run unit tests:  uv run pytest"
echo " Run linter:     uv run ruff check ."
echo " Run benchmark:  uv run python scripts/compare_new_ocr_models.py"
echo "=========================================================="
