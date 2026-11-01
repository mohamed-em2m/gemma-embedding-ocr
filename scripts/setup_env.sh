#!/usr/bin/env bash
# ==============================================================================
# EmbeddingOCR: Environment Setup Script
# Reproduces the full development and runtime environment on Debian/Ubuntu/Colab
# ==============================================================================

set -euo pipefail

echo "=========================================================="
echo " Starting EmbeddingOCR Environment Setup..."
echo "=========================================================="

# 1. System packages & 100+ typography font families
echo "[1/4] Installing system dependencies and clean fonts..."
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq \
    git curl wget \
    fontconfig \
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
echo "✓ System dependencies & fonts installed."

# 2. Install / ensure uv is present
echo "[2/4] Setting up uv package manager..."
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
echo "✓ uv is ready: $(uv --version)"

# 3. Python libraries
echo "[3/4] Installing core AI, vision, and vector search packages..."
python3 -m pip install --upgrade pip -q
python3 -m pip install -q \
    torch \
    transformers \
    "sentence-transformers>=6.1.0" \
    "faiss-cpu>=1.8.0" \
    pillow \
    numpy \
    pyyaml \
    pydantic \
    typer \
    rich \
    jiwer \
    rapidfuzz \
    opencv-python-headless \
    ruff \
    mypy \
    pytest \
    pytest-cov

# 4. Install local package in editable mode
echo "[4/4] Installing embedding-ocr in editable mode..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"
python3 -m pip install -e . -q

# 5. Build clean 166-font catalog
echo "Indexing clean document fonts..."
mkdir -p data
python3 -c "
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
echo " Run unit tests with:  pytest tests/"
echo " Run spike with:       python scripts/phase0_az_spike.py"
echo "=========================================================="
