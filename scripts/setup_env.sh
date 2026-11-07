#!/usr/bin/env bash
# ==============================================================================
# EmbeddingOCR: one-shot environment setup
#
# Installs system packages/fonts, uv, all core Python extras, the flash_attn
# compatibility shim, font catalogs, and git hooks in a single run.
#
# Usage:
#   bash scripts/setup_env.sh                  # full setup + core extras
#   bash scripts/setup_env.sh --all-extras     # also jina/craft/surya
#   bash scripts/setup_env.sh --no-system      # skip dnf/apt packages
#   bash scripts/setup_env.sh --no-extras      # core project deps only
#   bash scripts/setup_env.sh --check          # run lint/types/tests at the end
# ==============================================================================

set -Eeuo pipefail
trap 'echo "ERROR: setup failed at line $LINENO" >&2' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

INSTALL_SYSTEM=1
INSTALL_EXTRAS=1
ALL_EXTRAS=0
INSTALL_SHIM=1
BUILD_CATALOGS=1
INSTALL_HOOKS=1
RUN_CHECKS=0

usage() {
    cat <<'EOF'
Usage: bash scripts/setup_env.sh [OPTIONS]

Options:
  --no-system     Skip system package/font installation.
  --no-extras     Install only core project dependencies.
  --all-extras    Also install jina, craft, and surya extras.
  --no-shim       Skip the flash_attn compatibility shim.
  --no-catalogs   Skip font catalog generation.
  --no-hooks      Skip pre-commit hook installation.
  --check         Run ruff, mypy, and pytest after setup.
  -h, --help      Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-system) INSTALL_SYSTEM=0 ;;
        --no-extras) INSTALL_EXTRAS=0 ;;
        --all-extras) ALL_EXTRAS=1 ;;
        --no-shim) INSTALL_SHIM=0 ;;
        --no-catalogs) BUILD_CATALOGS=0 ;;
        --no-hooks) INSTALL_HOOKS=0 ;;
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
total=7
run_step() {
    step=$((step + 1))
    echo ""
    echo "[$step/$total] $*"
}

SUDO=()
if [[ $INSTALL_SYSTEM -eq 1 && ${EUID} -ne 0 ]]; then
    if command -v sudo >/dev/null 2>&1; then
        SUDO=(sudo)
    else
        echo "ERROR: run as root or install sudo, or pass --no-system." >&2
        exit 1
    fi
fi

echo "=========================================================="
echo " Starting EmbeddingOCR one-shot environment setup..."
echo "=========================================================="

# ---------------------------------------------------------------------------
# 1. System packages: OCR engines, build tools, Latin/Arabic/handwriting fonts
# ---------------------------------------------------------------------------
run_step "Installing system dependencies and fonts..."
if [[ $INSTALL_SYSTEM -eq 1 ]]; then
    if command -v dnf >/dev/null 2>&1; then
        PKG_MANAGER="dnf"
    elif command -v apt-get >/dev/null 2>&1; then
        PKG_MANAGER="apt"
    else
        echo "ERROR: neither dnf nor apt-get found. Use --no-system or install manually." >&2
        exit 1
    fi
    echo "Detected package manager: $PKG_MANAGER"

    if [[ "$PKG_MANAGER" == "dnf" ]]; then
        CORE_PACKAGES=(
            git curl wget fontconfig
            tesseract tesseract-langpack-eng
        )
        OPTIONAL_PACKAGES=(
            liberation-sans-fonts liberation-serif-fonts liberation-mono-fonts
            google-carlito-fonts open-sans
            abattis-cantarell-fonts abattis-cantarell-vf-fonts
            aajohan-comfortaa-fonts adobe-source-code-pro-fonts
            google-droid-sans-fonts julietaula-montserrat-fonts
            google-noto google-noto-vf google-noto-sans-mono-vf-fonts
            urw-base35-fonts
            google-noto-naskh-arabic-vf-fonts google-noto-sans-arabic-vf-fonts
            paktype-naskh-basic-fonts vazirmatn-vf-fonts
            dancing-script-fonts kaushan-script-fonts kristi-fonts sj-fonts
        )
        if ! "${SUDO[@]}" dnf install -y -q --skip-broken "${CORE_PACKAGES[@]}"; then
            echo "ERROR: failed to install core dnf packages." >&2
            exit 1
        fi
        for package in "${OPTIONAL_PACKAGES[@]}"; do
            "${SUDO[@]}" dnf install -y -q "$package" >/dev/null 2>&1 \
                || echo "  - optional package unavailable: $package"
        done
    else
        export DEBIAN_FRONTEND=noninteractive
        "${SUDO[@]}" apt-get update -qq
        CORE_PACKAGES=(
            git curl wget fontconfig
            tesseract-ocr tesseract-ocr-eng
        )
        OPTIONAL_PACKAGES=(
            fonts-liberation fonts-liberation2
            fonts-dejavu-core fonts-dejavu-extra
            fonts-noto-core fonts-noto-mono
            fonts-open-sans fonts-cantarell fonts-comfortaa
            fonts-inconsolata fonts-ubuntu fonts-roboto fonts-freefont-ttf
            fonts-noto-naskh-arabic fonts-noto-sans-arabic
            fonts-vazirmatn fonts-hosny-amiri fonts-sil-scheherazade
            fonts-dancingscript fonts-kaushanscript fonts-kristi fonts-sjfonts
        )
        if ! "${SUDO[@]}" apt-get install -y -qq "${CORE_PACKAGES[@]}"; then
            echo "ERROR: failed to install core apt packages." >&2
            exit 1
        fi
        for package in "${OPTIONAL_PACKAGES[@]}"; do
            "${SUDO[@]}" apt-get install -y -qq "$package" >/dev/null 2>&1 \
                || echo "  - optional package unavailable: $package"
        done
    fi

    fc-cache -f >/dev/null 2>&1 || true
    echo "✓ System dependencies and fonts installed."
else
    echo "Skipped (--no-system)."
fi

# ---------------------------------------------------------------------------
# 2. Install / ensure uv is present
# ---------------------------------------------------------------------------
run_step "Setting up the uv package manager..."
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi
if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: uv was installed but is not on PATH." >&2
    echo "Add one of these to your shell profile and re-run:" >&2
    echo "  export PATH=\"\$HOME/.local/bin:\$PATH\"" >&2
    echo "  export PATH=\"\$HOME/.cargo/bin:\$PATH\"" >&2
    exit 1
fi
echo "✓ uv is ready: $(uv --version)"

# ---------------------------------------------------------------------------
# 3. Synchronize Python environment (core + selected extras in one command)
# ---------------------------------------------------------------------------
run_step "Installing Python dependencies via uv sync..."
SYNC_ARGS=(sync --locked --group dev)
if [[ $INSTALL_EXTRAS -eq 1 ]]; then
    EXTRAS=(baseline tesseract easyocr experiments arabic)
    if [[ $ALL_EXTRAS -eq 1 ]]; then
        EXTRAS+=(jina craft surya)
    fi
    for extra in "${EXTRAS[@]}"; do
        SYNC_ARGS+=(--extra "$extra")
    done
    echo "Installing extras: ${EXTRAS[*]}"
else
    echo "Installing core project dependencies only (--no-extras)."
fi
uv "${SYNC_ARGS[@]}"
echo "✓ Python environment ready."

# ---------------------------------------------------------------------------
# 4. flash_attn compatibility shim
# ---------------------------------------------------------------------------
run_step "Setting up the flash_attn SDPA compatibility shim..."
if [[ $INSTALL_SHIM -eq 1 ]]; then
    uv run --no-sync python - <<'PYEOF'
import importlib.util
import sys
from pathlib import Path

spec = importlib.util.find_spec("flash_attn")
if spec is not None and spec.origin is not None:
    print("✓ flash_attn already resolved:", spec.origin)
    sys.exit(0)

site_dirs = [p for p in sys.path if "site-packages" in p]
if not site_dirs:
    print("WARNING: could not locate site-packages; skipping shim.")
    sys.exit(0)

target_dir = Path(site_dirs[0]) / "flash_attn"
target_dir.mkdir(parents=True, exist_ok=True)
(target_dir / "__init__.py").write_text(
    '''"""flash_attn compatibility shim.

Satisfies import checks in HuggingFace remote code (for example
dots-studio/dots.mocr). Attention is forced to SDPA/eager, so these stubs are
never called.
"""
__version__ = "2.0.0.stub"


def flash_attn_varlen_func(*args, **kwargs):
    raise NotImplementedError("flash_attn is not compiled; use SDPA attention.")


def flash_attn_func(*args, **kwargs):
    raise NotImplementedError("flash_attn is not compiled; use SDPA attention.")
'''
)
print("✓ flash_attn compatibility shim installed to", target_dir)
PYEOF
else
    echo "Skipped (--no-shim)."
fi

# ---------------------------------------------------------------------------
# 5. Build Latin, Arabic, and handwriting font catalogs
# ---------------------------------------------------------------------------
run_step "Indexing installed fonts..."
if [[ $BUILD_CATALOGS -eq 1 ]]; then
    mkdir -p data
    uv run --no-sync python - <<'PYEOF'
import re
from pathlib import Path

font_root = Path("/usr/share/fonts")
fonts = sorted(
    {
        str(path)
        for pattern in ("*.ttf", "*.otf")
        for path in font_root.rglob(pattern)
    }
)

latin_keep = re.compile(
    r"liberation|dejavu|roboto|noto.?sans[^-]|notosans-|noto.?serif[^-]|"
    r"notoserif-|ubuntu|lato|opensans|open.?sans|cascadia|firacode|inter|"
    r"inconsolata|cantarell|courier|carlito|source.?code|montserrat|"
    r"droid.?sans[^a-z]|comfortaa|caladea|adwaita|noto.?sans.?mono|noto.?mono",
    re.IGNORECASE,
)
latin_exclude = re.compile(
    r"arabic|naskh|kufi|vazir|amiri|scheherazade|paktype|tifinagh|taitham|"
    r"armenian|cherokee|balinese|sinhala|carian|hebrew|siddham|bengali|"
    r"devanagari|ethiopic|gurmukhi|thai|khmer|lao|myanmar|cham|sogdian|"
    r"multani|linear|sundanese|meetei|emoji|symbol|cjk|japanese|chinese|"
    r"korean|tibetan|georgian|tamil|kannada|telugu|oriya|mongolian",
    re.IGNORECASE,
)
arabic_keep = re.compile(
    r"arabic|naskh|kufi|vazir|amiri|scheherazade|paktype", re.IGNORECASE
)
handwriting_keep = re.compile(
    r"dancing|kaushan|kristi|stevehand|sjfonts|comfortaa|handwriting",
    re.IGNORECASE,
)

catalogs = {
    "data/standard_clean_fonts.txt": [
        path for path in fonts
        if latin_keep.search(path) and not latin_exclude.search(path)
    ],
    "data/arabic_fonts.txt": [path for path in fonts if arabic_keep.search(path)],
    "data/handwriting_fonts.txt": [
        path for path in fonts if handwriting_keep.search(path)
    ],
}

for target, matches in catalogs.items():
    output = Path(target)
    output.write_text("\n".join(matches) + ("\n" if matches else ""))
    print(f"✓ {len(matches):3d} fonts -> {output}")
PYEOF
else
    echo "Skipped (--no-catalogs)."
fi

# ---------------------------------------------------------------------------
# 6. Install pre-commit hooks
# ---------------------------------------------------------------------------
run_step "Installing git hooks..."
if [[ $INSTALL_HOOKS -eq 1 ]]; then
    if [[ -d .git ]]; then
        uv run --no-sync pre-commit install
        echo "✓ pre-commit hooks installed."
    else
        echo "Skipped: not a git repository."
    fi
else
    echo "Skipped (--no-hooks)."
fi

# ---------------------------------------------------------------------------
# 7. Optional validation
# ---------------------------------------------------------------------------
run_step "Running validation..."
if [[ $RUN_CHECKS -eq 1 ]]; then
    uv run --no-sync ruff check src scripts tests
    uv run --no-sync ruff format --check src scripts tests
    uv run --no-sync mypy src
    uv run --no-sync pytest -q
    echo "✓ Validation passed."
else
    echo "Skipped; re-run with --check to validate."
fi

echo ""
echo "=========================================================="
echo " EmbeddingOCR environment successfully configured!"
echo ""
echo " Run unit tests:   uv run pytest"
echo " Run linter:       uv run ruff check src scripts tests"
echo " Run type checker: uv run mypy src"
echo " Run CLI:          uv run embedding-ocr --help"
echo " Run benchmark:    uv run python scripts/compare_new_ocr_models.py"
echo "=========================================================="
