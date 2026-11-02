# EmbeddingOCR

Modular OCR system that replaces conventional character recognition with
**visual embedding matching**, while preserving the standard OCR detection
and processing pipeline.

> **Scope:** Latin script only (a-z / A-Z). Arabic and other connected scripts
> are out of scope for v1 and listed as future work.

---

## Overview

```
Input Image
  → 1. Preprocessing (resize, denoise, binarize, deskew)      KEEP
  → 2. Text detection (DBNet / CRAFT / PaddleOCR det)         KEEP
  → 3. Region cropping (word / line crops)                    KEEP
  → 4. Character recognition                                  REPLACED ← research contribution
        segmentation / alignment
        → embedding extraction (EmbeddingGemma 2)
        → FAISS prototype search (cosine similarity)
        → character selection
  → 5. Text reconstruction (ordering, spaces, lines)          KEEP
  → 6. Confidence scoring                                     MODIFIED
  → Recognized text
```

**Embedder:** [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2) —
multimodal model (text, image, audio) in a shared 768-d space (Matryoshka:
768/512/256/128), Apache 2.0 license.

---

## Installation

```bash
# Python >= 3.11 required; managed with uv only
uv sync --group dev
```

## Commands

```bash
uv run pytest                                             # run tests
uv run ruff check . && uv run ruff format .              # lint / format
uv run mypy src                                          # type-check
uv run python scripts/phase0_az_spike.py \
    --config configs/spike_az.yaml                       # Phase 0 spike
uv run python scripts/compare_new_ocr_models.py          # Gemma vs new OCR VLMs
```

### Generative OCR VLM comparison

`scripts/compare_new_ocr_models.py` runs one shared glyph benchmark against:

| Key | Model | Backend |
|-----|-------|---------|
| `gemma` | `google/embeddinggemma-2` | FAISS prototype matching |
| `lighton` | `lightonai/LightOnOCR-3-0.8B` | Transformers (`Qwen3_5ForConditionalGeneration`) |
| `dots` | `dots-studio/dots.mocr` | Transformers (`trust_remote_code=True`) |
| `oxr` | `SoMarkAI/OXR-1.0` | OpenAI-compatible local vLLM server |

```bash
# Run everything; unavailable engines are reported, not fatal
uv run python scripts/compare_new_ocr_models.py

# Quick smoke test / subset
uv run python scripts/compare_new_ocr_models.py --engines gemma lighton --limit 52

# OXR is served separately (model card ships an out-of-tree vLLM plugin)
uv run python scripts/compare_new_ocr_models.py \
    --engines oxr --oxr-base-url http://127.0.0.1:8080/v1
```

Results are written to `docs/new_ocr_models_comparison.{json,md}`. `dots`
optionally uses `qwen_vl_utils` when installed; otherwise it falls back to the
processor's chat template. The exact single-glyph prompt for `dots` and OXR is
**unverified** (both are document-parsing models), so override them with
`--dots-prompt` / `--oxr-prompt` when tuning.

---

## Project Status

| Phase | Status | Description |
|-------|--------|-------------|
| Phase 0 | ✅ In progress | A-Z embedding + FAISS search spike |
| Phase 1 | ⏳ Pending | Full project setup |
| Phase 2 | ⏳ Pending | Core abstractions |
| Phase 3 | ⏳ Pending | Baseline (PaddleOCR) |
| Phase 4 | ⏳ Pending | EmbeddingRecognizer MVP |
| Phase 5 | ⏳ Pending | Alignment (if needed) |
| Phase 6 | ⏳ Pending | Fair benchmark |
| Phase 7 | ⏳ Pending | Polish & release |

See [`AGENTS.md`](AGENTS.md) for the full plan and decision log.

---

## Phase 0 Results (Preliminary)

_Full results in [`docs/feasibility.md`](docs/feasibility.md) after Phase 0 completes._

| Dim | E1 Text Self Top-1 | E2 Image→Text Top-1 | E3 Image→Image Top-1 |
|-----|---------------------|----------------------|----------------------|
| 768 | 100% | 61.5% | 91.0% |
| 512 | 100% | 64.7% | 91.7% |
| 256 | 100% | 62.8% | 89.1% |
| 128 | 100% | TBD | TBD |

**Key finding:** E3 (image → image across fonts) tops out at ~91% due to
case-pair confusions (c/C, o/O, s/S, v/V, w/W, x/X, z/Z, u/U, p/P).
Top-5 is 100% at all tested dimensions.

---

## Fonts Used

**Set A (prototype index):** LiberationSans, LiberationSerif, DejaVuSans  
**Set B (held-out test):** Roboto, NotoSans, LiberationMono

All fonts are open-licensed (SIL OFL / Apache 2.0).

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
