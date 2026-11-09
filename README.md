# EmbeddingOCR

Modular OCR system that replaces conventional character recognition with
**visual embedding matching**, while preserving the standard OCR detection
and processing pipeline.

> **Scope:** Latin script first (a-z / A-Z). Multilingual segmentation work is
> now active for Arabic/connected scripts and Latin cursive; CJK, Indic, and
> Thai/Lao/Khmer remain future work.

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

Automatic Linux/Kaggle/Colab setup (system libraries, uv, Arabic shaping,
OpenCV repair, Gradio, and all benchmark extras):

```bash
bash scripts/setup_auto.sh                 # setup only
bash scripts/setup_auto.sh --launch        # setup + public Gradio link
bash scripts/setup_auto.sh --launch --no-share
bash scripts/setup_auto.sh --with-paddle   # include the PaddleOCR baseline
# or: make setup-auto
```

The full developer setup also installs font catalogs and pre-commit hooks:

```bash
bash scripts/setup_env.sh
bash scripts/setup_env.sh --all-extras   # also jina/craft/surya/train/ui
bash scripts/setup_env.sh --check        # also run lint/types/tests
# or: make setup
```

Or manage the Python environment manually (Python >= 3.11, uv only). Pass all
extras in **one** `uv sync`; each sync removes packages from extras you did not
select. The default `setup_env.sh` already selects this combination:

```bash
uv sync --group dev \
  --extra baseline --extra tesseract --extra easyocr \
  --extra experiments --extra arabic --extra ui
```

If a sync removed OpenCV (`ModuleNotFoundError: No module named 'cv2'`), rerun
the command above or install the core wheel directly:

```bash
uv pip install --python .venv/bin/python opencv-python-headless
```

## Web interface

```bash
.venv/bin/embedding-ocr ui                    # public share link by default
.venv/bin/embedding-ocr ui --no-share         # local-only
.venv/bin/embedding-ocr-ui --host 0.0.0.0 --port 7860
# or from a fresh checkout (syncs core + UI only)
uv run --extra ui embedding-ocr ui
```

`make ui` uses the already-synced environment directly, so it does not prune
baseline extras the way a bare `uv run` sync would.

The UI includes bundled Latin and Arabic example images that set the matching
profile and script. It also lets you upload or capture an image, choose a YAML
profile, override the recognizer/detector/script/device, and inspect recognized
text, confidence, region boxes, and JSON details. The first request builds prototypes/vocabulary
and may download `google/embeddinggemma-2`, so it can take several minutes.
Quick mode caps the hierarchical vocabulary at 4k words / 3k subwords.

## Commands

```bash
uv run pytest                                             # run tests
uv run ruff check . && uv run ruff format .              # lint / format
uv run mypy src                                          # type-check

uv run embedding-ocr version                             # CLI smoke test
uv run --extra ui embedding-ocr ui                       # Gradio web UI
uv run embedding-ocr show-config --config configs/default.yaml
uv run embedding-ocr recognize image.png --config configs/default.yaml --json
uv run embedding-ocr build-prototypes --config configs/embedding_seg.yaml \
    --output artifacts/prototypes/clean_alnum_768d
uv run embedding-ocr build-vocabulary --config configs/default.yaml
uv run embedding-ocr benchmark manifest.tsv \
    --config configs/embedding_seg.yaml --output docs/benchmark_report.json

uv run python scripts/phase0_az_spike.py \
    --config configs/spike_az.yaml                       # Phase 0 spike
uv run python scripts/probe_segmentation.py --mode all   # detector/embedder probe
uv run python scripts/compare_new_ocr_models.py          # Gemma vs new OCR VLMs
uv run python scripts/test_word_ocr_vs_glm.py \
    --gemma-dim 768 --augment                            # Benchmark vs GLM-OCR (synthetic)
uv run python scripts/test_iam_handwriting.py \
    --num-samples 50 --gemma-dim 768                     # Benchmark on IAM Handwriting
```

### Configuration

`configs/default.yaml` is the full reference. Key switches:

| Field | Values | Effect |
| --- | --- | --- |
| `recognition.backend` | `hierarchical`, `embedding`, `paddle`, `tesseract`, `easyocr` | Chooses the recognizer only |
| `detector.backend` | `full`, `visual`, `craft`, `paddle` | Chooses the text detector |
| `recognition.script` | `latin`, `arabic` | Sets classes, RTL order, and case rules |
| `embedder.backend` | `gemma`, `siglip`, `jina` | Swaps the visual embedder |
| `index.backend` | `faiss`, `numpy` | Exact FAISS or NumPy reference index |

Use `configs/baseline_paddle.yaml`, `configs/embedding_seg.yaml`, and
`configs/embedding_align.yaml` as starting points. Detection boxes are cached
under `pipeline.detection_cache` so baseline and embedding runs use identical
crops.

---

## Benchmark: Word & Phrase OCR vs GLM-OCR

We evaluated **EmbeddingOCR** (Hierarchical and Character-level) against **GLM-OCR** across printed synthetic document crops (clean and degraded) and real human handwriting crops from the IAM dataset.

### 1. Degraded Printed Crops (Noise, Blur, Skew)

| Recognizer | Primary Word Norm CER | Word Raw CER | Word Norm Acc | Phrase Norm CER | Latency / Word |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **GLM-OCR** (`zai-org/GLM-OCR`) | **0.00%** | **0.00%** | **100.0%** | **0.00%** | 618.6 ms |
| **Hierarchical EmbeddingOCR (768D)** | **4.12%** | **9.41%** | **80.0%** | **11.27%** | 793.4 ms |
| **Char-level EmbeddingOCR (768D)** | **17.25%** | **29.02%** | **30.7%** | **19.42%** | 5,157.5 ms |

_See [`docs/benchmark_word_ocr_vs_glm.md`](docs/benchmark_word_ocr_vs_glm.md) for full analysis._

### 2. IAM Handwriting Dataset (Real Human Cursive)

| Recognizer | Primary Word Norm CER | Word Raw CER | Word Norm Acc | Norm WER | Latency / Word |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **GLM-OCR** (`zai-org/GLM-OCR`) | **5.73%** | **6.17%** | **84.0%** | **18.00%** | 1,144.3 ms |
| **Hierarchical Multi-Font + Length-Filtered (768D)** | **55.51%** | **56.83%** | **44.0%** | **86.00%** | 1,829.8 ms |
| **Hierarchical Multi-Font + Deslanted (768D)** *(Sol 1 + 2)* | **64.32%** | **65.64%** | **48.0%** | **82.00%** | 1,815.8 ms |
| **Hierarchical Multi-Font (768D)** *(Solution 1)* | **62.11%** | **62.11%** | **40.0%** | **90.00%** | 1,760.6 ms |
| **Hierarchical Baseline (768D)** *(Single Font)* | **84.14%** | **91.19%** | **32.0%** | **104.00%** | 2,516.5 ms |

_See [`docs/iam_handwriting_benchmark_report.md`](docs/iam_handwriting_benchmark_report.md) for full analysis._

**Key Takeaways:**
1. **CER Drops Down to 55.51%:** Combining character capacity estimation with multi-font visual prototypes and deslanting dropped Normalized CER from **84.14% down to 55.51%** (-28.63 pp total drop).
2. **Subword Hallucinations Suppressed:** Prevents runaway subword strings on narrow crops.




### Generative OCR VLM comparison

`scripts/compare_new_ocr_models.py` runs one shared glyph benchmark against:

| Key | Model | Backend |
|-----|-------|---------|
| `gemma` | `google/embeddinggemma-2` | FAISS prototype matching |
| `siglip` | `google/siglip-so400m-patch14-384` | FAISS prototype matching (same pipeline) |
| `jina` | `jinaai/jina-embeddings-v4` | FAISS prototype matching (same pipeline) |
| `lighton` | `lightonai/LightOnOCR-3-0.8B` | Transformers (`Qwen3_5ForConditionalGeneration`) |
| `surya` | `datalab-to/surya-ocr-2.5` | `surya-ocr` package (vLLM / llama.cpp) |
| `oxr` | `SoMarkAI/OXR-1.0` | SoMark `/parse/sync` API or OpenAI-compatible vLLM |
| `dots` | `dots-studio/dots.mocr` | Transformers (`trust_remote_code=True`, opt-in) |

The embedding models (`gemma`, `siglip`, `jina`) all run through the **same**
prototype-rendering, FAISS `IndexFlatIP`, per-class max aggregation and
case-resolution matcher, so the comparison isolates the embedder.

```bash
# Run the default engines; unavailable engines are reported, not fatal
uv run python scripts/compare_new_ocr_models.py

# Compare embedders only at their native dimensions
uv run python scripts/compare_new_ocr_models.py --engines gemma siglip jina

# Matryoshka / truncation sweep (each adapter re-normalizes after slicing)
uv run python scripts/compare_new_ocr_models.py \
    --engines gemma siglip jina --gemma-dim 256 --siglip-dim 256 --jina-dim 256

# Quick smoke test / subset
uv run python scripts/compare_new_ocr_models.py --engines gemma siglip --limit 52

# Surya (optional dependency; auto-spawns a vLLM or llama.cpp backend)
uv sync --extra surya
uv run python scripts/compare_new_ocr_models.py --engines surya
# or reuse a running Surya server
uv run python scripts/compare_new_ocr_models.py --engines surya \
    --surya-inference-url http://127.0.0.1:8000/v1

# OXR: free API by default, or point at a local server
uv run python scripts/compare_new_ocr_models.py \
    --engines oxr --oxr-base-url https://free-oxr.somark.ai/api/v1
uv run python scripts/compare_new_ocr_models.py \
    --engines oxr --oxr-base-url http://127.0.0.1:8000/v1 --oxr-backend parse
uv run python scripts/compare_new_ocr_models.py \
    --engines oxr --oxr-base-url http://127.0.0.1:8080/v1 --oxr-backend chat

# Public API returned Cloudflare 524: retry with backoff and pacing
uv run python scripts/compare_new_ocr_models.py --engines oxr \
    --oxr-retries 4 --oxr-retry-backoff 3 --oxr-request-delay 0.3
```

Results are written to `docs/new_ocr_models_comparison.{json,md}`. `dots` is
opt-in (`--engines all` includes it) because its remote modeling code needs a
flash_attn compatibility shim; select it explicitly only when needed. The
exact single-glyph prompt for the document-parsing models is **unverified**, so
override it with `--dots-prompt` / `--oxr-prompt` when tuning.

---

## Project Status

| Phase | Status | Description |
|-------|--------|-------------|
| Phase 0 | ✅ Complete | A-Z embedding + FAISS search spike |
| Phase 1 | ✅ Complete | Project setup, config, CLI, CI |
| Phase 2 | ✅ Complete | Core abstractions, pipeline, detection cache |
| Phase 3 | 🟡 Adapter ready | PaddleOCR adapter + config; dataset/deps needed for reference numbers |
| Phase 4 | ✅ MVP implemented | Character segmentation, prototypes, hierarchical recognition |
| Phase 5 | ✅ Implemented | Arabic units + embedding-guided Viterbi alignment |
| Phase 6 | 🟡 Runner ready | Manifest CER/WER runner; full held-out comparison pending data |
| Phase 7 | 🟡 In progress | CI, docs, CLI; release tag pending |

See [`AGENTS.md`](AGENTS.md) for the full plan and decision log.

---

## Phase 0 Results

_See [`docs/feasibility.md`](docs/feasibility.md) for the full analysis._

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
