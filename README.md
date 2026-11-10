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

### Kaggle / Jupyter

Run the one-shot setup from a notebook cell before launching:

```python
%cd /kaggle/working/gemma-embedding-ocr
!./scripts/setup_env.sh
!make ui UI_ARGS="--no-share"
```

`setup_env.sh` installs the UI extra by default and repairs missing `cv2` and
Gradio. Use `--all-extras` to also install the optional jina/craft/surya/dtlr/
train backends.

`bash scripts/setup_auto.sh --launch --no-share` does not preload model weights.
The embedder uses the tuned cache when present, otherwise a free-memory
heuristic, and halves the batch on CUDA OOM. Run `make autotune` once for an
exact measurement; use `setup_auto.sh --autotune` only if you accept the extra
model load before launch.

Or manage the Python environment manually (Python >= 3.11, uv only). Do **not**
use `uv sync` on an existing/Kaggle environment: it removes packages from extras
you did not select. Install the project and selected extras without pruning:

```bash
uv pip install --python .venv/bin/python \
  -e ".[baseline,tesseract,easyocr,experiments,arabic,ui]" \
  pytest pytest-cov ruff mypy pre-commit
```

If OpenCV is still missing (`ModuleNotFoundError: No module named 'cv2'`),
force a clean headless reinstall:

```bash
uv pip uninstall --python .venv/bin/python opencv-python opencv-contrib-python
uv pip install --python .venv/bin/python --reinstall opencv-python-headless
```

## Web interface

```bash
make ui                                      # public share link by default
make ui UI_ARGS="--no-share"                 # local-only
make ui VENV=venv                            # use ./venv instead of ./.venv
bash scripts/run_ui.sh --no-share            # direct launcher
```

`make ui` runs `scripts/run_ui.sh`, which auto-detects `.venv` (preferred) or
`venv`, verifies `cv2` and Gradio, and repairs missing imports with `uv pip`
without running `uv sync` or pruning existing extras. From a fresh checkout,
install once with `bash scripts/setup_auto.sh` or
`uv pip install --python .venv/bin/python -e ".[ui]"` before launching.

The UI includes bundled Latin and Arabic example images that set the matching
profile and script. It also lets you upload or capture an image, choose a YAML
profile, override the recognizer/detector/script/device, and inspect recognized
text, confidence, detector region boxes, per-word segmentation boxes, and JSON
details. The first request builds
prototypes/vocabulary and may download `google/embeddinggemma-2`, so it can take
several minutes. Quick mode caps the hierarchical vocabulary at 4k words / 3k
subwords; uncheck it to use the full tokenizer vocabulary. Small caps can omit
rare words because tokenizer ids are not frequency-ordered.

### GPU selection and batch auto-tuning

`device: auto` checks every visible CUDA device and picks the one with the
most free memory, then the lowest utilisation. Explicit values such as
`cuda:1` are respected.

`batch_size: 0` means auto. The embedder reads `artifacts/gpu_tuning.json` if it
exists, otherwise uses a conservative free-memory estimate. If a batch still
runs out of memory, the embedder halves it and retries automatically.

Benchmark and cache the best batch for the current machine:

```bash
uv run --no-sync python scripts/autotune_gpu.py --device auto
```

The script reports every GPU, the selected device, the image/text batch sizes,
peak memory, and writes the recommendation to `artifacts/gpu_tuning.json`.
EmbeddingOCR uses one GPU per process; run separate processes with different
`CUDA_VISIBLE_DEVICES` values for parallel benchmarks.

### CUDA out of memory

The UI keeps only one pipeline in memory and releases the old one when the
profile/overrides change. `embedding_ocr` sets
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` by default. If a notebook
already loaded GLM-OCR or another model, restart the kernel before launching the
UI. The UI clears its pipeline cache automatically after a CUDA OOM so the next
request can retry.

## Commands

```bash
uv run --no-sync pytest                                             # run tests
uv run --no-sync ruff check . && uv run --no-sync ruff format .              # lint / format
uv run --no-sync mypy src                                          # type-check

uv run --no-sync embedding-ocr version                             # CLI smoke test
make ui                                                  # Gradio web UI
uv run --no-sync embedding-ocr show-config --config configs/default.yaml
uv run --no-sync embedding-ocr recognize image.png --config configs/default.yaml --json
uv run --no-sync embedding-ocr build-prototypes --config configs/embedding_seg.yaml \
    --output artifacts/prototypes/clean_alnum_768d
uv run --no-sync embedding-ocr build-vocabulary --config configs/default.yaml
uv run --no-sync embedding-ocr benchmark manifest.tsv \
    --config configs/embedding_seg.yaml --output docs/benchmark_report.json

uv run --no-sync python scripts/autotune_gpu.py --device auto  # pick GPU + batch
uv run --no-sync python scripts/phase0_az_spike.py \
    --config configs/spike_az.yaml                       # Phase 0 spike
uv run --no-sync python scripts/probe_segmentation.py --mode all   # detector/embedder probe
uv run --no-sync python scripts/compare_new_ocr_models.py          # Gemma vs new OCR VLMs
uv run --no-sync python scripts/test_word_ocr_vs_glm.py \
    --gemma-dim 768 --augment                            # Benchmark vs GLM-OCR (synthetic)
uv run --no-sync python scripts/test_iam_handwriting.py \
    --num-samples 50 --gemma-dim 768                     # Benchmark on IAM Handwriting
```

### Configuration

`configs/default.yaml` is the full reference. Key switches:

| Field | Values | Effect |
| --- | --- | --- |
| `recognition.backend` | `hierarchical`, `embedding`, `paddle`, `tesseract`, `easyocr` | Chooses the recognizer only |
| `detector.backend` | `auto`, `paddle`, `paddle_mobile`, `craft`, `yolo_words`, `lines`, `visual`, `full` | `auto` prefers PaddleOCR `PP-OCRv5_server_det`, then CRAFT, then line segmentation. `yolo_words` loads a word-box checkpoint. Use `full` only for already-cropped images. |
| `recognition.script` | `latin`, `arabic` | Sets classes, RTL order, and case rules |
| `embedder.backend` | `gemma`, `siglip`, `jina` | Swaps the visual embedder |
| `embedder.device` | `auto`, `cpu`, `cuda`, `cuda:N` | `auto` picks the least-used GPU |
| `embedder.batch_size` | `0` or positive int | `0` uses tuned cache/heuristic and OOM fallback |
| `index.backend` | `faiss_ivf_sq8`, `faiss_sq8`, `faiss_flat_ip`, `faiss`, `numpy` | Default is IVF + 8-bit scalar quantization (4x smaller vectors); `faiss_flat_ip` is exact float32. `numpy` is the exact reference. |

FAISS quantization is configured under `index`:

```yaml
index:
  backend: faiss_ivf_sq8   # or faiss_sq8, faiss_flat_ip, faiss, numpy
  nlist: 1024              # requested IVF centroids (clamped for small sets)
  nprobe: 16               # IVF lists probed per query
  quantizer_bits: 8        # only 8 is supported
  train_size: null         # null = train on all vectors
```

`faiss_ivf_sq8` stores each 768-d vector in 768 bytes instead of 3072 and uses
an IVF coarse quantizer for sub-linear search. Use `faiss_flat_ip` when exact
top-k is required, and benchmark the trade-off with:

```bash
uv run --no-sync python scripts/benchmark_index.py --n 100000 --dim 768
uv run --no-sync python scripts/benchmark_index.py \
    --vectors artifacts/vocab_cache/words_768d.npy --nprobe 32
```

Use `configs/baseline_paddle.yaml`, `configs/embedding_seg.yaml`, and
`configs/embedding_align.yaml` as starting points. Detection boxes are cached
under `pipeline.detection_cache` so baseline and embedding runs use identical
crops.

### YOLO word detector

Train a one-class YOLO detector that predicts word boxes, then feed those
boxes into any recognizer:

```bash
bash scripts/setup_env.sh --train
uv run --no-sync python scripts/build_word_dataset.py \
    --fonts /usr/share/fonts --out artifacts/yolo_words \
    --n 30000 --size 1280 --layout mixed
uv run --no-sync python scripts/train_yolo_words.py \
    --data artifacts/yolo_words/data.yaml --model yolo26s.pt --epochs 100
uv run --no-sync embedding-ocr recognize image.png --config configs/yolo_words.yaml
```

The generator renders Latin and Arabic words with real font shaping, writes one
YOLO class (`word`) per rendered word, and keeps validation/test fonts disjoint
from training fonts. `configs/yolo_words.yaml` uses `detector.backend:
yolo_words` and the hierarchical recognizer; `detector.yolo_model`,
`yolo_conf`, `yolo_iou`, `yolo_imgsz`, and `yolo_max_det` are configurable.
Training on CPU is possible for smoke tests but a CUDA GPU is strongly
recommended for full runs.

#### Real datasets (Roboflow Universe)

Real exports can be merged into the same one-class dataset. Every source class
(`word`, `texto`, ...) is rewritten to class `0`, and classification exports
are converted to one full-image `word` box per image.

```bash
# 1. Download public Roboflow Universe datasets (needs a free API key).
export ROBOFLOW_API_KEY=...
uv run --no-sync python scripts/download_roboflow_sources.py \
    --config configs/real_word_sources.yaml

# 2. Merge all sources listed in the config.
uv run --no-sync python scripts/build_real_word_dataset.py \
    --config configs/real_word_sources.yaml \
    --out artifacts/yolo_words_real

# 3. Train on the merged dataset.
uv run --no-sync python scripts/train_yolo_words.py \
    --data artifacts/yolo_words_real/data.yaml --model yolo26s.pt
```

Already-extracted directories and `.zip` exports can be passed directly with
`--source` (repeatable). Add a synthetic dataset as another `--source` to mix
real and synthetic words. `configs/real_word_sources.yaml` contains the
Devanagari, word-detection, `words-5tolc` (object detection with class
`Texto`) and math-words sources. The original math-words project has no
generated version, so the config uses the downloadable
`el-emam/math-words-oahhj` copy (same 5,297 images). Verify each `version` on
its Universe page before re-downloading.

The merger keeps each source's train/valid/test split when present, carves a
validation split from train when missing, prefixes filenames per source to
avoid collisions, and removes duplicate images across sources. Use
`--no-dedupe`, `--no-preserve-splits`, `--drop-empty`, `--copy-mode`, and
`--max-per-source` to change that behavior. The output `manifest.json` records
per-source counts and class mappings.

#### Publishing the merged dataset

The merged folder is Hugging Face ready: it carries a dataset card
(`README.md`), `sources.json` for attribution, `stats.json` with reproducible
statistics, `.gitattributes` with LFS rules, and `data.yaml`.

```bash
# Recompute statistics quoted in the dataset card.
make dataset-stats

# Validate the layout and preview what would be uploaded.
make hf-dry-run

# Upload (needs HF_TOKEN with write access).
make hf-upload
```

The canonical card is `docs/datasets/yolo_words_real.md`; the upload script
copies it into the dataset folder so the two cannot drift. All four upstream
projects are CC BY 4.0, so the merged release is CC BY 4.0 and requires
attribution to the four Roboflow authors.


#### Kaggle 2x T4

For the Kaggle GPU runtime with two T4s, use the helper script. It prepares the
training environment, builds the dataset, and launches YOLO26s on both GPUs
with a total batch size of 32:

```bash
# In a Kaggle notebook cell, from the repository root:
!bash scripts/kaggle_train_yolo_words.sh

# Shorter run, or memory-adjusted batch:
!EPOCHS=60 BATCH=24 bash scripts/kaggle_train_yolo_words.sh

# Merge real Roboflow sources (download first with DOWNLOAD=1).
!DATASET_MODE=mixed DOWNLOAD=1 ROBOFLOW_API_KEY=xxxx \
    bash scripts/kaggle_train_yolo_words.sh

# Or point at datasets already attached to the notebook.
!DATASET_MODE=mixed \
    REAL_SOURCES=/kaggle/input/words-a:/kaggle/input/words-b \
    bash scripts/kaggle_train_yolo_words.sh
```

Defaults are `DATASET_MODE=synthetic`, `DATASET=/kaggle/working/artifacts/yolo_words`,
`RUNS=/kaggle/working/artifacts/yolo_runs`, `IMGSZ=1280`, `EPOCHS=100`,
`DEVICES=0,1`, `BATCH=32`, `WORKERS=8`, and `CACHE=disk`. `DATASET_MODE` can
be `synthetic`, `real`, or `mixed`. Keep the resulting `best.pt` in a Kaggle
dataset or download it before the 12-hour session ends. The script uses `.venv`
and `uv run --no-sync`; it never runs `uv sync`.

The PaddleOCR detector requires `paddlepaddle>=3.2.2,<3.3` on CPU because
PaddlePaddle 3.3.x crashes in its oneDNN/PIR executor. Benchmark detectors with:

```bash
uv run --no-sync python scripts/benchmark_detection.py --backend auto --synthetic
uv run --no-sync python scripts/benchmark_detection.py --all --images tests/images
```

Annotations use `{"image.jpg": {"boxes": [[x, y, w, h], ...]}}`; images without
annotations still produce red/green overlay images.

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




### Held-out Latin/Arabic OOD comparison

```bash
# Full tokenizer vocabulary (default); no OOD evaluation words are injected.
uv run --no-sync python scripts/benchmark_ood_ocr.py \
    --engines glm gemma-hybrid \
    --output docs/ood_glm_vs_gemma

# Fast smoke test only. Small candidate caps exclude rare words, so use the
# full-vocabulary run for reported numbers.
uv run --no-sync python scripts/benchmark_ood_ocr.py \
    --engines gemma-hybrid --limit-per-script 20 \
    --hier-max-words 4000 --hier-max-subwords 3000
```

The hierarchical recognizer uses the full tokenizer vocabulary by default and
recognizes already-cropped word samples without re-segmenting them. Pass
`--hier-visual-prototypes` to render word/subword prototypes from the held-out
prototype fonts (visual prototypes then default to a 4k/3k cap).

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
uv run --no-sync python scripts/compare_new_ocr_models.py

# Compare embedders only at their native dimensions
uv run --no-sync python scripts/compare_new_ocr_models.py --engines gemma siglip jina

# Matryoshka / truncation sweep (each adapter re-normalizes after slicing)
uv run --no-sync python scripts/compare_new_ocr_models.py \
    --engines gemma siglip jina --gemma-dim 256 --siglip-dim 256 --jina-dim 256

# Quick smoke test / subset
uv run --no-sync python scripts/compare_new_ocr_models.py --engines gemma siglip --limit 52

# Surya (optional dependency; auto-spawns a vLLM or llama.cpp backend)
uv pip install --python .venv/bin/python -e ".[surya]"
uv run --no-sync python scripts/compare_new_ocr_models.py --engines surya
# or reuse a running Surya server
uv run --no-sync python scripts/compare_new_ocr_models.py --engines surya \
    --surya-inference-url http://127.0.0.1:8000/v1

# OXR: free API by default, or point at a local server
uv run --no-sync python scripts/compare_new_ocr_models.py \
    --engines oxr --oxr-base-url https://free-oxr.somark.ai/api/v1
uv run --no-sync python scripts/compare_new_ocr_models.py \
    --engines oxr --oxr-base-url http://127.0.0.1:8000/v1 --oxr-backend parse
uv run --no-sync python scripts/compare_new_ocr_models.py \
    --engines oxr --oxr-base-url http://127.0.0.1:8080/v1 --oxr-backend chat

# Public API returned Cloudflare 524: retry with backoff and pacing
uv run --no-sync python scripts/compare_new_ocr_models.py --engines oxr \
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
