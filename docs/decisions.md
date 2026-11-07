# Decisions Log: Phase 0 Spike

## 1. Embedder Selection & Integration
- Model: `google/embeddinggemma-2` (Hugging Face)
- Architecture: Gemma-based multimodal foundation model (~740M parameters: 270M text backbone + 170M vision + 300M audio).
- Embedding Dimension: Native 768-d, supports Matryoshka dimensions (768, 512, 256, 128).
- Loading library: `sentence-transformers>=3.0` / `transformers>=4.48`, using SentenceTransformer directly.
- Truncation rule: Truncate leading dimensions and apply L2 re-normalization (`normalize_embeddings=True`).

## 2. Font Selection
Six distinct open-licensed fonts partitioned into Set A (Index) and Set B (Held-out):
- **Set A (Index / Prototypes)**:
  1. LiberationSans-Regular (`/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf`)
  2. LiberationSerif-Regular (`/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf`)
  3. DejaVuSans (`/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf`)
- **Set B (Held-out Queries)**:
  1. Roboto-Regular (`/usr/share/fonts/truetype/roboto/unhinted/RobotoTTF/Roboto-Regular.ttf`)
  2. NotoSans-Regular (`/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf`)
  3. LiberationMono-Regular (`/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf`)

## 3. Glyph Rendering Parameters
- Canvas: White background (RGB 255, 255, 255), black text (RGB 0, 0, 0).
- Size: 64x64 pixels standard square canvas.
- Font size: 44 pt, centered horizontally and vertically using font bounding box offset.
- Target classes: 52 classes (Latin `a-z`, `A-Z`).
- Case-pair observation: Case distinction in isolated square-cropped glyphs (e.g. `c`/`C`, `o`/`O`, `s`/`S`, `v`/`V`, `w`/`W`, `x`/`X`, `z`/`Z`) will be measured specifically.

## 4. Vector Search
- Index: FAISS `IndexFlatIP` (Exact Inner Product on L2-normalized embeddings = Cosine similarity).
- Metric: Top-1 and Top-5 accuracy across the 52 classes.

## 5. Generative OCR VLM Comparison (November 2026)

Added generative OCR VLMs to the cross-engine comparison:

| Model | Loading path | Status |
| --- | --- | --- |
| `lightonai/LightOnOCR-3-0.8B` | `transformers>=5.5.4`, `Qwen3_5ForConditionalGeneration` | Verified from model card; transcription prompt is `""` |
| `datalab-to/surya-ocr-2.5` | `surya-ocr` package (`RecognitionPredictor`), optional extra | Verified from model card; requires a vLLM or llama.cpp backend |
| `SoMarkAI/OXR-1.0` | SoMark `POST /v1/parse/sync` (local port 8000 or free API), or OpenAI-compatible vLLM `/chat/completions` | Verified from model card; local `transformers` inference is not documented |
| `dots-studio/dots.mocr` | `AutoModelForCausalLM` + `AutoProcessor`, `trust_remote_code=True` | Opt-in; upstream code imports `flash_attn` even when SDPA is selected |

- Shared benchmark: `scripts/compare_new_ocr_models.py`.
- All engines use the same rendered glyph queries (default: 52 Latin classes,
  3 held-out Liberation fonts) and report exact, case-insensitive, and
  extracted-glyph accuracy plus per-item latency.
- Unavailable engines are recorded in the JSON/Markdown report instead of
  aborting the run, because the VLMs have different hardware and serving
  requirements.
- OXR defaults to its public free parsing API in the benchmark scripts; set
  `--oxr-base-url` to a local endpoint and `--oxr-backend parse|chat` to use a
  self-hosted deployment.
- `dots.mocr` is excluded from the default engine set because its remote
  modeling code requires a `flash_attn` shim; it remains available via
  `--engines all` or `--engines dots`.

## 6. Embedder Comparison Backends (November 2026)

The prototype-matching system is embedder-agnostic (`Embedder` protocol), so
additional multimodal models run through the identical rendering, FAISS and
matching pipeline:

| Key | Model | Native dim | Notes |
| --- | --- | --- | --- |
| `gemma` | `google/embeddinggemma-2` | 768 | Matryoshka 768/512/256/128 |
| `siglip` | `google/siglip-so400m-patch14-384` | 1152 | Separate image/text towers; no native Matryoshka (truncation is approximate) |
| `jina` | `jinaai/jina-embeddings-v4` | 2048 | Qwen2.5-VL-3B base; Matryoshka 2048/1024/512/256/128; remote code |

- Adapters live in `src/embedding_ocr/recognition/embedding/embedder.py` and
  implement `embed_images` / `embed_texts`, returning float32 `(n, dim)`
  L2-normalized vectors.
- All adapters slice to the active dimension and re-normalize, so a common
  `--*-dim` value can be used for a fair size-controlled sweep.
- Jina v4 requires `transformers>=4.52`, `peft>=0.15.2` and `torchvision`;
  `flash_attn` is optional and is never forced.
- SigLIP is trained for sigmoid image-text matching rather than Matryoshka
  retrieval, so low truncated dimensions are expected to degrade faster.

## 7. Multilingual Segmentation & RTL (November 2026)

**New scope:** Arabic/connected scripts and Latin cursive. Segmentation is
split into two stages: line→word and word→unit. Compute is inference-only; no
fine-tuning or custom detector training in this phase.

Decisions:
- Script is explicit configuration (`script="latin"|"arabic"`). Arabic and
  other RTL scripts use right-to-left reading order; words are joined in
  reading order to produce the logical string.
- Latin case-pair disambiguation is gated to Latin letters only. Arabic and
  other caseless scripts must never be remapped through `CASE_PAIRS`.
- `PrototypeDB` is script-aware. When `classes` is not given, classes come
  from the configured script (`ARABIC_CLASSES` for Arabic), and the script is
  persisted in prototype metadata.
- The Latin physical character-count estimate is not applied to Arabic; it is
  calibrated for horizontal Latin text and would reject valid candidates.
- Planned learned components (inference only): PaddleOCR PP-OCRv5 detection,
  Surya detection, CRAFT/EasyOCR word boxes, and optional pretrained CTC
  alignment. Classical components: improved projection/connected components,
  diacritic merging, watershed/seam carving.
- Embedding-guided Viterbi alignment (sliding-window embeddings + prototype
  similarity) is the planned segmentation-free fallback for cursive and
  connected scripts.

Unverified:
- EmbeddingGemma 2 Arabic text/image retrieval quality.
- PaddleOCR 3.x detection API and Arabic detection quality.
- Surya Arabic detection behavior.
- Availability of word/unit-level annotations in MADAR, IFN/ENIT, KHATT and
  AHCD.

Foundational implementation (this iteration):
- `src/embedding_ocr/data/scripts.py` (script detection, aliases, RTL).
- Arabic classes and `classes_for_script()` in `data/glyph_renderer.py`.
- Script-aware `PrototypeDB` and `PrototypeMatcher`.
- RTL ordering in `CharacterSegmenter`, `VisualWordSegmenter`,
  `CraftWordDetector`, `HierarchicalWordRecognizer`, and
  `EmbeddingRecognizer`.
- Script-aware `EmbeddingVocabulary` (Arabic default fonts, token filtering,
  cache tags, RTL rendering with libraqm/reshaper fallback).
- `EmbeddingAlignmentSegmenter` (sliding-window prototype scoring + Viterbi DP)
  and `ArabicUnitSegmenter` (diacritic merging, RTL unit order).
- `scripts/probe_segmentation.py` for optional detector/embedder comparison;
  missing dependencies are recorded rather than aborting the run.
- Tests: `tests/test_multilingual_segmentation.py`,
  `tests/test_arabic_alignment.py`.

## 8. Config, pipeline, and CLI (November 2026)

- Configuration is typed with Pydantic and loaded from YAML. Swapping the
  recognizer (`hierarchical`, `embedding`, `paddle`, `tesseract`, `easyocr`)
  changes only `recognition.backend`.
- Detection and recognition are independent protocols. `OCRPipeline` applies
  preprocessing, detects regions, crops them, calls the recognizer, and
  reconstructs line-major reading order.
- Detection boxes are cached by a hash of image bytes, image shape, detector
  class, and RTL flag. Baseline and embedding runs can therefore reuse exactly
  the same crops.
- Prototype loading validates the saved index dimension and embedder name
  against the active config. A mismatch raises instead of silently producing
  invalid cosine scores.
- The benchmark manifest format is `image_path<TAB>reference` per line;
  relative paths are resolved against the manifest directory. Reported metrics
  are normalized CER, WER, exact match, mean latency, and p95 latency.
- PaddleOCR 3.x detection schema and Arabic detection quality remain
  unverified; the adapter accepts both `predict` and legacy `ocr` shapes.
- `recognize --json` and `benchmark` records include the resolved config and
  git commit hash; benchmark runs also write a sibling `.config.yaml`.
