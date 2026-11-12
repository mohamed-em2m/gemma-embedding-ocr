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
- Index: FAISS inner-product search on L2-normalized embeddings (cosine similarity).
- Default: `faiss_ivf_sq8` (`IndexIVFScalarQuantizer`, `QT_8bit`, `METRIC_INNER_PRODUCT`).
  This stores 768-d vectors in 768 bytes instead of 3072 bytes (4x smaller) and
  uses `nlist`/`nprobe` for sub-linear search.
- Exact baseline: `faiss_flat_ip` (`IndexFlatIP`, float32). `numpy` is the
  dependency-free exact reference used in tests.
- Metadata records `backend`, `index_type`, `nlist`, `nprobe`, `quantizer_bits`,
  and `train_size`, so `FaissIndex.load()` restores the correct FAISS type.
  Older metadata without `backend` is inferred as `flat_ip`.
- `nlist` is clamped to roughly one centroid per 39 training vectors so small
  character vocabularies can still train; `nprobe` is clamped to `nlist`.
- Synthetic smoke benchmark (20k x 768d clustered vectors, 200 queries, k=10):
  `flat_ip` 58.6 MB / 0.52 ms per query; `ivf_sq8` 16.3 MB / 0.16 ms per query
  with exact-prototype recall@10 = 0.904 at `nprobe=16`. This measures exact
  prototype-ID recall, not per-class max aggregation. Use `flat_ip` for
  accuracy-critical or exact-recall runs; re-run `scripts/benchmark_index.py`
  on the real prototype cache before relying on these numbers.
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

## 9. Pruned Alignment Candidate Graph (November 2026)

The exhaustive stride × width graph was the dominant cost of
`EmbeddingAlignmentSegmenter`: every start and width combination was embedded,
and overlapping windows were re-embedded. A long word could require hundreds or
thousands of EmbeddingGemma calls even though the DP itself is cheap.

Decision:
- The default strategy is now `pruned`. Cheap visual geometry proposes a small
  set of cut positions; only edges between those positions are embedded.
- `VisualWordSegmenter.propose_cut_candidates()` combines vertical-projection
  valleys, zero-ink gaps, connected-component edges, horizontal gradients, and
  local projection minima. Non-maximum suppression enforces a minimum spacing.
- The candidate budget scales with the expected unit count, not the pixel width:
  `max(20, expected_units + 4)`, capped at 128. The default is two outgoing
  edges per start (`max_edges_per_start=2`), yielding roughly 40–50 windows for
  a 20-unit word instead of ~1,300.
- Width priors use `max(min_unit_width, 0.5 * crop_height)` so a badly spaced
  cut candidate cannot collapse the target unit width.
- If the pruned graph cannot cover the word (e.g. a missed boundary), the
  aligner retries once with the exhaustive graph when
  `fallback_exhaustive=true`. Fallbacks are visible in `AlignmentStats`.
- `AlignmentStats` reports boundary candidates, selected starts, candidate
  windows, embedded windows, strategy, and whether a fallback was used.
- `exhaustive` remains available via
  `recognition.alignment.strategy` for accuracy experiments and debugging.

Beam search is intentionally not used. The current DP is an exact
shortest-path/Viterbi pass over a small DAG; beam pruning only becomes useful
when adding label-transition or language-model scores, or when the graph is
large enough that exact decoding is the bottleneck. Reducing candidate
generation is the correct first fix.

## 10. Optional DTLR Line Baseline (November 2026)

DTLR (Detection-based Text Line Recognition) is a DINO-DETR-based line
recognizer. It is kept as an **optional external baseline**, not a default
backend, because it is not distributed as a pip-installable package.

Decision:
- `recognition.backend="dtlr"` loads a cloned official checkout through
  `recognition.dtlr_root` (or `DTLR_ROOT`) and an inference checkpoint through
  `recognition.dtlr_weights` (or `DTLR_WEIGHTS`).
- The adapter is line-level. It accepts a line crop, applies DTLR's inference
  resize (short side 800, longest side 1333), ImageNet normalization, and the
  repository's non-NMS decoding convention. It returns one `RecognitionResult`
  per line crop.
- The default config and charset are `config/Latin_CTC.py` and
  `datasets/default_charset.json` inside the checkout. Override them with
  `dtlr_config` / `dtlr_charset` (or `DTLR_CONFIG` / `DTLR_CHARSET`).
- The official Python files call `torch.cuda.set_device(0)` at import and
  require compiled multi-scale deformable-attention ops. The adapter therefore
  requires a CUDA torch build and a compatible GPU environment. In addition to
  the DTLR checkout, install its inference imports (`addict`, `yapf`, `timm`,
  `scipy`, `torchvision`) and compile the deformable-attention extension as
  described by the upstream repository.
- `bash scripts/setup_env.sh --dtlr` installs the `dtlr` extra, clones the
  official checkout into `third_party/DTLR` (or `$DTLR_ROOT`), and prints the
  checkpoint and extension-compilation commands. `--all-extras` includes the
  Python dependencies but does not clone the external repository.
- Checkpoint weights can be local files. Unofficial community conversions
  exist on the Hugging Face Hub, but they remain unofficial; pin a revision and
  validate the charset/output convention before trusting benchmark numbers.
- DTLR is not added to the glyph-level unified comparison script. Use the
  line-manifest benchmark (`embedding-ocr benchmark`) for a like-for-like
  comparison against the embedding pipeline.

Example:
```yaml
recognition:
  backend: dtlr
  dtlr_root: /opt/DTLR
  dtlr_weights: /opt/DTLR/checkpoints/line/checkpoint.pth
  dtlr_device: cuda
```

Limitations:
- DTLR has no verified Arabic or connected-script training. Arabic comparison
  requires line-level Arabic fine-tuning data that this project does not have.
- The adapter assumes the Latin charset by default; a checkpoint trained with a
  different charset must ship a matching JSON charset.

## 11. Held-out Latin/Arabic OOD Benchmark (November 2026)

The previous word benchmark added `EVAL_WORDS` to the EmbeddingOCR
hierarchical vocabulary, and its prototype fonts were not guaranteed to be
disjoint from the evaluation fonts. Those numbers are therefore not a clean
held-out test.

Decision:
- `scripts/build_ood_dataset.py` renders a deterministic held-out dataset with
  Latin and Arabic glyphs, words, phrases, Arabic diacritics and ligatures.
- Prototype fonts and evaluation fonts are disjoint for each script; the split
  is recorded in `dataset.json`.
- `scripts/benchmark_ood_ocr.py` reads the manifest, feeds identical crops to
  GLM-OCR and EmbeddingOCR, and never passes evaluation words as `extra_lexicon`.
- Conditions are `clean`, `mild` and `hard`. Hard combines downscaling, JPEG
  artifacts, mild perspective, blur, noise and contrast changes while keeping
  text human-readable.
- Generated images live under `data/` (gitignored); only the generator,
  manifest schema and tests are committed.
- The real `google/embeddinggemma-2` and `zai-org/GLM-OCR` checkpoints are not
  cached in this environment, so a full local benchmark run still requires
  downloading them and enough RAM/GPU memory.

Follow-up after the first OOD run (2026-11-09):
- Root cause of the weak Latin result was candidate starvation, not the
  embedder: the benchmark capped the hierarchical vocabulary at 4k words / 3k
  subwords, sorted by token id. Gemma token ids are not frequency ordered, so
  tokens such as `▁fjord`, `▁sphinx` and `▁labyrinth` were silently removed.
- The benchmark now uses the full tokenizer vocabulary by default, increases
  raw FAISS depth for text prototypes, and keeps OOD words out of
  `extra_lexicon`.
- Already-cropped word/diacritic/ligature samples now call
  `recognize_word_crop()` directly, so a single unit is not split into
  pseudo-words by the line word-grouper.
- `VisualWordSegmenter` now uses a higher, conservative gap threshold
  (`max(8, 0.40 * typical_height)`) and never lowers it via the dynamic
  candidate; this avoids splitting words such as `fjord` and `sphinx` on wide
  or tracked fonts.
- `configs/default.yaml` now uses the full tokenizer vocabulary
  (`max_words: null`, `max_subwords: null`). The UI quick mode remains an
  explicit 4k/3k speed cap.

Detection-first fix (2026-11-10):
- Root cause of misplaced/missed words on full images: `detector.backend: full`
  plus a word segmenter that sorted connected components by x only, merging
  multiple text lines into one crop.
- `detector.backend` now defaults to `auto`: PaddleOCR `TextDetection`
  (`PP-OCRv5_server_det` by default, `PP-OCRv5_mobile_det` available) with
  CRAFT and a dependency-free line detector as fallbacks. `full` remains only
  for images that are already cropped.
- PaddleOCR is loaded through the standalone `TextDetection` module so the
  recognition model is not loaded during detection.
- PaddlePaddle is pinned to `>=3.2.2,<3.3` for CPU because 3.3.x crashes in its
  oneDNN/PIR executor (`ConvertPirAttribute2RuntimeAttribute`).
- `VisualWordSegmenter` groups components into line bands before splitting
  words, chooses foreground polarity for light-on-dark images, and retries with
  Otsu for low-contrast inputs.
- Added `scripts/benchmark_detection.py` with per-image precision/recall/F1,
  synthetic multi-line samples, JSON reports and red/green overlays. Real
  images need JSON annotation; unannotated images still produce overlays.

YOLO word detector (2026-11-10):
- Added a one-class word-localization path alongside the existing
  character/unit YOLO path. `label_level="word"` in
  `training/letter_dataset.py` writes one YOLO label per rendered word; the
  dataset generator remains font-disjoint across train/val/test.
- `scripts/build_word_dataset.py` defaults to mixed line/collage layouts,
  30k images, 1280 px, and class `word`.
- `scripts/train_yolo_words.py` trains Ultralytics YOLO26 with
  `single_cls=True`, no horizontal/vertical flips, bounded rotation/scale, and
  `max_det=300`.
- Runtime: `detector.backend: yolo_words` is implemented by
  `YoloWordDetector`; boxes are padded, deduplicated, and ordered line-major
  with RTL support. `configs/yolo_words.yaml` combines the detector with the
  hierarchical recognizer.
- Ultralytics remains an optional `train` extra and is AGPL-3.0 or
  commercially licensed. Full training requires a GPU; CPU is supported for
  smoke runs.

Kaggle word-detector training (2026-11-10):
- Target runtime is Kaggle 2x T4 (16 GB each). Default training is YOLO26s at
  1280 px, 30k synthetic word images, total batch 32 (16 per GPU), disk cache,
  8 workers.
- `scripts/kaggle_train_yolo_words.sh` installs the `train` extra directly
  into `.venv` with `uv pip install -e ".[train]"` (after creating the venv),
  then verifies `ultralytics` and `uharfbuzz` import. This avoids a stale
  `.venv` silently skipping the training extra. It generates the dataset under
  `/kaggle/working/artifacts` and launches DDP on devices `0,1`. It uses
  `uv run --no-sync`, never `uv sync`.
- Checkpoints go to `/kaggle/working/artifacts/yolo_runs/yolo_words/weights/`;
  copy `best.pt` to a Kaggle dataset before the 12-hour session ends.
- If T4 memory is tight, lower `BATCH` to 24 or `IMGSZ` to 1024; keep the same
  `IMGSZ` at inference in `configs/yolo_words.yaml`.

## CTC line recognizer (branch `feat/ctc-ocr`)

Data sources (checked 2026-11-12):
- `Teklia/IAM-line` (English handwriting, line image + text): 6482/976/2915.
  The split is named `validation`, not `val`; the loader maps split aliases.
- `Teklia/KHATT_v1.0` (Arabic handwriting, same schema): 4672/963/1038.
- `staghado/Bentham` was proposed but is **not** the Bentham corpus: it is a
  text-only config with 24 rows and no images. Bentham lines require the
  registered official distribution, so Bentham is deferred.
- `nizarcharrada/khattarabic` (Kaggle) needs credentials and its schema is
  unverified. `scripts/prepare_line_datasets.py --inspect-local` prints the
  tree and label-file samples before any import; no schema is guessed.

Text normalization (recorded because it sets the CER ceiling):
- NFKC, which folds Arabic presentation forms to base letters.
- Zero-width/bidi controls removed; whitespace collapsed.
- Arabic harakat/Quranic marks and tatweel stripped.
- Alef variants `أ إ آ ٱ ٲ ٳ ٵ` folded to `ا`.
- `ة/ه`, `ى/ي`, `ء/ؤ/ئ` stay distinct.
- Latin case is preserved; CER is reported case-sensitive with a
  case-insensitive secondary number.

Charset:
- Built from the train split only, sorted by code point, blank at index 0.
- Val/test rows containing out-of-vocabulary characters are dropped and
  counted in `totals.coverage_dropped`, never silently mapped.

Encoder time axis:
- The CTC constraint `T >= 2L - 1` is enforced at manifest time via
  `FilterConfig.image_height` and `FilterConfig.width_stride`.
- Measured on 400 KHATT train lines: at height 32, width stride 4 (classic
  CRNN) is feasible for only 37.5% of lines, stride 2 for 84.8%. IAM is
  89.5% / 100.0% respectively. The encoder must therefore downsample width
  by 2, not 4.
- KHATT labels are long (median 62 chars, p90 78, max 94); IAM median is 42.
  Long KHATT lines are dropped as `ctc_infeasible` rather than trained with a
  mis-aligned label.

Prepared line corpus (2026-11-12, `artifacts/lines`, gitignored):
- 16,031 usable lines: 10,428 train / 1,786 val / 3,817 test.
- 10,369 Latin (IAM-line) and 5,662 Arabic (KHATT); 115 characters.
- Raw rows 17,046 -> collected 16,033 -> kept 16,031. Drops: 1 out-of-
  vocabulary val row, 1 duplicate image, plus the per-split `too_long` and
  `ctc_infeasible` rejections counted in `stats.json`.
- 346 distinct texts appear in more than one split. This is text overlap, not
  image overlap (images are hash-deduplicated); handwriting corpora reuse
  common phrases. It is reported rather than removed so the test score stays
  honest.

Encoder normalization: GroupNorm, not BatchNorm.
- The encoder collapses height to a single row, so BatchNorm computes its
  statistics over `B * T` activations and mixes information across the batch.
- On an 8-line memorization fixture the BatchNorm model exploited this: it
  emitted confident characters (`z`, `y`) over the blank right edge of the
  crop, and its CTC loss moved from 0.09 (batch of 4) to 0.75 (batch of 8) on
  the same samples. Training loss reached 0.06 while the saved checkpoint
  scored 0.75 in eval mode, so the reported progress was not real.
- With GroupNorm the same fixture learns real words, train and eval agree,
  and `tests/test_ctc_model.py` pins the property that a sample's encoder
  output does not depend on its batch companions.
- Consequence: the model no longer depends on batch size, so `eval_batch_size`
  can differ from `batch_size` without changing results.

LR schedule:
- `train.scheduler: cosine` with `min_lr_ratio: 0.05` is the default for real
  runs. The cosine decay is computed from `epochs * steps_per_epoch`, so a
  short run anneals almost immediately.
- A cosine schedule that reaches zero starves CTC: on the fixture, 400 steps
  at constant 3e-3 reached CER 0.35, the same 400 steps on a cosine decay
  reached 0.80. `train.scheduler: constant` (CLI `--scheduler constant`) is
  therefore required for memorization checks and short smoke runs.
- Memorization check result (`--overfit 8 --scheduler constant`): CTC CER
  reaches 0.0000 with `ctc_weight: 1.0` after ~500 steps, and the transformer
  decoder reaches CER 0.0000 with the default `ctc_weight: 0.3`. The two heads
  trade off as expected: at 70% decoder weight the CTC head converges slower
  because the shared encoder is mostly optimized for the decoder.
