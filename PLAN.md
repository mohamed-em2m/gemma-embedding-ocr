# EmbeddingOCR: Project Plan & Current Roadmap

> **Single Source of Truth** for current project state, active experiments, and next steps to resume execution seamlessly across sessions.

---

## 1. Project Overview & Current Architecture

**EmbeddingOCR** is a modular OCR architecture replacing traditional CTC / attention-based character recognition with **direct visual embedding prototype retrieval in vector space** using **EmbeddingGemma 2 (768D)** and **FAISS**.

### Key Evolution: Hierarchical Multi-Font EmbeddingOCR
1. **Tier 1 (Word Lexicon)**: High-confidence direct full-word image-to-prototype matching (handles non-segmentable cursive scripts and common vocabulary in single inference).
2. **Tier 2 (Subwords / BPE)**: Sub-lexical components matched when full word is not in lexicon.
3. **Tier 3 (Characters)**: Vertical projection segmentation + character glyph prototype retrieval as robust fallback.
4. **Multi-Font Visual Prototypes (Solution 1)**:
   - For real-world handwriting (IAM Handwriting Dataset), text-string embeddings alone fail to capture cursive stroke variations.
   - We generate visual prototype embeddings across multiple handwriting fonts (`LiberationSans`, `DancingScript`, `KaushanScript`, `SteveHand`, `Kristi`) with affine shear/slant angles.
   - Vector retrieval applies **max-similarity aggregation per candidate token class** over top-k hits.

---

## 2. Completed Milestones

- [x] **Phase 0: A-Z Spike**
  - Text-to-text, image-to-text cross-modal, and image-to-image glyph matching verified.
  - Matryoshka dimension scaling (768, 512, 256, 128) evaluated.
- [x] **Phase 1 & 2: Core Abstractions & Pipeline Setup**
  - `RecognitionEngine`, `RecognitionResult`, `Embedder` protocol, `VectorIndex` protocol (`FaissIndex`, `NumpyIndex`).
  - Preprocessing, horizontal crop projection segmentation, baseline reconstruction.
- [x] **Phase 3 & Synthetic Word Benchmark vs GLM-OCR**
  - Hierarchical word + subword + character multi-tier matching implemented.
  - Outperformed GLM-OCR on synthetic rendered crops (**Norm CER: 12.83% vs 37.88%**).
- [x] **IAM Handwriting Dataset Integration**
  - Benchmark script created (`scripts/test_iam_handwriting.py`) evaluating real-world human handwriting crops from `priyank-m/IAM_words_text_recognition`.
  - Baseline single-font hierarchical result recorded: **84.14% Norm CER**, **32.0% Norm Acc** (vs GLM-OCR **5.73% Norm CER**, **84.0% Norm Acc**).
- [x] **Handwriting Fonts & Visual Prototypes Engine (Solution 1)**
  - System handwriting fonts installed: `fonts-dancingscript`, `fonts-kaushanscript`, `fonts-kristi`, `fonts-sjfonts`.
  - Fast SentencePiece token extraction implemented in `vocabulary.py` (down from 30m stall to 6.8s).
  - Multi-font prototype rendering with slant augmentation and max-similarity aggregation implemented in `src/embedding_ocr/recognition/embedding/hierarchical/vocabulary.py`.
  - Staged, tested (pytest, ruff, mypy green), committed (`7ae563a`), and pushed to GitHub `origin/main`.

---

## 3. Active Running State (Colab / Runtime)

- **Active Process**: Multi-font visual prototype generation & IAM handwriting benchmark.
  - Command:
    ```bash
    uv run python scripts/test_iam_handwriting.py \
        --num-samples 50 \
        --gemma-dim 768 \
        --skip-glm \
        --use-visual-prototypes \
        --max-words 300 \
        --max-subwords 100
    ```
- **Cache Location**: `artifacts/vocab_cache_iam_multifont/`
  - Tier 1 Words: `words_visual_c353c32d_768d.npy` (1,500 prototype embeddings, cached)
  - Tier 2 Subwords: `subwords_visual_c353c32d_768d.npy` (500 prototype embeddings, cached)
  - Tier 3 Characters: `chars_visual_c353c32d_768d.npy` (475 prototype embeddings, generating)
- **Monitoring Cadence**: 360-second status check cycle.

---

## 4. Immediate Next Steps (To Resume In Case Session Ends)

1. **Verify / Run Benchmark with Cached Multi-Font Prototypes**:
   - The multi-font prototypes in `artifacts/vocab_cache_iam_multifont/` are persistent.
   - Run or inspect:
     ```bash
     uv run python scripts/test_iam_handwriting.py \
         --num-samples 50 \
         --gemma-dim 768 \
         --skip-glm \
         --use-visual-prototypes \
         --max-words 300 \
         --max-subwords 100
     ```
2. **Record Multi-Font Results in Reports**:
   - Compare new Normalized CER against single-font baseline (**84.14%**).
   - Update `docs/iam_handwriting_benchmark_report.md` and `README.md`.
3. **Explore Solution 2 & Solution 3 if Needed**:
   - **Solution 2 (Image Preprocessing & Normalization)**: Deslanting (Slant correction via shear angle optimization), Otsu adaptive thresholding, and skeleton thinning before embedding.
   - **Solution 3 (Lightweight Contrastive Fine-Tuning / LoRA)**: Fine-tune `EmbeddingGemma 2` vision tower projection head on IAM handwriting word crops using InfoNCE loss.
4. **Git Sync**:
   - Ensure `uv run ruff check .`, `uv run mypy src`, `uv run pytest` pass.
   - Commit and push report updates to `main`.

---

## 5. Multilingual Segmentation Roadmap (Active)

**Scope:** Arabic/connected scripts and Latin cursive. **Goal:** separate
line→word and word→unit segmentation stages. **Constraint:** inference-first;
YOLO character/unit detector training is available through the `train` extra
(see section 7).

### 5.1 Foundational fixes (done)
- `src/embedding_ocr/data/scripts.py`: Unicode-range script detection,
  script aliases, RTL detection, reading-order helper.
- Arabic character classes (`ARABIC_LETTERS`, `ARABIC_CLASSES`,
  `classes_for_script`) in `data/glyph_renderer.py`.
- `PrototypeDB` is script-aware: `script="latin"|"arabic"`, classes default
  from the script, and `script` is persisted in metadata.
- `PrototypeMatcher` only applies Latin case-pair logic to Latin letters and
  disables it for RTL scripts (Arabic has no case).
- RTL reading order added to `CharacterSegmenter`, `VisualWordSegmenter`,
  `CraftWordDetector`, `HierarchicalWordRecognizer`, and
  `EmbeddingRecognizer`; recursive subword cuts now return crops in logical
  order.
- Latin-only physical length estimates are skipped for non-Latin scripts.

### 5.2 Remaining phases
1. **Probe detectors/embedders.** Compare PaddleOCR PP-OCRv5 det, Surya det,
   CRAFT, and EasyOCR on small Arabic + IAM samples; verify EmbeddingGemma 2
   Arabic retrieval against Jina v4 / SigLIP. Run
   `scripts/probe_segmentation.py`; it writes
   `docs/segmentation_probe.json` with box counts, top-1/top-5 retrieval, and
   latency. Full word-box F1 requires annotated samples and is deferred.
2. **Line→word adapters.** Add a `WordSegmenter` protocol and backends:
   PaddleOCR/Surya line det + whitespace split, CRAFT/EasyOCR word boxes, and
   the improved gap segmenter. Keep current behavior as the default.
3. **Word→unit segmentation.** Arabic: diacritic merging, positional-form
   prototypes, RTL grapheme handling. Latin cursive: watershed/seam carving
   for touching glyphs plus embedding-guided Viterbi alignment for OOV words.
4. **Multilingual prototypes/vocabulary.** The ASCII-only filter is removed;
   `EmbeddingVocabulary` is script-aware (Arabic fonts, token filtering,
   cache tags, RTL rendering) and `EmbeddingAlignmentSegmenter` plus
   `ArabicUnitSegmenter` provide word→unit alignment. Still to compare text vs
   visual prototypes and verify tokenizer coverage on real Arabic data.
5. **Benchmark.** Synthetic Arabic/Latin-cursive renders with held-out fonts,
   plus IAM and Arabic datasets (MADAR, IFN/ENIT, KHATT, AHCD; verify
   annotations). Metrics: word-box F1@IoU 0.5, boundary F1, reading-order
   accuracy, unit accuracy, CER/WER, latency, GPU memory.

### 5.3 Deferred
- CJK / Thai / Indic scriptio-continua segmentation.
- Custom DBNet++/FAST/U-Net training. A YOLO character/unit path is available
  instead; see section 7.
- Pretrained CTC aligner experiments (PaddleOCR/EasyOCR internals).

---

## 6. Implementation status (November 2026)

- **Phase 1 — complete:** typed Pydantic config (`config.py`), default and
  backend-specific YAML configs, Typer CLI (`embedding-ocr version`,
  `show-config`, `recognize`, `build-prototypes`, `build-vocabulary`,
  `benchmark`), Makefile, pre-commit, GitHub Actions, `.gitattributes`.
- **Phase 2 — complete:** `TextDetector` protocol with full/visual/CRAFT/Paddle
  backends, region cropping, line grouping/reconstruction, `OCRPipeline`,
  on-disk detection cache, and a config-driven recognizer/embedder factory.
  Prototype/index dim and embedder-name mismatches raise on load.
- **Phase 3 — adapter ready:** `PaddleRecognizer` and a Paddle detection
  adapter exist behind config. Reproducible baseline numbers still require the
  Paddle runtime and a fixed dataset manifest.
- **Phase 4 — MVP implemented:** character segmentation, FAISS prototype
  matching, hierarchical word/subword/char fallback, script-aware vocabulary,
  and CLI prototype/vocabulary builders.
- **Phase 5 — implemented:** `ArabicUnitSegmenter` and
  `EmbeddingAlignmentSegmenter` with cost-minimizing Viterbi DP. Config wiring
  is available through `recognition.use_alignment`.
- **Phase 6 — runner ready:** manifest-driven CER/WER/exact-match benchmark
  (`embedding-ocr benchmark`) with per-sample latency. Full fair comparison
  still needs held-out real datasets and detector annotations.
- **Phase 7 — in progress:** CI, Makefile, pre-commit, docs, and CLI are in
  place; the `v0.1.0` release tag is pending.

---

## 7. YOLO character/unit detection (optional)

- Dataset generator: `scripts/build_letter_dataset.py` renders Latin and Arabic
  words with HarfBuzz shaping and writes YOLO labels per grapheme cluster.
- Fonts are split before generation; train/val/test never share a font.
- Training: `scripts/train_yolo_units.py` targets Ultralytics YOLO26 at 1280
  input resolution with text-safe augmentations.
- Inference: `recognition.backend: yolo_units` reads classes directly from the
  detector, groups boxes into lines, and joins text in reading order.
- Example:

```bash
bash scripts/setup_env.sh --train
uv run python scripts/build_letter_dataset.py \
    --fonts /usr/share/fonts --out artifacts/yolo_letters \
    --n 50000 --size 1280 --layout line
uv run python scripts/train_yolo_units.py \
    --data artifacts/yolo_letters/data.yaml
uv run embedding-ocr recognize image.png --config configs/yolo_units.yaml
```

- Ultralytics YOLO is AGPL-3.0 or commercially licensed; review that before
  distributing weights or integrating the backend.

Validation in this iteration: `ruff`, `mypy src`, and 133 tests pass. Optional
model/detector probes report missing dependencies rather than crashing.
