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
