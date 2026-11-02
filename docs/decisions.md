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

Added three document-OCR VLMs to the cross-engine comparison:

| Model | Loading path | Status |
| --- | --- | --- |
| `lightonai/LightOnOCR-3-0.8B` | `transformers>=5.5.4`, `Qwen3_5ForConditionalGeneration` | Verified from model card; transcription prompt is `""` |
| `dots-studio/dots.mocr` | `AutoModelForCausalLM` + `AutoProcessor`, `trust_remote_code=True` | Verified from model card; single-glyph prompt is **unverified** |
| `SoMarkAI/OXR-1.0` | OpenAI-compatible HTTP server (out-of-tree vLLM plugin) | Local `transformers` inference is not documented; adapter talks to a separately started vLLM server |

- Shared benchmark: `scripts/compare_new_ocr_models.py`.
- All engines use the same rendered glyph queries (default: 52 Latin classes,
  3 held-out Liberation fonts) and report exact, case-insensitive, and
  extracted-glyph accuracy plus per-item latency.
- Unavailable engines are recorded in the JSON/Markdown report instead of
  aborting the run, because the three VLMs have different hardware and
  serving requirements.
- OXR is a document parser served through vLLM; its default prompt and model
  name must be supplied by the user when the server uses different values.
