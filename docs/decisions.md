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
