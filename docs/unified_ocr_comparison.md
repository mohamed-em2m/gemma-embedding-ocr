# Unified OCR Algorithm & Model Benchmark Comparison

Queries: **1300** rendered glyphs from held-out typography fonts.

| Category | Algorithm / Engine | Status | Exact Top-1 | Case-Insens | Extracted Glyph | Latency (ms) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| EmbeddingOCR | EmbeddingGemma (768d, Max-Agg + Matcher) | **ok** | 76.0% | 98.8% | 76.0% | 754.9 ms |
| EmbeddingOCR | EmbeddingGemma (768d, 1-NN) | **ok** | 92.3% | 98.8% | 92.3% | 751.6 ms |
| EmbeddingOCR | EmbeddingGemma (768d, 5-NN Weighted) | **ok** | 83.8% | 98.8% | 83.8% | 753.3 ms |
| EmbeddingOCR | SigLIP SO400M (1152d, Max-Agg + Matcher) | **ok** | 74.0% | 98.8% | 74.0% | 233.9 ms |
| EmbeddingOCR | jina | *unavailable* | - | - | - | - |
| Classical CV | Pixel NCC Template Match | **ok** | 30.8% | 37.1% | 30.8% | 14.2 ms |
| Classical CV | Structural Similarity (SSIM) | **ok** | 26.5% | 36.5% | 26.5% | 92.5 ms |
| Conventional OCR | tesseract | **ok** | 77.7% | 83.4% | 77.7% | 220.0 ms |
| Conventional OCR | easyocr | **ok** | 62.4% | 69.5% | 62.4% | 16.9 ms |
| Conventional OCR | paddle | **ok** | 86.1% | 92.3% | 86.1% | 95.5 ms |
| Generative VLM | lighton | **ok** | 71.0% | 84.2% | 82.2% | 1761.2 ms |
| Generative VLM | surya | *unavailable* | - | - | - | - |
| Generative VLM | glm | **ok** | 85.7% | 96.4% | 85.7% | 389.8 ms |
| Generative VLM | oxr | **ok** | 71.9% | 84.2% | 74.4% | 3555.5 ms |

## Unavailable Engines / Services

- **jina** (EmbeddingOCR): `AttributeError: 'NoneType' object has no attribute 'convert_tokens_to_ids'`
- **surya** (Generative VLM): `ImportError: surya-ocr is not installed. Install it with uv sync --extra surya or uv pip install surya-ocr, then ensure a vLLM (CUDA) or llama.cpp (CPU) backend is available or point SURYA_INFERENCE_URL at a running server.`

Notes: `Extracted Glyph` evaluates robust string normalization from chatty generative outputs. `Exact Top-1` requires direct character equality. This 1300-query report supersedes the earlier 10-query pilot, whose raw results remain in `docs/unified_ocr_comparison.json`.
