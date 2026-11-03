# Unified OCR Algorithm & Model Benchmark Comparison

Queries: **10** rendered glyphs from held-out typography fonts.

| Category | Algorithm / Engine | Status | Exact Top-1 | Case-Insens | Extracted Glyph | Latency (ms) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Conventional OCR | tesseract | **ok** | 70.0% | 70.0% | 70.0% | 228.3 ms |
| Conventional OCR | easyocr | **ok** | 60.0% | 60.0% | 60.0% | 131.7 ms |
| EmbeddingOCR | EmbeddingGemma (Max-Agg + Matcher) | **ok** | 90.0% | 100.0% | 90.0% | 752.5 ms |

Notes: `Extracted Glyph` evaluates robust string normalization from chatty generative outputs. `Exact Top-1` requires direct character equality.
