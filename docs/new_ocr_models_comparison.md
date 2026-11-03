# EmbeddingGemma 2 vs generative OCR VLMs

Queries: **156** rendered Latin glyphs from held-out fonts.

| Engine | Status | Exact | Case-insensitive | Extracted glyph | Latency |
| :--- | :--- | :---: | :---: | :---: | :---: |
| EmbeddingGemma 2 | ok | 78.8% | 100.0% | 78.8% | 766.1 ms |
| lighton | ok | 60.9% | 73.7% | 82.7% | 2992.2 ms |
| dots | ok | 0.0% | 0.0% | 1.9% | 582.9 ms |
| oxr | unavailable | - | - | - | - |

## Unavailable engines

- **oxr**: `RuntimeError: OXR request failed at http://127.0.0.1:8080/v1. Start an OpenAI-compatible OXR/vLLM server first: HTTP Error 404: Not Found`

`Extracted glyph` accepts a single character recovered from a chatty or JSON response; `Exact` requires the cleaned response to equal the target character.
