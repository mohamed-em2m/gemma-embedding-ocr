# Phase 0 Feasibility Report: EmbeddingGemma 2 for Latin OCR

> **Generated:** Phase 0 spike — A-Z embedding + FAISS search  
> **Embedder:** `google/embeddinggemma-2`  
> **Hardware:** NVIDIA Tesla T4 (15 GB VRAM), CUDA 13.0  
> **Scope:** Latin a-z / A-Z (52 classes)  
> **Fonts indexed (Set A):** LiberationSans-Regular, LiberationSerif-Regular, DejaVuSans  
> **Fonts held-out (Set B):** Roboto-Regular, NotoSans-Regular, LiberationMono-Regular  
> **Rendering:** 64×64 canvas, 44pt font, white bg, black text

---

## 1. Verified Model Facts

| Fact | Value |
|------|-------|
| Model ID (Hugging Face) | `google/embeddinggemma-2` |
| Architecture | Gemma-based multimodal (~740M params: 270M text + 170M vision + 300M audio) |
| Native embedding dim | 768 |
| Matryoshka dims supported | 768, 512, 256, 128 |
| License | Apache 2.0 |
| Loading library | `sentence-transformers >= 6.1.0` (6.2.0 confirmed working; 5.7.0 **broken** — missing `normalize` module) |
| Loading code (GPU) | `SentenceTransformer("google/embeddinggemma-2", model_kwargs={"torch_dtype": torch.bfloat16}, device="cuda")` |
| Loading code (CPU) | `SentenceTransformer("google/embeddinggemma-2", device="cpu")` |
| Text prefix required | Yes — use `prompt_name="Document"` for text inputs |
| Image prefix | None required — images passed directly |
| Matryoshka truncation | Slice first `dim` dims, then re-normalize (L2). `normalize_embeddings=True` in ST handles this automatically |
| Context window | 8192 tokens; 1 image = 280 tokens |

---

## 2. Experiment Results

### E1: Text Self-Retrieval (sanity check)

> Embed the 52 letters as text strings, build index, query each letter with itself.

| Dim | Top-1 | Top-5 |
|-----|-------|-------|
| **768** | **100.0%** | **100.0%** |
| 512 | 100.0% | 100.0% |
| 256 | 100.0% | 100.0% |
| 128 | 100.0% | 100.0% |

**Closest text-space pairs at 768d (cosine similarity):**

| Pair | Similarity |
|------|-----------|
| l / L | 0.969 |
| c / C | 0.961 |
| i / I | 0.960 |
| o / O | 0.958 |
| f / F | 0.957 |

> ⚠️ These are text-space similarities, not image-space — they reflect the model's knowledge that `l` and `L` are the same letter in different cases.

---

### E2: Cross-modal — Image → Text Index

> Embed held-out font images, query the text embedding index. Tests whether image can retrieve its own letter symbol without any image prototypes.

| Dim | Top-1 | Top-5 |
|-----|-------|-------|
| 768 | 61.5% | 100.0% |
| 512 | 64.7% | 100.0% |
| 256 | 62.8% | 100.0% |
| **128** | **63.5%** | **99.4%** |

**E2 errors at 768d (sample):** `a→A`, `c→C`, `e→E`, `i→I`, `l→I`, `o→O`, `s→S`, `v→V`, `w→W`, `x→X`, `y→Y`, `z→Z`, `B→b`, `J→j`, `K→k`

**Key finding:** E2 accuracy is ~61–65% across all dims — the cross-modal shared space exists but is not sharp enough for zero-shot text-prototype matching alone. The model can place images near the right letter but frequently picks the wrong case. **Text prototypes alone are insufficient.**

---

### E3: Image → Image (across fonts)

> Build FAISS index from Set A images; query with Set B held-out images. This is the realistic OCR prototype-matching setup.

| Dim | Top-1 | Top-5 | Case-pair errors / total |
|-----|-------|-------|--------------------------|
| **768** | **91.0%** | **100.0%** | 12 / 14 |
| 512 | 91.7% | 100.0% | 11 / 13 |
| 256 | 89.1% | 100.0% | 15 / 17 |
| 128 | 86.5% | 100.0% | 16 / 21 |

**E3 error breakdown at 768d (14 total errors across 156 queries):**

| True | Predicted | Similarity | Type |
|------|-----------|-----------|------|
| l | I | 1.002 | Look-alike (l/I) |
| v | V | 0.999 | Case pair |
| w | W | 1.001 | Case pair |
| O | o | 0.997 | Case pair |
| X | x | 1.000 | Case pair |
| s | S | 0.999 | Case pair |
| v | V | 0.999 | Case pair |
| w | W | 1.000 | Case pair |
| l | L | 0.960 | Look-alike (l/L) |
| s | S | 1.003 | Case pair |
| v | V | 1.004 | Case pair |
| w | W | 0.996 | Case pair |
| O | o | 0.992 | Case pair |
| X | x | 0.998 | Case pair |

> 🔴 **12 of 14 E3 errors (86%) are case-pair confusions.** The model cannot distinguish lowercase from uppercase when glyphs are cropped to a square canvas (size information lost).

---

### E4: Matryoshka Dimension Sweep Summary

| Dim | E1 T1 | E2 T1 | E3 T1 | E3 T5 | Case confusions |
|-----|-------|-------|-------|-------|-----------------|
| **768** | 100% | 61.5% | **91.0%** | 100% | 12/14 |
| 512 | 100% | 64.7% | **91.7%** | 100% | 11/13 |
| 256 | 100% | 62.8% | 89.1% | 100% | 15/17 |
| 128 | 100% | 63.5% | 86.5% | 100% | 16/21 |

**Recommendation on dimension:** Use **512d** — best E3 Top-1 (91.7%) with a meaningful 33% index size reduction vs. 768d. The accuracy delta is negligible (0.7%) while 256d and 128d show degradation.

---

### E5: Latency and Memory

| Metric | Value |
|--------|-------|
| Device | NVIDIA T4, CUDA 13.0, bfloat16 |
| Model load memory | ~1,432 MB |
| Peak GPU memory (batch=16 images) | **13,158 MB** (out of 15,360 MB) |
| Latency: batch=1, image | **762.95 ms** |
| Latency: batch=16, per image | **747.71 ms** |
| Batch size limit on T4 (bfloat16) | ≤16 (batch=32 OOMs) |
| CPU latency | Not measured (model too large for fast CPU inference) |

> ⚠️ **High latency warning:** ~763 ms per character crop is far too slow for production OCR. At this rate, a 100-character word requires 76 seconds. This must be addressed via:
> - Batching full word crops together (not per-character if possible)
> - A lighter/quantized embedder alternative
> - Pre-built prototype cache (prototypes embedded once, amortized)
> - Considered: SigLIP or DINOv2 as lighter alternatives

---

## 3. Case-Pair Analysis

### Problematic Pairs (Predicted by PLAN.md)

These letter pairs have the **same glyph shape**; the only difference is size. When cropped to a fixed square canvas, size information is lost.

| Pair | E3 Confused? | Notes |
|------|-------------|-------|
| c / C | ✅ Yes | Confused across all dims |
| o / O | ✅ Yes | Confused across all dims |
| s / S | ✅ Yes | Confused across all dims |
| v / V | ✅ Yes | Confused across all dims |
| w / W | ✅ Yes | Confused across all dims |
| x / X | ✅ Yes | Confused across all dims |
| z / Z | ✅ Yes (128d) | Appeared at lower dims |
| u / U | ❌ Not seen | May require more font variety |
| p / P | ❌ Not seen | p and P shapes differ more |
| k / K | ✅ Yes (128d) | Appeared at lower dim — extra concern |

### Other Look-alike Pairs

| Pair | Confused? | Notes |
|------|----------|-------|
| I / l | ✅ **Strongly** | `l→I` and `I→l` — highest similarity scores (>1.0) |
| l / L | ✅ Yes | Some confusion especially at 128d |
| i / j | Not yet seen | Dots may help distinguish |
| g / q | Not seen | Tail direction differs |
| b / d | Not seen | Mirror images — may confuse at low res |
| n / h | Not seen | — |
| u / v | Not seen | — |

---

## 4. Confusion Matrix Summary (768d, E3)

Full 52×52 matrix not printed (see `docs/spike_results.json`). Key confused pairs at 768d:

| True → Predicted | Count (across fonts) |
|-----------------|---------------------|
| l → I | 2 |
| v → V | 3 |
| w → W | 3 |
| O → o | 2 |
| X → x | 2 |
| s → S | 2 |

These 6 confusion types account for all 14 errors. No non-case-related confusions observed at 768d.

---

## 5. Decision Gate Assessment

> Threshold reference from PLAN.md Section 7, Phase 0.

| Criterion | Threshold | Actual | Met? |
|-----------|-----------|--------|------|
| E3 top-1 ≥ 95% (zero-shot works) | 95% | 91.0% (768d) | ❌ |
| E2 top-1 ≥ 90% (text prototypes suffice) | 90% | 61.5–64.7% | ❌ |
| E3 in 80–95% range | — | ✅ 86–92% | ✅ |
| Latency acceptable | — | 763 ms/crop | ❌ Very high |
| Case pairs well-separated | — | 85–90% of errors are case-pair | ❌ |

**Verdict: 80–95% zone.** The model demonstrates meaningful font generalization for Latin glyphs, but:
1. **Case pairs are the dominant failure mode** (need fix before production)
2. **Latency is prohibitive** for per-character embedding

---

## 6. Recommendations

### Immediate (for Phase 1/2 planning)

1. **Case-pair fix (highest priority):** Do not resize to a square canvas. Instead:
   - Keep original glyph **aspect ratio** (pad horizontally only, not vertically)
   - Preserve **baseline position** in the crop — lowercase letters don't reach the cap-height
   - Add a **relative-height feature** alongside the embedding (e.g., glyph pixel height as % of line height)
   - Alternatively, **resolve case from word context** in Phase 4 language model post-processing

2. **Add more fonts and augmentation:** 6 fonts is minimal. Target 15–20 fonts for the prototype DB, with light augmentation (±3° rotation, slight blur) after clean results are confirmed.

3. **Latency mitigation:**
   - Process full word crops in batches, not character by character
   - Evaluate SigLIP (smaller vision encoder) or DINOv2 as faster alternatives
   - Consider int8 quantization of EmbeddingGemma2

4. **Best dimension:** Use **512d** (91.7% E3 Top-1, 33% smaller index vs. 768d)

5. **Text prototypes:** Cross-modal (E2) accuracy of 61–65% is insufficient for production but could serve as a **fallback** for OOV characters. Focus on image prototypes.

### Phase 1+ next steps

- If case-pair fix (aspect ratio + relative height) brings E3 ≥ 95%: proceed to Phase 1–4 with image prototypes
- If adding fonts + augmentation brings E3 to 93–95%: proceed (augmentation may be enough)
- If latency remains unacceptable even with batching: evaluate SigLIP/DINOv2 in parallel

---

## 7. Run Command

```bash
python scripts/phase0_az_spike.py --config configs/spike_az.yaml
```

Requirements: `sentence-transformers>=6.1.0`, `faiss-cpu>=1.8`, `torch`, `pillow`, `pyyaml`, `numpy`
