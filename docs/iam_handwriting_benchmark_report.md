# Benchmark Report: IAM Handwriting Dataset (EmbeddingOCR vs GLM-OCR)

> **Date:** November 3, 2026  
> **Dataset:** IAM Handwriting Word Crops (`priyank-m/IAM_words_text_recognition`, test split)  
> **Evaluation Mode:** Real human handwriting (variable cursive slants, ligatures, strokes)  
> **Primary Evaluation Metric:** **Normalized CER** (case-insensitive, whitespace-trimmed)  
> **Hardware:** Tesla T4 GPU (CUDA)  

---

## 1. Executive Summary

To stress-test model robustness beyond synthetic printed fonts, we evaluated **GLM-OCR** and **Hierarchical EmbeddingOCR (768D)** on real human handwriting word crops from the IAM test split.

Unlike synthetic printed benchmarks where models achieved near-perfect CER, the IAM Handwriting benchmark proves significantly harder:
- **GLM-OCR's** 100% clean accuracy drops: Normalized CER rises from **0.00% to 5.73%** (84.0% word accuracy).
- **Hierarchical EmbeddingOCR (768D)** exhibits a stark domain gap: Normalized CER rises to **84.14%** (32.0% word accuracy) because EmbeddingGemma 2's prototypes were indexed exclusively against standard clean digital fonts (`LiberationSans`, `DejaVuSans`, etc.).

---

## 2. Benchmark Results

| Recognizer | Primary Word Norm CER (%) | Word Raw CER (%) | Word Norm Acc (%) | Word Raw Acc (%) | Norm WER (%) | Avg Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **GLM-OCR** (`zai-org/GLM-OCR`) | **5.73%** | **6.17%** | **84.0%** | **82.0%** | **18.00%** | 1,144.3 |
| **Hierarchical Multi-Font + Deslanted (768D)** *(Sol 1 + Sol 2)* | **64.32%** | **65.64%** | **48.0%** | **46.0%** | **82.00%** | 1,815.8 |
| **Hierarchical Multi-Font (768D)** *(Solution 1)* | **62.11%** | **62.11%** | **40.0%** | **40.0%** | **90.00%** | 1,760.6 |
| **Hierarchical Baseline (768D)** *(Single Font)* | **84.14%** | **91.19%** | **32.0%** | **20.0%** | **104.00%** | 2,516.5 |

> **Key Observations:**
> 1. **Word Accuracy Jump:** Combining Multi-Font Prototypes with Deslanting & Binarization surged exact word recognition accuracy to **48.0%** (up from **32.0%** baseline and **40.0%** without deslanting).
> 2. **Newly Solved Words:** Cursive words that previously failed completely now match cleanly at 100% word confidence:
>    - GT `"For"` $\to$ Pred `"For"` (previously mispredicted as `"ctolpp"`)
>    - GT `"new"` $\to$ Pred `"new"` (previously mispredicted as `"mem"`)
>    - GT `"which"` $\to$ Pred `"which"` (previously mispredicted as `"ri"`)
>    - GT `"found"` $\to$ Pred `"found"` (previously mispredicted as `"form"`)
> 3. **Normalized Word Error Rate (WER):** Decreased significantly from **104% $\to$ 90% $\to$ 82%**.



---

## 3. Error Analysis & Comparative Observations

### 3.1 GLM-OCR Real-World Behavior
On real handwriting, GLM-OCR's autoregressive decoder makes characteristic handwriting errors:
- **Extra strokes mistaken for letters:** GT `"It"` $\to$ Pred `"Itr"`
- **Ambiguous stroke ligatures:** GT `"an"` $\to$ Pred `"am"`
- **Suffix truncation:** GT `"man"` $\to$ Pred `"ma"`
- **Hallucinated phrase insertion:** GT `"familiar"` $\to$ Pred `"family has"`

Despite these challenges, GLM-OCR maintains strong overall reading capability (84% accuracy) thanks to its end-to-end vision encoder trained on diverse documents.

### 3.2 Hierarchical EmbeddingOCR Domain Gap
1. **The Font Prototype Gap:**
   - EmbeddingOCR matches query image embeddings against digital font prototypes.
   - For cursive, slanted, and connected handwriting (e.g. `"new"`, `"endeavour"`, `"which"`), the visual shape representation in EmbeddingGemma 2 space is fundamentally different from a digital typeface.
2. **Successes on Distinct Word Shapes:**
   - Distinctive words with clear visual silhouettes still matched accurately at the whole-word level:
     - GT `"little"` $\to$ Pred `"little"` (Match)
     - GT `"happened"` $\to$ Pred `"happened"` (Match)
     - GT `"exhausted"` $\to$ Pred `"exhausted"` (Match)
   - When whole-word matching missed, subword and character fallbacks struggled because cursive letters cannot be cleanly segmented by standard vertical projection.

---

## 4. Key Takeaways & Recommendations

1. **IAM Handwriting successfully breaks the "100% accuracy ceiling":** It provides a genuine test of real-world generalization.
2. **Path to Improve EmbeddingOCR for Handwriting:**
   - Indexing with **calligraphic / handwriting prototypes** (e.g., cursive fonts or representative handwriting prototype clusters).
   - Fine-tuning EmbeddingGemma 2 contrastively on IAM word pairs to map cursive strokes closer to standard typographic word embeddings.

---

## 5. Artifacts and Reproducibility

- Benchmark Script: [`scripts/test_iam_handwriting.py`](../scripts/test_iam_handwriting.py)
- Full Output JSON: [`docs/iam_handwriting_benchmark_report.json`](iam_handwriting_benchmark_report.json)
- Run command:
  ```bash
  uv run python scripts/test_iam_handwriting.py --num-samples 50 --gemma-dim 768
  ```
