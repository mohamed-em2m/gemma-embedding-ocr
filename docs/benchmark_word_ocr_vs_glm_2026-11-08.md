# Benchmark Report: Line Detection + Improved Viterbi OCR vs GLM-OCR

> **Date:** November 8, 2026  
> **Benchmark:** Synthetic word and phrase crops under augmentation  
> **Pipeline variant:** line detection for text-region/word proposal and improved Viterbi alignment decoding  
> **Primary metric:** Normalized CER (case-insensitive, whitespace-trimmed; lower is better)  
> **Source JSON:** [`docs/word_ocr_vs_glm_report_2026-11-08.json`](word_ocr_vs_glm_report_2026-11-08.json)  
> **Run command:** `python scripts/test_word_ocr_vs_glm.py --gemma-dim 768 --augment`

Hardware and runtime environment were not recorded in the supplied JSON. The JSON also does not record the line-detection or Viterbi parameters; those settings are noted here from the experiment configuration. Latency comparisons are therefore valid only within this run.

## 1. Executive Summary

Three recognizers were evaluated on augmented synthetic word and phrase crops:

1. **GLM-OCR** (`zai-org/GLM-OCR`): generative vision-language OCR baseline.
2. **Hierarchical EmbeddingOCR (768D)**: word, subword, and character embedding matching with `google/embeddinggemma-2`.
3. **Character EmbeddingOCR (768D)**: connected-component segmentation followed by character-prototype matching.

GLM-OCR was perfect on this synthetic benchmark: **0.00% normalized CER** and **100% word/phrase accuracy**. Hierarchical EmbeddingOCR was the strongest embedding-based variant, reaching **4.12% word normalized CER** and **80% normalized word accuracy**, but degraded to **10.79% phrase normalized CER**. Character-level EmbeddingOCR performed worst and was substantially slower.

### 1.1 Method Note

This run uses line detection to propose text-line/word regions before recognition and an improved Viterbi alignment decoder for segmentation-free unit selection. GLM-OCR is evaluated as the end-to-end generative baseline. Exact line-detection thresholds, Viterbi edge costs, and model paths were not present in the supplied JSON.

## 2. Results

The evaluation covered **75 word crops** (25 words × 3 test fonts) and **18 phrase crops** (6 phrases × 3 test fonts).

### 2.1 Word-Level Metrics

| Recognizer | Norm CER | Raw CER | Norm WER | Norm Acc | Raw Acc | Avg latency |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **GLM-OCR** | **0.00%** | **0.00%** | **0.00%** | **100.0%** | **100.0%** | **435.8 ms** |
| **Hierarchical EmbeddingOCR (768D)** | 4.12% | 9.41% | 20.00% | 80.0% | 44.0% | 736.5 ms |
| **Character EmbeddingOCR (768D)** | 17.25% | 29.02% | 69.33% | 30.7% | 18.7% | 5019.6 ms |

### 2.2 Phrase-Level Metrics

| Recognizer | Norm CER | Raw CER | Norm WER | Norm Acc | Raw Acc | Avg latency |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **GLM-OCR** | **0.00%** | **0.00%** | **0.00%** | **100.0%** | **100.0%** | **965.1 ms** |
| **Hierarchical EmbeddingOCR (768D)** | 10.79% | 18.94% | 27.78% | 33.3% | 0.0% | 2316.1 ms |
| **Character EmbeddingOCR (768D)** | 19.42% | 30.94% | 77.78% | 0.0% | 0.0% | 16117.8 ms |

### 2.3 Latency Comparison

| Recognizer | Word latency | Relative to GLM | Phrase latency | Relative to GLM |
| :--- | ---: | ---: | ---: | ---: |
| **GLM-OCR** | 435.8 ms | 1.00× | 965.1 ms | 1.00× |
| **Hierarchical EmbeddingOCR (768D)** | 736.5 ms | 1.69× slower | 2316.1 ms | 2.40× slower |
| **Character EmbeddingOCR (768D)** | 5019.6 ms | 11.52× slower | 16117.8 ms | 16.70× slower |

## 3. Error Analysis

### 3.1 GLM-OCR

GLM-OCR matched every logged sample exactly, including multi-word phrases and the `2026` token. On this synthetic test set, its language prior and end-to-end decoder remove character-level ambiguity.

### 3.2 Hierarchical EmbeddingOCR (768D)

- **Word truncation:** `service → ervice`, `system → syst`, `neural → neur`.
- **Capitalization differences:** `total → Total`, `customer → Customer`, `payment → Payment`, `network → Network`, `python → Python`. These do not affect normalized accuracy.
- **Phrase errors:** `customer account balance → Customer account balanced` (extra character) and `optical character recognition → Opt Character recogn` (abbreviation/truncation).
- **Phrase recall:** only **33.3%** normalized phrase accuracy, showing that small word-level errors and boundary decisions compound across longer text.

### 3.3 Character EmbeddingOCR (768D)

Errors are dominated by visually similar glyph pairs:

- `0` / `o`: `invoice → 0Invo0Ice`
- `I` / `l`: `total → totaI`, `digital → d0IgItaI`
- `M` / `m`: `customer → custoMel`, `system → systeM`
- `K` / `k`: `network → newolK`
- `S` / `s`, `V` / `v`: `digital document service → d0IgI0taI document SelVICe`
- `r` / `l` and connected-stroke confusion: `neural → neulaI`, `optical character recognition → opt0IcaI chalactel lecognIt0Ion`

The segmentation path is also slow: **5.0 s per word** and **16.1 s per phrase** in this run.

## 4. Sample Predictions

| Type | Ground truth | GLM-OCR | Hierarchical (768D) | Character (768D) |
| :--- | :--- | :--- | :--- | :--- |
| word | `invoice` | `invoice` | `invoice` | `0Invo0Ice` |
| word | `total` | `total` | `Total` | `totaI` |
| word | `customer` | `customer` | `Customer` | `custoMel` |
| word | `payment` | `payment` | `Payment` | `payment` |
| word | `service` | `service` | `ervice` | `selv0Ice` |
| word | `network` | `network` | `Network` | `newolK` |
| word | `system` | `system` | `syst` | `systeM` |
| word | `neural` | `neural` | `neur` | `neulaI` |
| word | `python` | `python` | `Python` | `python` |
| word | `digital` | `digital` | `digital` | `d0IgItaI` |
| phrase | `invoice total 2026` | `invoice total 2026` | `invoice Total 2026` | `0InvoIce totaI 2026` |
| phrase | `customer account balance` | `customer account balance` | `Customer account balanced` | `customel account baIance` |
| phrase | `optical character recognition` | `optical character recognition` | `Opt Character recogn` | `opt0IcaI chalactel lecognIt0Ion` |
| phrase | `neural network system` | `neural network system` | `neur Network System` | `neulaI netwolK systeM` |
| phrase | `digital document service` | `digital document service` | `digital Document service` | `d0IgI0taI document SelVICe` |

## 5. Interpretation

1. **GLM-OCR remains the accuracy reference** on synthetic Latin text. Its 0.00% normalized CER is expected for common words rendered in clean digital fonts, even with light augmentation.
2. **Hierarchical EmbeddingOCR is competitive at word level after normalization**, but case sensitivity and suffix truncation hurt raw accuracy and phrase accuracy.
3. **Character-level segmentation is the weakest path** because it turns glyph ambiguity and segmentation noise into character substitutions, and its per-component embedding calls dominate latency.
4. **Normalization matters:** hierarchical raw word accuracy is 44.0%, but normalized accuracy is 80.0%; most gains come from case handling rather than visual matching.
5. **Synthetic results do not establish handwriting robustness.** The IAM handwriting report remains the harder real-world comparison.

## 6. Reproducibility

```bash
uv run python scripts/test_word_ocr_vs_glm.py --gemma-dim 768 --augment
```

The script writes its raw output to:

```text
docs/word_ocr_vs_glm_report.json
```

The exact JSON used for this report is preserved at:

```text
docs/word_ocr_vs_glm_report_2026-11-08.json
```
