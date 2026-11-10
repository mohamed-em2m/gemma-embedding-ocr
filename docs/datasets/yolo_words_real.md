---
license: cc-by-4.0
task_categories:
  - object-detection
language:
  - en
tags:
  - yolo
  - ocr
  - word-detection
  - text-detection
  - document-analysis
  - roboflow
pretty_name: Word Detection (multi-source YOLO)
size_categories:
  - 1K<n<10K
---

# Word Detection — multi-source YOLO dataset

A single-class object-detection dataset for locating **word bounding boxes** in
scanned documents, screenshots, handwritten notes and photographed pages. Four
public Roboflow datasets were merged into one YOLO-format release with a single
`word` class.

This dataset is the detection stage of an OCR pipeline: it finds words, it does
not read them. Recognition is a separate model.

## At a glance

| | |
| --- | --- |
| Task | Object detection (bounding boxes) |
| Classes | 1 — `word` |
| Images | 7,003 |
| Boxes | 574,576 |
| Annotation format | YOLO (`class cx cy w h`, normalized) |
| Image format | JPEG |
| License | CC BY 4.0 (attribution required) |
| Sources | 4 Roboflow projects |

## Splits

| Split | Images | Boxes | Boxes / image (mean · median · max) |
| --- | ---: | ---: | --- |
| `train` | 6,337 | 520,866 | 82.2 · 74 · 360 |
| `val` | 422 | 38,505 | 91.2 · 76 · 334 |
| `test` | 244 | 15,205 | 62.3 · 47 · 250 |
| **total** | **7,003** | **574,576** | 82.1 · 74 · 360 |

Splits come from the upstream projects and were preserved, so images never move
between `train`, `val` and `test`. 80 duplicate images were dropped during the
merge (all from `math_words`).

## Sources and attribution

Every upstream project is **CC BY 4.0**. Attribution is required if you use this
dataset.

| Source | Roboflow project | Original classes | Images | Boxes |
| --- | --- | --- | ---: | ---: |
| `devanagari_words` | [aaa-8kwq1/devanagari-word-detection](https://universe.roboflow.com/aaa-8kwq1/devanagari-word-detection) | `word` | 297 | 23,662 |
| `word_detection_yolo` | [yogesh-jangir-3utla/word-detection-using-yolo](https://universe.roboflow.com/yogesh-jangir-3utla/word-detection-using-yolo) | `word` | 458 | 86,454 |
| `words_5tolc` | [aabis-siyas-hoisi/words-5tolc](https://universe.roboflow.com/aabis-siyas-hoisi/words-5tolc) | `Texto` | 1,031 | 14,075 |
| `math_words` | [kyiv-national-university/math-words](https://universe.roboflow.com/kyiv-national-university/math-words) | `Text`, `W`, `math`, `text` | 5,217 | 450,385 |

`math_words` is distributed through the mirror
[`el-emam/math-words-oahhj`](https://universe.roboflow.com/el-emam/math-words-oahhj),
because the original project has no generated export version.

### What the merge changed

- **All classes collapsed to `word` (id 0).** `Texto`, `Text`, `W`, `math` and
  `text` were treated as the same concept. No boxes were dropped for class
  reasons.
- **Split names normalized**: `valid` → `val`, `test` kept.
- **Duplicate images removed** (80, all from `math_words`), identified across
  sources.
- **Coordinates validated**: every label line has exactly 5 fields, class id `0`,
  and coordinates inside `[0, 1]` with positive width and height.
- **Filenames namespaced** as `<source>__<original-name>.jpg` so the origin of
  any image is recoverable.

## Structure

```
.
├── data.yaml            # Ultralytics dataset config (nc: 1, names: [word])
├── classes.json         # class list and id mapping
├── sources.json         # provenance and per-source counts
├── stats.json           # full statistics, machine readable
├── manifest.json        # merge manifest produced by the builder
├── README.md
├── images/
│   ├── train/  (6,337)
│   ├── val/    (422)
│   └── test/   (244)
└── labels/
    ├── train/  (6,337 .txt)
    ├── val/    (422 .txt)
    └── test/   (244 .txt)
```

Each label file matches an image by basename and contains one line per word:

```
class_id  center_x  center_y  width  height     # all normalized to [0, 1]
```

## Geometry

| Property | Min | Median | Max |
| --- | ---: | ---: | ---: |
| Image width (px) | 640 | 640 | 4,032 |
| Image height (px) | 328 | 640 | 4,032 |
| Box width (normalized) | — | 0.060 | 0.987 |
| Box height (normalized) | — | 0.037 | 0.914 |

The median word box is **0.060 × 0.037** of the image, i.e. roughly **39 × 24
pixels on a 640 px image**. This is a small-object detection problem: training
at 640 px will merge or lose many boxes. **960–1280 px is recommended.**

Median box aspect ratio is 1.54 (p95 = 5.5), so words are typically wider than
tall, but long compound words reach ratios above 5.

## Usage

### Ultralytics YOLO

```python
from ultralytics import YOLO

model = YOLO("yolo26s.pt")
model.train(
    data="data.yaml",
    imgsz=960,
    batch=16,
    epochs=40,
    single_cls=True,
    max_det=512,      # up to 360 boxes per image
)
metrics = model.val(split="test")
```

`max_det` matters here: some images contain more than 300 words, so the
Ultralytics default of 300 truncates predictions and understates mAP.

### Loading the images and labels

```python
from pathlib import Path

from PIL import Image

root = Path(".")  # dataset root
for label in (root / "labels" / "val").glob("*.txt"):
    image = Image.open(root / "images" / "val" / f"{label.stem}.jpg")
    for line in label.read_text().splitlines():
        cls, cx, cy, w, h = map(float, line.split())
        box = [
            (cx - w / 2) * image.width,
            (cy - h / 2) * image.height,
            (cx + w / 2) * image.width,
            (cy + h / 2) * image.height,
        ]
```

## Limitations

- **One class only.** No distinction between printed and handwritten text, no
  script labels, no reading order, no transcription.
- **Script imbalance.** `devanagari_words` contains Devanagari text, while the
  other three are mostly Latin and mathematical notation. Boxes are correct, but
  the dataset is not script-balanced and should not be described as Latin-only.
- **Domain imbalance.** `math_words` contributes 78% of all boxes; mathematical
  documents dominate the label distribution. Plain prose pages are
  under-represented.
- **Small boxes.** Median word height is ~3.7% of the image. Downstream users
  should train at high resolution and validate on their own domain.
- **No negative images.** Every image contains at least one word box, so the
  model sees no explicit background-only examples.
- **Inherited label noise.** Box quality is that of the upstream projects; no
  manual re-annotation was performed.

## License

Released under **CC BY 4.0**, inherited from the four upstream datasets. You
must credit:

- aaa-8kwq1 — *Devanagari word detection*
- yogesh-jangir-3utla — *Word Detection using yolo*
- aabis-siyas-hoisi — *Words*
- Kyiv National University — *math-words*

## Citation

```bibtex
@misc{gemma_ocr_words_real,
  title  = {Word Detection: multi-source YOLO dataset},
  author = {Emam},
  year   = {2026},
  note   = {Merged from four CC BY 4.0 Roboflow datasets},
  url    = {https://huggingface.co/datasets/elemam/gemma-ocr-words-real}
}
```

## Provenance

| Field | Value |
| --- | --- |
| Version | 1.0.0 |
| Generated | 2026-11-10 |
| Merge seed | 42 |
| Builder | `scripts/build_real_word_dataset.py` |
| Statistics | `scripts/dataset_stats.py` |
