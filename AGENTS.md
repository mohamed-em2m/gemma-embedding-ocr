
# EmbeddingOCR: Project Plan (v2)

> Single source of truth for planning and building EmbeddingOCR with OpenCode.
> Keep this file in the repo root next to `AGENTS.md`.
>
> **v2 changes:** EmbeddingGemma 2 (multimodal) is now the primary embedder; scope is
> **Latin only** at first; Phase 0 is a small **a-z / A-Z embedding + FAISS search spike**,
> and everything after it is decided from its results.

## 0. How to use this file with OpenCode

1. Create the repo folder, run `git init`, then start `opencode` in it.
2. Run `/init` so OpenCode picks up `AGENTS.md` (Section 11).
3. Stay in **Plan mode** (Tab toggles Plan/Build). Tell it:
   `Read PLAN.md and AGENTS.md. Confirm the task list for Phase 0 and wait for my approval.`
4. After approval, switch to **Build mode** and execute **one phase at a time**.
5. At the end of every phase the agent follows the Git workflow (Section 10), then stops and waits for confirmation.
6. **Do not plan past Phase 0 in detail.** Phases 1-7 are provisional and get revised after the Phase 0 decision gate.

---

## 1. Project definition

**EmbeddingOCR**: a modular OCR system that replaces conventional character recognition with visual embedding matching, while preserving the standard OCR detection and processing pipeline.

We are **not** building a new OCR from scratch. We keep a traditional OCR pipeline and replace only the Recognition Engine.

```
Input Image
  -> 1. Preprocessing (resize, denoise, binarize, deskew)      KEEP
  -> 2. Text detection (DBNet / CRAFT / PaddleOCR det)         KEEP
  -> 3. Region cropping (word / line crops)                    KEEP
  -> 4. Character recognition                                  REPLACE  <- research contribution
        segmentation OR alignment
        -> embedding extraction (EmbeddingGemma 2)
        -> FAISS prototype search (cosine similarity)
        -> character selection
  -> 5. Text reconstruction (ordering, spaces, lines)          KEEP
  -> 6. Confidence scoring                                     MODIFY
  -> Recognized text
```

| OCR component         | Decision |
| --------------------- | -------- |
| Image preprocessing   | Keep     |
| Text detection        | Keep     |
| Region cropping       | Keep     |
| Character recognition | Replace  |
| Text reconstruction   | Keep     |
| Reading order         | Keep     |
| Confidence scoring    | Modify   |

## 2. Key decisions

| Topic | Decision |
| ----- | -------- |
| Scope | **Latin only** (a-z, A-Z first; digits and punctuation later). Arabic and other connected scripts are out of scope for v1 and listed as future work. |
| Embedder | **EmbeddingGemma 2** (`google/embeddinggemma-2`, verify the exact id on Hugging Face). Multimodal: text, image, video, audio mapped into one 768-d space. ~740M parameters, built on the Gemma 4 decoder, Apache 2.0, Matryoshka dims 768/512/256/128, fine-tunable. Behind an `Embedder` protocol so it can be swapped. |
| Recognition interface | `RecognitionEngine` with `recognize(image) -> RecognitionResult(text, confidence)`. Swapping PaddleOCR and EmbeddingOCR needs config changes only. |
| MVP strategy | Character **Segmentation** first. Study **Alignment** only if segmentation fails on touching glyphs. |
| Vector search | **FAISS** `IndexFlatIP` on L2-normalized vectors (exact; inner product = cosine). Behind a `VectorIndex` protocol. |
| Aggregation | Max similarity **per character class** over top-k hits. |
| Fair comparison | Same preprocessing and same cached detection crops for both systems. Only the recognizer differs. |
| Hardware | **CPU and GPU both supported.** `device: auto`; the smoke test must pass on CPU. |
| Tooling | Python >= 3.11, **uv** only, `pyproject.toml`, src layout, ruff, mypy, pytest, pre-commit, GitHub Actions. |
| Datasets / fonts | **Deferred.** Decided after Phase 0. Phase 0 uses only rendered a-z / A-Z. |

> Note on model facts: the description of EmbeddingGemma 2 above comes from Google's
> model page (checked at planning time). The exact Hugging Face loading code
> (which `transformers` class, minimum `transformers` version, image processor
> behavior) is **unverified**. Phase 0 must confirm it from the model card.

## 3. Metrics

- Character Error Rate (CER)
- Word Error Rate (WER)
- Recognition accuracy
- End-to-end OCR accuracy
- Latency per text region
- GPU memory usage

For Phase 0 only: top-1 / top-5 retrieval accuracy over the 52 classes, latency per embedding, memory.

---

## 4. Repository layout

Phase 0 only needs the minimal part marked `(P0)`. The rest appears in Phase 1+.

```
embedding-ocr/
├── pyproject.toml                (P0)
├── uv.lock                       (P0, committed)
├── .python-version               (P0)
├── README.md
├── PLAN.md                       (P0)
├── AGENTS.md                     (P0)
├── opencode.json                 (P0)
├── Makefile                      # or justfile
├── .gitignore                    (P0)
├── .pre-commit-config.yaml
├── .github/workflows/ci.yml
├── configs/
│   ├── spike_az.yaml             (P0)
│   ├── default.yaml
│   ├── baseline_paddle.yaml
│   ├── embedding_seg.yaml
│   └── embedding_align.yaml
├── docs/
│   ├── feasibility.md            (P0 output)
│   ├── decisions.md              (P0)
│   └── benchmark_report.md
├── scripts/
│   └── phase0_az_spike.py        (P0)
├── tests/
└── src/embedding_ocr/
    ├── __init__.py  py.typed  cli.py  config.py
    ├── pipeline/       # preprocess, detect, crop, reconstruct, pipeline
    ├── recognition/
    │   ├── base.py     # RecognitionEngine + RecognitionResult
    │   ├── paddle.py   # baseline adapter
    │   └── embedding/
    │       ├── segmenter.py   # projection + connected components (MVP)
    │       ├── aligner.py     # sequence alignment (only if needed)
    │       ├── embedder.py    # Embedder protocol + EmbeddingGemma2 backend   (P0)
    │       ├── index.py       # VectorIndex protocol, FaissIndex, NumpyIndex   (P0)
    │       ├── prototypes.py  # build/load prototype DB + metadata
    │       ├── matcher.py     # per-class aggregation + confidence
    │       └── engine.py      # EmbeddingRecognizer(RecognitionEngine)
    ├── data/           # glyph rendering from fonts (Pillow)                  (P0)
    ├── eval/           # metrics, benchmark, profiling
    └── utils/          # seed, logging, io, device                            (P0)
```

## 5. pyproject.toml requirements

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "embedding-ocr"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
  "numpy", "pillow", "opencv-python-headless",
  "pydantic>=2", "pyyaml", "typer", "rich",
  "torch",
  "transformers",            # pin a version that supports EmbeddingGemma 2 (verify on model card)
  "faiss-cpu>=1.8",
  "jiwer", "rapidfuzz",
]

[project.optional-dependencies]
baseline = ["paddleocr", "paddlepaddle"]
faiss-gpu = ["faiss-gpu-cu12"]   # verify package name and CUDA match before use
tracking = []

[project.scripts]
embedding-ocr = "embedding_ocr.cli:app"

[dependency-groups]
dev = ["pytest", "pytest-cov", "ruff", "mypy", "pre-commit"]
notebooks = ["jupyter", "matplotlib"]

[tool.uv.sources]
# configure torch CPU/CUDA index here

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM", "NPY", "PT"]

[tool.pytest.ini_options]
testpaths = ["tests"]

[tool.mypy]
strict = true
```

Commands to document in the README:

```bash
uv sync --group dev
uv run pytest
uv run ruff check . && uv run ruff format .
uv run mypy src
uv run python scripts/phase0_az_spike.py --config configs/spike_az.yaml
```

---

## 6. Core interfaces

```python
@dataclass(frozen=True)
class RecognitionResult:
    text: str
    confidence: float

class RecognitionEngine(Protocol):
    def recognize(self, image: np.ndarray) -> RecognitionResult:
        """Input: text-region image (H, W, C). Output: text and confidence."""

class Embedder(Protocol):
    name: str
    dim: int                                  # active dimension (e.g. 768 / 256)
    def embed_images(self, images: list[np.ndarray]) -> np.ndarray: ...
    def embed_texts(self, texts: list[str]) -> np.ndarray: ...
    # both return float32 (n, dim), already truncated to `dim` and L2-normalized

class VectorIndex(Protocol):
    def build(self, vectors: np.ndarray, labels: np.ndarray) -> None: ...
    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (similarities, prototype_ids), each (n_queries, k)."""
    def save(self, path: Path) -> None: ...
    @classmethod
    def load(cls, path: Path) -> "VectorIndex": ...
```

Rules:
- **Matryoshka truncation:** when `dim < 768`, slice the first `dim` values and **re-normalize** before indexing or querying. Otherwise cosine scores are wrong.
- `faiss.normalize_L2` on prototypes and queries; `IndexFlatIP` for exact cosine.
- Search `k = 20-50`, then aggregate to max per class in `matcher.py`; one batched `search` per word crop.
- Save with `faiss.write_index` plus `meta.json` (embedder name/version, **active dim**, modality, font list, augmentation config, git hash). On load, **raise** if embedder name or dim does not match the config.
- Set `faiss.omp_set_num_threads` from config to avoid oversubscription with torch.

Config:

```yaml
embedder:
  name: google/embeddinggemma-2     # verify exact id
  dim: 768                           # 768 | 512 | 256 | 128
  device: auto                       # auto | cpu | cuda
  dtype: float32
  batch_size: 32
index:
  backend: faiss_flat_ip            # later: faiss_hnsw, faiss_ivf
  top_k: 30
  aggregation: max_per_class
  path: artifacts/prototypes/v1/
```

---

## 7. Phases

Each phase ends with: lint + types + tests green, a commit (Section 10), a short summary, open questions, and **a stop for approval**.

### Phase 0: A-Z embedding + FAISS search spike (FIRST STEP)

**Goal:** the smallest possible end-to-end test. Embed the 52 Latin letters (a-z, A-Z), put them in FAISS, run vector search, look at the results, and then decide what to do next. No OCR pipeline, no detection, no segmentation yet.

**Minimal setup:** `uv init`, a minimal `pyproject.toml` (torch, transformers, faiss-cpu, pillow, numpy, pyyaml), `.gitignore`, `scripts/phase0_az_spike.py`, `configs/spike_az.yaml`. Full project structure comes in Phase 1.

**Experiments** (all on the 52 classes `a-z` + `A-Z`):

| ID | Experiment | What it tells us |
| -- | ---------- | ---------------- |
| E1 | **Text index.** Embed the 52 letters as text strings, L2-normalize, build `IndexFlatIP`. Query each letter with itself and inspect the full 52x52 similarity matrix. | Sanity check of the pipeline; which letters are close in text space (`a`/`A`, `I`/`l`). |
| E2 | **Image -> text (cross-modal).** Render each letter as an image (several fonts), embed the images, search the **text** index. Report top-1 / top-5. | Whether the shared multimodal space can match a glyph picture to its letter with **no font prototypes at all**. |
| E3 | **Image -> image across fonts.** Build the index from glyph images in fonts set A, query with fonts set B (unseen). Report top-1 / top-5. | The realistic prototype-matching setup, including generalization to new fonts. |
| E4 | **Matryoshka sweep.** Repeat E2 and E3 at dims 768, 512, 256, 128 (re-normalize after truncation). | Accuracy vs. index size and speed trade-off. |
| E5 | **Cost.** Latency per embedding (batch 1 and batch 32), memory, on CPU and GPU if available. | Whether per-character embedding is affordable in the OCR loop. |

**Rendering details for glyph images (decide and record in `docs/decisions.md`):**
- Render each letter on a white square canvas, centered, with padding; try a couple of canvas sizes and note how the model's image preprocessor resizes them.
- Use at least 6 open-licensed fonts (e.g. DejaVu Sans, Liberation Sans/Serif, Noto Sans/Serif, Roboto, Open Sans). Split into fonts A (indexed) and fonts B (held out).
- Optionally add light blur/noise/rotation as a second pass, only after the clean results are recorded.

**Analysis to include in `docs/feasibility.md`:**
- Tables: top-1/top-5 per experiment and per dimension; latency and memory.
- 52x52 confusion matrix for the best setting, plus the top confused pairs.
- **Case-pair warning:** `c/C, o/O, s/S, v/V, w/W, x/X, z/Z, u/U, p/P` have the same shape and differ mostly by **size**. If each glyph is cropped and resized to a square, that size cue is lost and these pairs may be inherently ambiguous. Report these separately and propose a fix if needed (e.g. keep original aspect ratio and baseline position, add a relative-height feature, or resolve case from word context in Phase 4).
- Other look-alikes: `I/l`, and `i/j` dots, `g/q`, `b/d`, `n/h`, `u/v`.
- Verified facts about EmbeddingGemma 2: exact model id, loading code, minimum `transformers` version, image preprocessing behavior, license, any prompt/prefix required for text vs. image inputs.

**Decision gate (proposed thresholds; adjust after seeing the data):**

| Result on unseen fonts, clean glyphs | Next step |
| ------------------------------------ | --------- |
| E3 top-1 >= ~95% | Zero-shot works. Proceed to Phase 1-4 with image prototypes. |
| E2 top-1 high (>= ~90%) | Text prototypes may suffice, so fonts may not be needed for the prototype DB. Compare with E3 in Phase 4. |
| 80-95% | Add more fonts/augmentations; consider light fine-tuning. |
| < 80%, or case pairs badly confused | Fine-tune EmbeddingGemma 2 on rendered glyphs (contrastive loss), and/or compare SigLIP / DINOv2 as alternatives. |
| Latency too high on CPU | Use smaller dim, smaller image size, batching, or a lighter embedder. |

**Acceptance:** `docs/feasibility.md` exists with the tables above and a written recommendation; the spike runs from one command on CPU; **stop and wait for the user to choose the next step.**

### Phase 1: Project setup (provisional)

Tasks: full src layout, pyproject (Section 5), ruff/mypy/pytest config, pre-commit, `.gitattributes`, `Makefile`, CI workflow, README skeleton. Move reusable spike code (rendering, embedder, index) into `src/embedding_ocr/`.

Acceptance: `uv sync --group dev`, `uv run pytest`, ruff, mypy all pass; CI green on first push.

### Phase 2: Core abstractions (provisional)

Tasks: `RecognitionEngine`, `RecognitionResult`, typed config, `Embedder` protocol with `EmbeddingGemma2` backend, `VectorIndex` with `FaissIndex` and `NumpyIndex`, pipeline wiring where the recognizer is chosen **purely by config**, detection cache to disk.

Acceptance: `FaissIndex` and `NumpyIndex` return identical top-k on random data; pipeline runs with a dummy recognizer; cached detections reload byte-identically.

### Phase 3: Baseline (provisional)

Tasks: `PaddleRecognizer` adapter (or a better baseline chosen in Phase 0), run the full pipeline on a small dataset, record CER/WER as reference numbers.

Acceptance: baseline numbers reproducible from one CLI command.

### Phase 4: EmbeddingRecognizer MVP with segmentation (provisional)

Tasks:
1. Prototype DB from the Phase 0 winner (text prototypes, image prototypes, or both); extend charset to digits and punctuation here.
2. Segmenter: vertical projection plus connected components, with logging of touching or merged characters.
3. Matcher: per-class max aggregation; confidence from the top-1 / top-2 margin.
4. Spaces and ordering via `pipeline/reconstruct.py`; case resolution using size/baseline cues if Phase 0 showed case-pair confusion.
5. Tests: segmentation on synthetic words, matcher on known glyphs, valid `RecognitionResult`, index metadata mismatch raises.

Acceptance: engine swaps in via config only; smoke test under 60 s on CPU.

### Phase 5: Alignment (only if Phase 4 fails on touching glyphs)

Sliding-window features over the crop matched to prototypes, decoded with CTC-style or Viterbi alignment. Compare with segmentation on the same benchmark.

### Phase 6: Benchmark (fair comparison)

- Same cached detections for both systems.
- All metrics from Section 3, including a held-out-font split and real scans.
- Error analysis: touching glyphs, `O/0`, `l/1/I`, case pairs.
- Datasets and prototype/unseen font splits are **chosen here**, informed by Phase 0.
- Only if prototypes exceed ~500k: compare `IndexFlatIP`, `IndexHNSWFlat`, `IndexIVFFlat`.

### Phase 7: Polish

README (overview, install, quickstart, config guide, results), CI with `uv sync --locked`, limitations section (including "Latin only"), tag `v0.1.0`.

---

## 8. Test strategy

- Unit: shapes, determinism under fixed seed, Faiss vs Numpy agreement, Matryoshka truncation + re-normalization, matcher behavior, segmenter on synthetic words.
- Sanity: each text letter retrieves itself at rank 1; prototype round-trip (render, embed, search returns the same class).
- Smoke: tiny end-to-end run under 60 s on CPU.
- Regression: baseline CER/WER on the fixed mini-dataset must not drift.

## 9. Working rules for the agent

- Plan first; wait for approval before each phase. Phase 0 is the only phase planned in detail.
- Do not guess model names, APIs, or versions. Verify them from the model card or docs, or mark as **unverified**.
- Never silently invent details. Record assumptions in `docs/decisions.md`.
- Run `uv run ruff check . && uv run ruff format --check . && uv run mypy src && uv run pytest` before declaring a step done (once Phase 1 tooling exists; in Phase 0 run what is configured).
- Be honest about anything untested.

---

## 10. Git workflow: commit and push

### 10.1 One-time setup

```bash
git init -b main
git config user.name  "Your Name"
git config user.email "you@example.com"

# create an empty repo on GitHub first, then:
git remote add origin git@github.com:<user>/embedding-ocr.git
# or HTTPS: https://github.com/<user>/embedding-ocr.git

git add .
git commit -m "chore: add PLAN.md, AGENTS.md and minimal project skeleton"
git push -u origin main
```

With GitHub CLI: `gh repo create embedding-ocr --private --source=. --remote=origin --push`.

### 10.2 Branching

- `main`: always green, protected if you work in a team.
- One branch per phase: `phase-0-spike-az`, `phase-1-setup`, `phase-2-core`, `phase-3-baseline`, `phase-4-embedding-mvp`, `phase-5-alignment`, `phase-6-benchmark`, `phase-7-polish`.
- Optional smaller branches: `feat/<name>`, `fix/<name>`, `docs/<name>`.

```bash
git switch main && git pull --ff-only
git switch -c phase-0-spike-az
```

### 10.3 Commit message format (Conventional Commits)

```
<type>(<scope>): <short imperative summary>

<optional body: what and why>
```

Types: `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `build`, `ci`, `chore`.
Scopes: `spike`, `pipeline`, `recognition`, `embedder`, `index`, `segmenter`, `matcher`, `eval`, `config`, `data`, `deps`.

Examples:

```
feat(spike): render a-z and A-Z glyphs for embedding experiments
feat(embedder): add EmbeddingGemma 2 backend with Matryoshka truncation
feat(index): add FaissIndex with L2-normalized IndexFlatIP
test(index): assert FaissIndex and NumpyIndex return identical top-k
docs(feasibility): record E1-E5 results and decision-gate recommendation
build(deps): add faiss-cpu and update uv.lock
```

Rules:
- One logical change per commit.
- Commit `uv.lock` together with the `pyproject.toml` change that caused it.
- Never commit datasets, checkpoints, model weights, FAISS indexes, `.env`, or API keys.
- Font files: commit only if the license allows; otherwise list them in the README and download via a script.

### 10.4 Pre-commit checklist (run before every commit)

```bash
uv run ruff check . --fix
uv run ruff format .
uv run mypy src
uv run pytest
git status
git diff --staged
```

Install hooks once: `uv run pre-commit install`.

### 10.5 Commit and push a phase

```bash
git add -p                       # review hunks; avoid blind `git add .`
git commit -m "docs(feasibility): record Phase 0 results"
git push -u origin phase-0-spike-az      # first push of the branch
git push                                 # later pushes
```

Open a PR into `main` (`gh pr create --fill` or via GitHub), wait for CI green, then squash-merge or merge. Afterwards:

```bash
git switch main && git pull --ff-only
git tag -a phase-0-done -m "Phase 0: A-Z embedding + FAISS spike"
git push origin phase-0-done
git branch -d phase-0-spike-az
```

Release tag at the end: `git tag -a v0.1.0 -m "First release" && git push origin v0.1.0`.

### 10.6 Fixing mistakes safely

```bash
git restore --staged <file>        # unstage
git restore <file>                 # discard working changes (destructive)
git commit --amend                 # fix the last commit (only if NOT pushed)
git revert <sha>                   # undo a pushed commit safely
```

Never force-push `main`. On your own feature branch use `git push --force-with-lease` only if you must.

### 10.7 Rules for the OpenCode agent

- The agent **may** run `git status`, `git diff`, `git add`, and (after checks pass) propose `git commit`.
- The agent **must ask me** before `git push`, `git tag`, `git reset --hard`, `git rebase`, `git clean`, or any force operation.
- The agent must show the staged diff summary and the proposed commit message before committing.
- The agent must never commit secrets, data, model weights, checkpoints, or indexes.

`opencode.json`:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "permission": {
    "edit": "ask",
    "bash": {
      "uv run *": "allow",
      "git status*": "allow",
      "git diff*": "allow",
      "git add*": "allow",
      "git commit*": "ask",
      "git push*": "ask",
      "git tag*": "ask",
      "git reset*": "ask",
      "git rebase*": "ask",
      "git clean*": "ask",
      "*": "ask"
    }
  }
}
```

Check the current OpenCode docs for exact permission syntax; it may change between versions.

### 10.8 `.gitignore` essentials

```gitignore
# Python / uv
.venv/
__pycache__/
*.pyc
.pytest_cache/
.mypy_cache/
.ruff_cache/
.coverage
htmlcov/
dist/

# Project artifacts (never commit)
data/
artifacts/
checkpoints/
*.faiss
*.index
.cache/
runs/
wandb/
models/
*.safetensors

# Secrets / editor
.env
.env.*
.vscode/
.idea/
.DS_Store
```

### 10.9 CI (`.github/workflows/ci.yml`)

```yaml
name: ci
on:
  push:
    branches: [main]
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
      - run: uv sync --locked --group dev
      - run: uv run ruff check .
      - run: uv run ruff format --check .
      - run: uv run mypy src
      - run: uv run pytest --cov
```

CI must **not** download the large embedder. Mark tests that need the real model with `@pytest.mark.model` and skip them in CI, or use a tiny fake `Embedder`. Pin action versions to current releases when creating the file.

---

## 11. `AGENTS.md` (put in repo root)

```markdown
# EmbeddingOCR

Modular OCR system that replaces conventional character recognition with visual
embedding matching, while keeping the standard OCR detection/processing pipeline.
Scope: Latin script only for now.

## Stack and commands
- Python >=3.11, managed with **uv** only (never pip or poetry).
- `uv sync --group dev` install | `uv run pytest` test
- `uv run ruff check . && uv run ruff format .` lint/format
- `uv run mypy src` types
- `uv run embedding-ocr <command>` CLI (Typer)

## Architecture rules (non-negotiable)
- Only the Recognition stage is replaced. Preprocessing, detection, cropping,
  reconstruction, and reading order stay as in the baseline pipeline.
- All recognizers implement `RecognitionEngine` (recognition/base.py).
  Swapping PaddleOCR <-> EmbeddingOCR needs zero changes outside config.
- Primary embedder: EmbeddingGemma 2 (multimodal, 768-d, Matryoshka 768/512/256/128),
  behind the `Embedder` protocol. Never hardcode the model. Verify model id,
  loading code, and required library versions from the model card; mark anything
  unverified.
- If embedding dim < 768, truncate then re-normalize.
- Prototype search uses a FAISS index behind `VectorIndex`. Embeddings are
  L2-normalized (prototypes and queries). The saved index records embedder name
  and active dim; loading a mismatched index must raise.
- Aggregate hits per character class (max similarity); batch one search per word.
- Detection results are cached to disk so baseline and EmbeddingOCR are
  evaluated on identical crops.

## Code style
- Full type hints, `from __future__ import annotations`, pathlib, logging/rich (no print).
- No hardcoded hyperparameters: typed config (pydantic) + YAML in `configs/`.
- Small single-purpose modules. Docstrings state array shapes.
- Seeds via `utils.seed.set_seed()`. Every run saves config + git hash.
- Device-agnostic: `device: auto`, CPU must always work.
- Never commit data, model weights, checkpoints, or FAISS indexes.

## Workflow
- Follow PLAN.md phase by phase; stop after each phase for approval.
- Phase 0 (a-z / A-Z spike) comes first; do not build later phases until the
  user picks the next step from its results.
- Before declaring a step done: ruff, mypy, pytest must pass (where configured).
- Git: Conventional Commits, one logical change per commit, one branch per phase.
  Ask before push, tag, reset, rebase, or clean.
- Be honest about untested or uncertain parts. Never invent model or API details;
  record assumptions in `docs/decisions.md`.
```

---

## 12. Decisions made and questions still open

**Decided**
1. Embedder: EmbeddingGemma 2 is the primary candidate (multimodal). Alternatives are only fallbacks if Phase 0 shows weak glyph discrimination.
2. Script scope: **Latin only** first. Arabic and other connected scripts are future work.
3. Datasets and fonts for the full benchmark: **deferred** until after the Phase 0 results. Phase 0 uses rendered a-z / A-Z only.
4. Hardware: CPU and GPU both available; code stays device-agnostic.
