# Fine-tuning run: bug log

Issues found while running `scripts/kaggle_train_embeddinggemma.sh` end to end.
Kept here so the fixes and the remaining risks are traceable across runs.

## Fixed

| # | Symptom | Cause | Fix |
| - | ------- | ----- | --- |
| 1 | `torchrun` exited before training on a single-GPU host | launcher hardcoded `--nproc_per_node=2` | `NPROC=auto` reads `torch.cuda.device_count()`; DDP gathering is only enabled when >1 process |
| 2 | The bf16 branch never changed anything | the check only printed a message while `--precision float32` was hardcoded | `PRECISION=auto` resolves the dtype and the same value is passed to training and retrieval eval; compute capability >= 8.0 is required |
| 3 | `Failed to initialize NCCL communicator: ... CUDA calloc async` right after batch probing | the probe steps a real AdamW, leaving ~6 GB of optimizer state and cached blocks on a 15 GB T4 | clear the optimizer state and empty the CUDA cache before the trainer initializes NCCL |
| 4 | `ValueError: The prompts provided to the trainer are a nested dictionary ... not a DatasetDict` | the per-dataset prompt map was only given to the collator, so the trainer never injected `dataset_name` | pass the same `prompts` mapping to `SentenceTransformerTrainingArguments` |
| 5 | One of four text corpora was skipped | `facebook/cc100` no longer exists and dataset scripts are unsupported in `datasets>=5` | replace it with `HuggingFaceFW/fineweb` (sample-10BT, parquet) |
| 6 | `IsADirectoryError` when `REAL_CROPS` pointed at `data/real_crops` | `load_real_crops` only accepted a JSONL file | accept a directory and resolve `real.jsonl` (or the first `*.jsonl`) |
| 7 | Full re-render on every Kaggle restart | the build step ran unconditionally | reuse `summary.json` unless `REBUILD=1` |
| 8 | Row counts, epochs and batch sizes were hardcoded | no env overrides | expose `TEXT_IMAGE_ROWS`, `IMAGE_IMAGE_ROWS`, `REAL_ROWS`, `EPOCHS`, `MAX_STEPS`, `SAVE_STEPS`, `BATCH_SIZE`, `EVAL_BATCH_SIZE`, `MINI_BATCH_SIZE`, `GRAD_ACCUM`, `DATALOADER_WORKERS`, `MODEL_ID`, `RESUME` |

## Open risks

- **Single-T4 throughput.** Full fine-tuning in fp32 probes to batch size 1
  (~7-9 s/step with gradient checkpointing). 60k text-image + 20k image-image
  rows is multi-day on one T4; the 2xT4 Kaggle target is assumed for full runs.
- **No in-batch negatives at batch 1.** `CachedMultipleNegativesRankingLoss`
  needs at least two examples per contrastive batch. Single-GPU runs without
  `--gather-across-devices` therefore train on a degenerate loss; use the
  2-GPU target or raise the effective batch.
- **`warmup_ratio` deprecation.** Transformers v5 wants a float `warmup_steps`;
  the warning is harmless now but will break on the next major upgrade.
- **`UniqueTextBatchSampler.__len__`** ignores duplicate `text_id` skips, so the
  reported steps per epoch can exceed the batches actually yielded.
- **Dataloader workers.** `DATALOADER_WORKERS=2` with a `DatasetDict` of
  on-disk images is fine, but workers are forked after the model is loaded;
  check RSS when raising it on Kaggle.
