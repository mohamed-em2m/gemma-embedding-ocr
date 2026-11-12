#!/usr/bin/env python
"""Evaluate a trained CTC line recognizer on a prepared manifest.

Reports character error rate (CER) and word error rate (WER) per split, per
source and per script, with optional prefix beam search and lexicon rescoring,
plus a per-example dump for error analysis.

Examples::

    # Greedy CTC on the validation split
    uv run --no-sync python scripts/eval_ctc_ocr.py \
        --checkpoint artifacts/ctc_ocr/best.pt

    # Beam search with a lexicon built from the training transcripts
    uv run --no-sync python scripts/eval_ctc_ocr.py \
        --checkpoint artifacts/ctc_ocr/best.pt --beam-width 10 --use-lexicon

    # Compare decoders side by side on the same crops
    uv run --no-sync python scripts/eval_ctc_ocr.py \
        --checkpoint artifacts/ctc_ocr/best.pt --compare-decoders
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

import jiwer

from embedding_ocr.recognition.ctc.engine import CTCRecognizer, DecodeSettings
from embedding_ocr.recognition.ctc.lexicon import LexiconScorer, WordLexicon
from embedding_ocr.recognition.ctc.normalize import normalize_for_cer
from embedding_ocr.training.ctc_dataset import load_gray, to_gray
from embedding_ocr.training.line_dataset import LineRecord, read_manifest

_REPO_ROOT = Path(__file__).resolve().parents[1]
LOGGER = logging.getLogger("eval_ctc_ocr")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=_REPO_ROOT / "artifacts/lines/manifest.jsonl",
    )
    parser.add_argument(
        "--split",
        nargs="+",
        default=["val"],
        help="Splits to score (default: val). Use 'train' for a fit check.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--beam-width", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument(
        "--lexicon",
        type=Path,
        default=None,
        help="Word list, one transcript per line (default: build from the manifest).",
    )
    parser.add_argument("--lexicon-min-count", type=int, default=2)
    parser.add_argument("--use-lexicon", action="store_true")
    parser.add_argument("--lexicon-weight", type=float, default=1.0)
    parser.add_argument(
        "--compare-decoders",
        action="store_true",
        help="Score greedy, beam, and beam+lexicon on the same crops.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--examples", type=int, default=10)
    parser.add_argument("--case-insensitive", action="store_true")
    return parser.parse_args()


def _load_lexicon(
    args: argparse.Namespace, records: Sequence[LineRecord]
) -> WordLexicon:
    if args.lexicon is not None:
        return WordLexicon.from_file(args.lexicon, min_count=args.lexicon_min_count)
    train_texts = [record.text_norm for record in records if record.split == "train"]
    if not train_texts:
        message = "no train rows in the manifest; pass --lexicon explicitly"
        raise ValueError(message)
    return WordLexicon.from_texts(train_texts, min_count=args.lexicon_min_count)


def _score(
    references: Sequence[str], hypotheses: Sequence[str], *, case_sensitive: bool
) -> dict[str, float]:
    """Return CER and WER over a set of pairs."""
    if not references:
        return {"cer": 0.0, "wer": 0.0, "samples": 0, "characters": 0}
    refs = [
        normalize_for_cer(text, case_sensitive=case_sensitive) for text in references
    ]
    hyps = [
        normalize_for_cer(text, case_sensitive=case_sensitive) for text in hypotheses
    ]
    return {
        "cer": float(jiwer.cer(refs, hyps)),
        "wer": float(jiwer.wer(refs, hyps)),
        "samples": len(refs),
        "characters": sum(len(text) for text in refs),
    }


def _predict(
    recognizer: CTCRecognizer,
    records: Sequence[LineRecord],
    *,
    batch_size: int,
) -> list[str]:
    predictions: list[str] = []
    for start in range(0, len(records), batch_size):
        chunk = records[start : start + batch_size]
        images = [to_gray(load_gray(record.image)) for record in chunk]
        predictions.extend(result.text for result in recognizer.recognize_batch(images))
    return predictions


def _breakdown(
    records: Sequence[LineRecord],
    predictions: Sequence[str],
    *,
    case_sensitive: bool,
) -> dict[str, dict[str, float]]:
    """Score overall and grouped by split, source and script."""
    groups: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        groups["all"].append(index)
        groups[f"split:{record.split}"].append(index)
        groups[f"source:{record.source}"].append(index)
        groups[f"script:{record.script}"].append(index)

    report: dict[str, dict[str, float]] = {}
    for name, indices in sorted(groups.items()):
        report[name] = _score(
            [records[index].text_norm for index in indices],
            [predictions[index] for index in indices],
            case_sensitive=case_sensitive,
        )
    return report


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()

    records = read_manifest(args.manifest)
    selected = [record for record in records if record.split in set(args.split)]
    if not selected:
        message = f"no records for splits {args.split} in {args.manifest}"
        raise ValueError(message)
    if args.max_samples:
        selected = selected[: args.max_samples]

    lexicon = None
    if args.use_lexicon or args.compare_decoders:
        lexicon = _load_lexicon(args, records)
        LOGGER.info("lexicon: %d words", len(lexicon))

    configurations: list[tuple[str, DecodeSettings]] = []
    if args.compare_decoders:
        configurations = [
            ("greedy", DecodeSettings(beam_width=1)),
            (
                "beam",
                DecodeSettings(beam_width=max(2, args.beam_width), top_k=args.top_k),
            ),
            (
                "beam+lexicon",
                DecodeSettings(
                    beam_width=max(2, args.beam_width),
                    top_k=args.top_k,
                    lexicon_weight=args.lexicon_weight,
                ),
            ),
        ]
    else:
        configurations = [
            (
                "beam" if args.beam_width > 1 else "greedy",
                DecodeSettings(
                    beam_width=args.beam_width,
                    top_k=args.top_k,
                    lexicon_weight=args.lexicon_weight if args.use_lexicon else 0.0,
                ),
            )
        ]

    base = CTCRecognizer.from_checkpoint(
        args.checkpoint, device=args.device, decode=configurations[0][1]
    )
    scorer = None
    if lexicon is not None:
        scorer = LexiconScorer(base.charset, lexicon)

    decoders: dict[str, dict[str, object]] = {}

    for name, settings in configurations:
        recognizer = CTCRecognizer(
            base.model,
            base.charset,
            device=base.device,
            max_width=base.max_width,
            decode=settings,
            scorer=scorer,
        )
        LOGGER.info("decoding %d lines with %s", len(selected), name)
        predictions = _predict(recognizer, selected, batch_size=args.batch_size)
        report = _breakdown(
            selected, predictions, case_sensitive=not args.case_insensitive
        )
        overall = report["all"]
        LOGGER.info(
            "%s: cer=%.4f wer=%.4f on %d lines",
            name,
            overall["cer"],
            overall["wer"],
            overall["samples"],
        )
        entry: dict[str, object] = {"settings": asdict(settings), "metrics": report}
        if name == configurations[0][0]:
            entry["examples"] = [
                {
                    "source": record.source,
                    "script": record.script,
                    "reference": record.text_norm,
                    "hypothesis": predictions[index],
                }
                for index, record in enumerate(selected[: args.examples])
            ]
        decoders[name] = entry

    if len(configurations) > 1:
        LOGGER.info("--- CER by configuration ---")
        for name, entry in decoders.items():
            metrics = entry["metrics"]
            assert isinstance(metrics, dict)
            overall = metrics["all"]
            assert isinstance(overall, dict)
            LOGGER.info("%-14s cer=%.4f wer=%.4f", name, overall["cer"], overall["wer"])

    results = {
        "checkpoint": str(args.checkpoint),
        "manifest": str(args.manifest),
        "splits": args.split,
        "samples": len(selected),
        "decoders": decoders,
    }
    payload = json.dumps(results, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
        LOGGER.info("wrote %s", args.output)
    else:
        print(payload)


if __name__ == "__main__":
    main()
