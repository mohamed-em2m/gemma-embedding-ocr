"""OCR evaluation metrics."""

from __future__ import annotations

import re
from collections.abc import Sequence

import jiwer

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Lowercase and collapse whitespace for fair comparison."""
    return _WHITESPACE_RE.sub(" ", text.lower()).strip()


def character_error_rate(
    references: Sequence[str], hypotheses: Sequence[str], normalize: bool = True
) -> float:
    """Compute character error rate (CER)."""
    if not references:
        return 0.0
    refs = (
        [normalize_text(text) for text in references] if normalize else list(references)
    )
    hyps = (
        [normalize_text(text) for text in hypotheses] if normalize else list(hypotheses)
    )
    return float(jiwer.cer(refs, hyps))


def word_error_rate(
    references: Sequence[str], hypotheses: Sequence[str], normalize: bool = True
) -> float:
    """Compute word error rate (WER)."""
    if not references:
        return 0.0
    refs = (
        [normalize_text(text) for text in references] if normalize else list(references)
    )
    hyps = (
        [normalize_text(text) for text in hypotheses] if normalize else list(hypotheses)
    )
    return float(jiwer.wer(refs, hyps))


def exact_match(
    references: Sequence[str], hypotheses: Sequence[str], normalize: bool = True
) -> float:
    """Fraction of hypotheses that exactly match their reference."""
    if not references:
        return 0.0
    matches = 0
    for reference, hypothesis in zip(references, hypotheses, strict=False):
        ref = normalize_text(reference) if normalize else reference
        hyp = normalize_text(hypothesis) if normalize else hypothesis
        matches += int(ref == hyp)
    return matches / len(references)


def evaluate(
    references: Sequence[str], hypotheses: Sequence[str], normalize: bool = True
) -> dict[str, float]:
    """Return CER, WER, and exact-match accuracy in one dictionary."""
    return {
        "cer": character_error_rate(references, hypotheses, normalize=normalize),
        "wer": word_error_rate(references, hypotheses, normalize=normalize),
        "exact_match": exact_match(references, hypotheses, normalize=normalize),
        "num_samples": float(len(references)),
    }
