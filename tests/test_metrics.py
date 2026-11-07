"""Tests for OCR metrics."""

from __future__ import annotations

from embedding_ocr.eval.metrics import (
    character_error_rate,
    evaluate,
    exact_match,
    normalize_text,
    word_error_rate,
)


def test_normalize_text() -> None:
    assert normalize_text("  Hello   WORLD \n") == "hello world"


def test_perfect_predictions() -> None:
    refs = ["hello world", "invoice 2026"]
    hyps = ["Hello   World", "invoice 2026"]
    assert character_error_rate(refs, hyps) == 0.0
    assert word_error_rate(refs, hyps) == 0.0
    assert exact_match(refs, hyps) == 1.0


def test_error_rates_are_positive() -> None:
    refs = ["hello world"]
    hyps = ["helo wrld"]
    assert character_error_rate(refs, hyps) > 0.0
    assert word_error_rate(refs, hyps) > 0.0
    assert exact_match(refs, hyps) == 0.0


def test_evaluate_returns_all_metrics() -> None:
    metrics = evaluate(["abc"], ["abc"])
    assert metrics["cer"] == 0.0
    assert metrics["wer"] == 0.0
    assert metrics["exact_match"] == 1.0
    assert metrics["num_samples"] == 1
