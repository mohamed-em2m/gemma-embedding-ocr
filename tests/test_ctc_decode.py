"""Tests for greedy and prefix-beam CTC decoding."""

from __future__ import annotations

import itertools
import math
from collections import defaultdict
from collections.abc import Sequence

import numpy as np
import pytest

from embedding_ocr.recognition.ctc.decode import (
    Decoded,
    greedy_decode,
    greedy_decode_batch,
    log_softmax,
    mask_padded,
    prefix_beam_search,
)

BLANK = 0


def _collapse(path: Sequence[int], blank: int = BLANK) -> tuple[int, ...]:
    emitted: list[int] = []
    previous = -1
    for index in path:
        if index != blank and index != previous:
            emitted.append(index)
        previous = index
    return tuple(emitted)


def _exact_scores(
    log_probs: np.ndarray, blank: int = BLANK
) -> dict[tuple[int, ...], float]:
    """Enumerate every CTC path and sum probabilities per collapsed label."""
    steps, classes = log_probs.shape
    scores: dict[tuple[int, ...], float] = defaultdict(lambda: -math.inf)
    for path in itertools.product(range(classes), repeat=steps):
        total = sum(float(log_probs[t, index]) for t, index in enumerate(path))
        key = _collapse(path, blank)
        scores[key] = float(np.logaddexp(scores[key], total))
    return dict(scores)


def _one_hot(path: Sequence[int], classes: int) -> np.ndarray:
    log_probs = np.full((len(path), classes), -1e9, dtype=np.float32)
    for step, index in enumerate(path):
        log_probs[step, index] = 0.0
    return log_probs


def test_greedy_collapses_repeats_and_blanks() -> None:
    # a a <blank> a b b  ->  "aab"
    log_probs = _one_hot([1, 1, 0, 1, 2, 2], classes=3)
    decoded = greedy_decode(log_probs)
    assert decoded.indices == (1, 1, 2)
    assert decoded.confidence == pytest.approx(1.0, abs=1e-3)


def test_greedy_confidence_is_mean_probability() -> None:
    log_probs = log_softmax(np.array([[0.0, 2.0, 0.0], [0.0, 0.0, 2.0]]))
    decoded = greedy_decode(log_probs)
    expected = float(np.mean([np.exp(log_probs[0, 1]), np.exp(log_probs[1, 2])]))
    assert decoded.confidence == pytest.approx(expected, rel=1e-5)
    assert 0.0 <= decoded.confidence <= 1.0


def test_greedy_empty_when_all_blank() -> None:
    decoded = greedy_decode(_one_hot([0, 0, 0], classes=3))
    assert decoded == Decoded((), 0.0)


def test_greedy_batch_matches_single() -> None:
    batch = np.stack([_one_hot([1, 1, 2], 3), _one_hot([2, 0, 1], 3)])
    assert greedy_decode_batch(batch) == [
        greedy_decode(batch[0]),
        greedy_decode(batch[1]),
    ]


def test_greedy_rejects_bad_shape() -> None:
    with pytest.raises(ValueError, match="expected \\(T, C\\)"):
        greedy_decode(np.zeros((2, 3, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="expected \\(B, T, C\\)"):
        greedy_decode_batch(np.zeros((3, 4), dtype=np.float32))


@pytest.mark.parametrize(
    ("seed", "steps", "classes"), [(7, 4, 3), (1, 3, 4), (2, 4, 4)]
)
def test_beam_matches_exhaustive_search(seed: int, steps: int, classes: int) -> None:
    rng = np.random.default_rng(seed)
    log_probs = log_softmax(rng.normal(size=(steps, classes)).astype(np.float32))
    exact = _exact_scores(log_probs)
    expected = max(exact.items(), key=lambda item: item[1])[0]

    decoded = prefix_beam_search(log_probs, beam_width=256, top_k=classes)
    assert decoded.indices == expected
    assert 0.0 <= decoded.confidence <= 1.0


def test_beam_matches_greedy_on_confident_input() -> None:
    log_probs = _one_hot([1, 1, 0, 2, 2, 0, 1], classes=3)
    assert prefix_beam_search(log_probs).indices == greedy_decode(log_probs).indices


def test_beam_confidence_is_bounded() -> None:
    rng = np.random.default_rng(3)
    log_probs = log_softmax(rng.normal(size=(12, 5)).astype(np.float32))
    decoded = prefix_beam_search(log_probs, beam_width=8, top_k=5)
    assert 0.0 <= decoded.confidence <= 1.0


class _PreferFirst:
    def score(self, prefix: Sequence[int], next_index: int) -> float:
        del prefix
        return 0.0 if next_index == 1 else -10.0


def test_scorer_can_override_the_acoustic_preference() -> None:
    log_probs = log_softmax(np.array([[0.0, 0.5, 1.0]], dtype=np.float32))
    acoustic = prefix_beam_search(log_probs, beam_width=4, top_k=3)
    rescored = prefix_beam_search(
        log_probs, beam_width=4, top_k=3, scorer=_PreferFirst(), scorer_weight=1.0
    )
    assert acoustic.indices == (2,)
    assert rescored.indices == (1,)


def test_beam_rejects_bad_parameters() -> None:
    log_probs = np.zeros((2, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="beam_width"):
        prefix_beam_search(log_probs, beam_width=0)
    with pytest.raises(ValueError, match="expected \\(T, C\\)"):
        prefix_beam_search(np.zeros((2, 2, 3), dtype=np.float32))


def test_log_softmax_normalizes() -> None:
    probabilities = np.exp(log_softmax(np.array([[1.0, 2.0, 3.0]], dtype=np.float32)))
    assert float(probabilities.sum()) == pytest.approx(1.0, rel=1e-5)


def test_mask_padded_drops_characters_from_padded_steps() -> None:
    # Sample 0 has 4 valid steps; steps 4 and 5 are padding that still emits
    # the same character. Without the mask, decoding appends a spurious "a".
    log_probs = np.full((2, 6, 3), -1e9, dtype=np.float32)
    log_probs[:, :, 0] = 0.0  # blank by default
    log_probs[0, 0, 1] = 0.0
    log_probs[0, 0, 0] = -1e9
    log_probs[1, 0, 2] = 0.0
    log_probs[1, 0, 0] = -1e9
    log_probs[0, 4, 1] = 0.0
    log_probs[0, 4, 0] = -1e9
    log_probs[0, 5, 1] = 0.0
    log_probs[0, 5, 0] = -1e9

    unmasked = greedy_decode_batch(log_probs)
    assert unmasked[0].indices == (1, 1)

    masked = greedy_decode_batch(mask_padded(log_probs, [4, 6]))
    assert masked[0].indices == (1,)
    assert masked[1].indices == (2,)


def test_mask_padded_keeps_valid_region_untouched() -> None:
    rng = np.random.default_rng(0)
    log_probs = log_softmax(rng.normal(size=(1, 5, 4)).astype(np.float32))
    masked = mask_padded(log_probs, [5])
    assert np.array_equal(masked, log_probs)


def test_mask_padded_rejects_bad_shape() -> None:
    with pytest.raises(ValueError, match="expected \\(B, T, C\\)"):
        mask_padded(np.zeros((2, 3), dtype=np.float32), [2])
