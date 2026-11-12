"""CTC decoding: greedy collapse and prefix beam search.

Decoders consume **log-probabilities** of shape ``(T, C)`` where ``C`` is the
charset size (blank included). Use :func:`log_softmax` on raw model logits.

Both decoders return :class:`Decoded` with the logit indices of the emitted
characters and a confidence in ``[0, 1]``:

* greedy confidence is the mean probability of the emitted characters,
* beam confidence is the geometric mean probability of the best prefix,
  including the paths that collapsed onto it.

``PrefixScorer`` is the hook for word-level language-model or lexicon
rescoring; the default scorer is neutral, so decoding is exact CTC search.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

BLANK_INDEX = 0
_NEG_INF = float("-inf")


class PrefixScorer(Protocol):
    """Score the next token given the emitted prefix.

    Implementations must return a log-domain score; ``0.0`` means neutral.
    """

    def score(self, prefix: Sequence[int], next_index: int) -> float: ...


@dataclass(frozen=True)
class Decoded:
    """A decoded sequence.

    Attributes:
        indices: Emitted logit indices, blank excluded.
        confidence: Confidence in ``[0, 1]``.
    """

    indices: tuple[int, ...]
    confidence: float


def log_softmax(logits: np.ndarray) -> np.ndarray:
    """Return log-probabilities along the last axis, numerically stable."""
    array = np.asarray(logits, dtype=np.float32)
    shifted = array - array.max(axis=-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


def _logsumexp(values: Sequence[float]) -> float:
    finite = [value for value in values if value != _NEG_INF]
    if not finite:
        return _NEG_INF
    peak = max(finite)
    return peak + float(np.log(sum(np.exp(value - peak) for value in finite)))


def greedy_decode(log_probs: np.ndarray, blank: int = BLANK_INDEX) -> Decoded:
    """Decode by per-timestep argmax, collapsing repeats and dropping blanks."""
    probs = np.asarray(log_probs, dtype=np.float32)
    if probs.ndim != 2:
        message = f"expected (T, C) log-probabilities, got shape {probs.shape}"
        raise ValueError(message)

    best = probs.argmax(axis=1)
    emitted: list[int] = []
    confidences: list[float] = []
    previous = -1
    for step, index in enumerate(best):
        index = int(index)
        if index != blank and index != previous:
            emitted.append(index)
            confidences.append(float(np.exp(probs[step, index])))
        previous = index
    confidence = float(np.mean(confidences)) if confidences else 0.0
    return Decoded(tuple(emitted), min(max(confidence, 0.0), 1.0))


def greedy_decode_batch(
    log_probs: np.ndarray, blank: int = BLANK_INDEX
) -> list[Decoded]:
    """Greedy-decode a ``(B, T, C)`` batch."""
    probs = np.asarray(log_probs, dtype=np.float32)
    if probs.ndim != 3:
        message = f"expected (B, T, C) log-probabilities, got shape {probs.shape}"
        raise ValueError(message)
    return [greedy_decode(probs[index], blank=blank) for index in range(probs.shape[0])]


def _top_candidates(row: np.ndarray, top_k: int) -> list[int]:
    size = row.shape[0]
    if top_k >= size:
        return [int(index) for index in np.argsort(row)[::-1]]
    selected = np.argpartition(row, -top_k)[-top_k:]
    return [int(index) for index in selected[np.argsort(row[selected])[::-1]]]


def prefix_beam_search(
    log_probs: np.ndarray,
    *,
    blank: int = BLANK_INDEX,
    beam_width: int = 10,
    top_k: int = 8,
    scorer: PrefixScorer | None = None,
    scorer_weight: float = 1.0,
) -> Decoded:
    """CTC prefix beam search with optional prefix rescoring.

    Args:
        log_probs: ``(T, C)`` log-probabilities.
        blank: Blank index.
        beam_width: Number of prefixes kept per timestep.
        top_k: Candidate classes considered per timestep. Set to ``C`` for an
            exhaustive (slow) search, which is what the tests compare against.
        scorer: Optional prefix scorer (language model or lexicon).
        scorer_weight: Multiplier for the scorer's log-domain score.

    Returns:
        The best prefix and its geometric-mean confidence.
    """
    probs = np.asarray(log_probs, dtype=np.float32)
    if probs.ndim != 2:
        message = f"expected (T, C) log-probabilities, got shape {probs.shape}"
        raise ValueError(message)
    if beam_width < 1:
        message = "beam_width must be >= 1"
        raise ValueError(message)

    # prefix -> [log P(prefix ending in blank), log P(prefix ending in non-blank)]
    beams: dict[tuple[int, ...], list[float]] = {(): [0.0, _NEG_INF]}

    for step in range(probs.shape[0]):
        candidates = _top_candidates(probs[step], top_k)
        next_beams: dict[tuple[int, ...], list[float]] = defaultdict(
            lambda: [_NEG_INF, _NEG_INF]
        )
        for prefix, (prob_blank, prob_non_blank) in beams.items():
            prob_total = _logsumexp([prob_blank, prob_non_blank])
            for index in candidates:
                log_p = float(probs[step, index])
                if index == blank:
                    slot = next_beams[prefix]
                    slot[0] = _logsumexp([slot[0], prob_total + log_p])
                    continue
                if prefix and prefix[-1] == index:
                    # A repeated label character stays in the same prefix, so
                    # only blank-ended paths can extend to ``prefix + index``.
                    slot = next_beams[prefix]
                    slot[1] = _logsumexp([slot[1], prob_non_blank + log_p])
                    extension = prob_blank + log_p
                else:
                    extension = prob_total + log_p
                extended = (*prefix, index)
                bonus = 0.0
                if scorer is not None:
                    bonus = scorer_weight * scorer.score(prefix, index)
                slot = next_beams[extended]
                slot[1] = _logsumexp([slot[1], extension + bonus])

        if not next_beams:
            break
        ranked = sorted(
            next_beams.items(),
            key=lambda item: _logsumexp(item[1]),
            reverse=True,
        )[:beam_width]
        beams = dict(ranked)

    best_prefix, best_scores = max(beams.items(), key=lambda item: _logsumexp(item[1]))
    log_probability = _logsumexp(best_scores)
    length = max(1, len(best_prefix))
    confidence = float(np.exp(log_probability / length))
    return Decoded(best_prefix, min(max(confidence, 0.0), 1.0))
