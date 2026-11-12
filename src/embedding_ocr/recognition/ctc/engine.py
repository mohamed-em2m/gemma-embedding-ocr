"""Inference engine for the CTC line recognizer.

Wraps a trained checkpoint behind the project-wide
:class:`~embedding_ocr.recognition.base.RecognitionEngine` protocol so it can
be swapped for PaddleOCR or the embedding recognizer by configuration alone.

Preprocessing is shared with training through
:func:`embedding_ocr.training.ctc_dataset.to_model_tensor`, so the crop the
engine sees at inference is byte-identical to what the encoder was trained on.

Arabic text needs no reordering here: the model is trained on logical-order
labels, and CTC emits characters in that same order.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from embedding_ocr.recognition.base import RecognitionResult
from embedding_ocr.recognition.ctc.charset import CharSet
from embedding_ocr.recognition.ctc.checkpoint import load_model
from embedding_ocr.recognition.ctc.decode import (
    greedy_decode,
    log_softmax,
    mask_padded,
    prefix_beam_search,
)
from embedding_ocr.recognition.ctc.lexicon import LexiconScorer, WordLexicon
from embedding_ocr.recognition.ctc.model import CTCSeq2Seq
from embedding_ocr.training.ctc_dataset import (
    resize_to_height,
    to_gray,
    to_model_tensor,
)


def resolve_device(name: str | torch.device) -> torch.device:
    """Resolve ``auto`` to CUDA when available, else CPU."""
    if isinstance(name, torch.device):
        return name
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass(frozen=True)
class DecodeSettings:
    """Decoding strategy for the CTC head.

    Attributes:
        beam_width: ``1`` means greedy; ``> 1`` runs prefix beam search.
        top_k: Candidate classes considered per time step by the beam search.
        lexicon_weight: Multiplier on the lexicon score. ``0`` disables the
            lexicon even when one is configured.
    """

    beam_width: int = 1
    top_k: int = 8
    lexicon_weight: float = 0.0

    def __post_init__(self) -> None:
        if self.beam_width < 1:
            message = "beam_width must be >= 1"
            raise ValueError(message)


class CTCRecognizer:
    """A trained CRNN+CTC line recognizer.

    Attributes:
        charset: Vocabulary the model emits.
        image_height: Height crops are resized to before the encoder.
        max_width: Widest crop accepted; wider crops are resized down.
    """

    def __init__(
        self,
        model: CTCSeq2Seq,
        charset: CharSet,
        *,
        device: str | torch.device = "auto",
        max_width: int = 1280,
        decode: DecodeSettings | None = None,
        scorer: LexiconScorer | None = None,
    ) -> None:
        self.model = model
        self.charset = charset
        self.device = resolve_device(device)
        self.max_width = max_width
        self.decode = decode or DecodeSettings()
        self.scorer = scorer
        self.image_height = model.config.image_height
        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def from_checkpoint(
        cls,
        path: Path,
        *,
        device: str | torch.device = "auto",
        max_width: int = 1280,
        decode: DecodeSettings | None = None,
        lexicon: WordLexicon | None = None,
        lexicon_penalty: float = 2.0,
    ) -> CTCRecognizer:
        """Load a checkpoint written by the trainer.

        Args:
            path: Checkpoint file.
            device: ``auto``, ``cpu``, ``cuda`` or a device string.
            max_width: Widest crop accepted before downscaling.
            decode: Decoding strategy.
            lexicon: Optional word list for prefix-beam rescoring. The scorer
                is built from the checkpoint's own charset.
            lexicon_penalty: Log-domain penalty for impossible prefixes.
        """
        model, charset = load_model(Path(path), device=resolve_device(device))
        scorer = None
        if lexicon is not None and len(lexicon) > 0:
            scorer = LexiconScorer(charset, lexicon, penalty=lexicon_penalty)
        return cls(
            model,
            charset,
            device=device,
            max_width=max_width,
            decode=decode,
            scorer=scorer,
        )

    def _prepare(self, image: np.ndarray) -> tuple[torch.Tensor, int]:
        """Convert one crop into a normalized tensor and its width.

        Training rejects crops wider than ``max_width`` because the CTC time
        axis must stay aligned with the label. Inference has no label, so an
        over-wide crop is downscaled instead of rejected.
        """
        resized = resize_to_height(to_gray(image), self.image_height)
        width = int(resized.shape[1])
        if width > self.max_width:
            resized = cv2.resize(
                resized,
                (self.max_width, self.image_height),
                interpolation=cv2.INTER_AREA,
            )
            width = self.max_width
        return to_model_tensor(resized), width

    def _batch(self, images: Sequence[np.ndarray]) -> tuple[torch.Tensor, torch.Tensor]:
        """Pad a list of crops with white and stack them into one batch."""
        prepared = [self._prepare(image) for image in images]
        height = self.image_height
        widths = torch.tensor([width for _, width in prepared], dtype=torch.long)
        batch = torch.ones(
            (len(prepared), 1, height, int(widths.max())), dtype=torch.float32
        )
        for row, (tensor, width) in enumerate(prepared):
            batch[row, :, :, :width] = tensor
        return batch, widths

    def recognize_batch(self, images: Sequence[np.ndarray]) -> list[RecognitionResult]:
        """Recognize several line crops in one encoder pass.

        Args:
            images: Crops as uint8 arrays, ``(H, W)`` or ``(H, W, C)``.

        Returns:
            One :class:`RecognitionResult` per crop, in input order.
        """
        if not images:
            return []
        batch, widths = self._batch(images)
        with torch.no_grad():
            ctc_logits, _, lengths = self.model(
                batch.to(self.device), widths.to(self.device)
            )
        log_probs = mask_padded(
            log_softmax(ctc_logits.float().cpu().numpy()), lengths.cpu().tolist()
        )
        results: list[RecognitionResult] = []
        for row, length in enumerate(lengths.cpu().tolist()):
            line = log_probs[row, : int(length)]
            results.append(self._decode_line(line))
        return results

    def _decode_line(self, log_probs: np.ndarray) -> RecognitionResult:
        """Decode one line of log-probabilities with the configured strategy."""
        if self.decode.beam_width > 1:
            if self.scorer is not None:
                self.scorer.reset()
            decoded = prefix_beam_search(
                log_probs,
                beam_width=self.decode.beam_width,
                top_k=self.decode.top_k,
                scorer=self.scorer if self.decode.lexicon_weight > 0 else None,
                scorer_weight=self.decode.lexicon_weight,
            )
        else:
            decoded = greedy_decode(log_probs)
        text = self.charset.decode(decoded.indices)
        return RecognitionResult(
            text=text,
            confidence=decoded.confidence,
            metadata={"decoder": "beam" if self.decode.beam_width > 1 else "greedy"},
        )

    def recognize(self, image: np.ndarray) -> RecognitionResult:
        """Recognize a single line crop.

        Args:
            image: Crop as a uint8 array, ``(H, W)`` or ``(H, W, C)``.

        Returns:
            The recognized line and its confidence.
        """
        return self.recognize_batch([image])[0]
