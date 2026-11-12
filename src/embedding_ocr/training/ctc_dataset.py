"""Line-crop dataset, augmentation, and CTC batching.

Images are resized to ``image_height`` keeping the aspect ratio, converted to
grayscale, and normalized so that white background is ``+1.0`` and black ink
is ``-1.0``. Batches pad on the right with white, never with black, because
black padding would look like ink to the encoder.

Targets are encoded with the charset from the manifest directory. The decoder
gets ``[BOS] + label`` as input and ``label + [EOS]`` as target; the CTC head
uses the bare label.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, Sampler

from embedding_ocr.recognition.ctc.charset import CharSet
from embedding_ocr.training.line_dataset import LineRecord

BOS_INDEX = 0
IGNORE_INDEX = -100


@dataclass(frozen=True)
class LineDatasetConfig:
    """Crop preprocessing and augmentation settings."""

    image_height: int = 32
    max_width: int = 1280
    augment: bool = True


@dataclass(frozen=True)
class LineSample:
    """One encoded crop."""

    image: Tensor  # (1, H, W) float32
    target: Tensor  # (L,) int64, charset indices
    width: int
    text: str


@dataclass(frozen=True)
class LineBatch:
    """A padded batch ready for :class:`CTCSeq2Seq`."""

    images: Tensor  # (B, 1, H, Wmax)
    widths: Tensor  # (B,)
    ctc_targets: Tensor  # (sum L,)
    target_lengths: Tensor  # (B,)
    decoder_inputs: Tensor  # (B, Lmax)
    decoder_targets: Tensor  # (B, Lmax), EOS included, padding = -100
    texts: tuple[str, ...]


def load_gray(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        buffer = np.frombuffer(handle.read(), dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_GRAYSCALE)
    if image is None:
        message = f"could not decode image: {path}"
        raise ValueError(message)
    return image


def resize_to_height(image: np.ndarray, height: int) -> np.ndarray:
    """Resize a grayscale crop to ``height`` while keeping the aspect ratio."""
    source_height, source_width = image.shape[:2]
    if source_height <= 0 or source_width <= 0:
        message = "cannot resize an empty image"
        raise ValueError(message)
    if source_height == height:
        return image
    width = max(1, round(source_width * height / source_height))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def to_model_tensor(image: np.ndarray) -> Tensor:
    """Convert an 8-bit grayscale crop into a normalized ``(1, H, W)`` tensor.

    White background maps to ``+1.0`` and black ink to ``-1.0``, which is the
    convention the encoder is trained on. Training and inference both call
    this, so preprocessing cannot drift between them.
    """
    return torch.from_numpy(image).float().div(255.0).sub(0.5).div(0.5).unsqueeze(0)


def to_gray(image: np.ndarray) -> np.ndarray:
    """Return an 8-bit single-channel view of ``image``.

    Accepts ``(H, W)``, ``(H, W, 1)``, ``(H, W, 3)`` and ``(H, W, 4)`` uint8
    arrays, which is what detectors and callers hand over.
    """
    array = np.asarray(image)
    if array.ndim == 2:
        return array.astype(np.uint8, copy=False)
    if array.ndim == 3 and array.shape[2] == 1:
        return array[:, :, 0].astype(np.uint8, copy=False)
    if array.ndim == 3 and array.shape[2] in {3, 4}:
        code = cv2.COLOR_RGBA2GRAY if array.shape[2] == 4 else cv2.COLOR_RGB2GRAY
        return cv2.cvtColor(array.astype(np.uint8, copy=False), code)
    message = f"expected a 2D or 3D uint8 image, got shape {array.shape}"
    raise ValueError(message)


def augment_gray(image: np.ndarray) -> np.ndarray:
    """Apply light handwriting-safe augmentation to a resized crop.

    Uses the global :mod:`random` state so DataLoader workers produce a
    different augmentation each epoch while the run stays reproducible from
    its seed.
    """
    height, width = image.shape[:2]
    angle = random.uniform(-2.0, 2.0)
    scale = random.uniform(0.9, 1.1)
    shear = math.tan(math.radians(random.uniform(-3.0, 3.0)))
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, scale)
    matrix[0, 1] += shear
    image = cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255.0, 255.0, 255.0),
    )

    alpha = random.uniform(0.9, 1.1)
    beta = random.uniform(-10.0, 10.0)
    image = np.clip(image.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

    if random.random() < 0.3:
        image = cv2.GaussianBlur(image, (3, 3), sigmaX=random.uniform(0.3, 0.8))

    if random.random() < 0.2:
        area = random.uniform(0.01, 0.03) * width * height
        box_w = max(2, int(math.sqrt(area)))
        box_h = max(2, int(area / box_w))
        x = random.randrange(0, max(1, width - box_w))
        y = random.randrange(0, max(1, height - box_h))
        image[y : y + box_h, x : x + box_w] = 255
    return image


class LineCropDataset(Dataset[LineSample]):
    """A manifest-backed dataset of line crops."""

    def __init__(
        self,
        records: Sequence[LineRecord],
        charset: CharSet,
        config: LineDatasetConfig | None = None,
        *,
        max_decoder_len: int = 128,
    ) -> None:
        self.records = list(records)
        self.charset = charset
        self.config = config or LineDatasetConfig()
        self.max_decoder_len = max_decoder_len

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> LineSample:
        record = self.records[index]
        target = self.charset.encode(record.text_norm)
        if len(target) + 1 > self.max_decoder_len:
            message = (
                f"label of {len(target)} characters exceeds max_decoder_len="
                f"{self.max_decoder_len} for {record.image}"
            )
            raise ValueError(message)

        image = resize_to_height(load_gray(record.image), self.config.image_height)
        if image.shape[1] > self.config.max_width:
            message = (
                f"crop width {image.shape[1]} exceeds max_width="
                f"{self.config.max_width} for {record.image}"
            )
            raise ValueError(message)
        if self.config.augment:
            image = augment_gray(image)

        return LineSample(
            image=to_model_tensor(image),
            target=torch.tensor(target, dtype=torch.long),
            width=int(image.shape[1]),
            text=record.text_norm,
        )

    def sort_key(self, index: int) -> float:
        """Aspect-preserving width estimate used by the batch sampler."""
        record = self.records[index]
        if record.height <= 0:
            return 0.0
        return record.width * self.config.image_height / record.height


def collate_lines(batch: Sequence[LineSample], *, eos_index: int) -> LineBatch:
    """Pad crops with white and build CTC and decoder targets."""
    if not batch:
        message = "cannot collate an empty batch"
        raise ValueError(message)

    height = batch[0].image.shape[1]
    widths = torch.tensor([sample.width for sample in batch], dtype=torch.long)
    max_width = int(widths.max())
    images = torch.ones((len(batch), 1, height, max_width), dtype=torch.float32)
    for row, sample in enumerate(batch):
        images[row, :, :, : sample.width] = sample.image

    targets = [sample.target for sample in batch]
    target_lengths = torch.tensor([len(target) for target in targets], dtype=torch.long)
    ctc_targets = torch.cat(targets) if targets else torch.zeros(0, dtype=torch.long)

    # The decoder is trained with an explicit EOS so generation can stop.
    decoder_inputs = torch.zeros(
        (len(batch), int(target_lengths.max()) + 1), dtype=torch.long
    )
    decoder_targets = torch.full_like(decoder_inputs, IGNORE_INDEX)
    for row, target in enumerate(targets):
        length = len(target)
        decoder_inputs[row, 1 : length + 1] = target
        decoder_targets[row, :length] = target
        decoder_targets[row, length] = eos_index

    return LineBatch(
        images=images,
        widths=widths,
        ctc_targets=ctc_targets,
        target_lengths=target_lengths,
        decoder_inputs=decoder_inputs,
        decoder_targets=decoder_targets,
        texts=tuple(sample.text for sample in batch),
    )


class BucketBatchSampler(Sampler[list[int]]):
    """Sort by width, chunk into batches, then shuffle batch order.

    Width bucketing keeps right-padding small, which matters because every
    padded column still costs encoder and LSTM time steps.
    """

    def __init__(
        self,
        widths: Sequence[float],
        batch_size: int,
        *,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        if batch_size < 1:
            message = "batch_size must be >= 1"
            raise ValueError(message)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self._widths = [float(width) for width in widths]
        self._epoch = 0

    def __iter__(self) -> Iterator[list[int]]:
        order = sorted(range(len(self._widths)), key=lambda index: self._widths[index])
        batches = [
            order[start : start + self.batch_size]
            for start in range(0, len(order), self.batch_size)
        ]
        if self.drop_last and batches and len(batches[-1]) < self.batch_size:
            batches = batches[:-1]
        if self.shuffle:
            random.Random(self.seed + self._epoch).shuffle(batches)
        self._epoch += 1
        yield from batches

    def __len__(self) -> int:
        if self.drop_last:
            return len(self._widths) // self.batch_size
        return math.ceil(len(self._widths) / self.batch_size)
