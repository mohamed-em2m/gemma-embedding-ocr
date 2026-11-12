"""Tests for the CTC model, checkpointing, and the line dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from embedding_ocr.recognition.ctc.charset import CharSet
from embedding_ocr.recognition.ctc.checkpoint import (
    build_model,
    describe_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from embedding_ocr.recognition.ctc.model import CTCSeq2Seq, ModelConfig
from embedding_ocr.training.ctc_dataset import (
    BucketBatchSampler,
    LineCropDataset,
    LineDatasetConfig,
    collate_lines,
    resize_to_height,
)
from embedding_ocr.training.line_dataset import LineRecord


def _config(**overrides: object) -> ModelConfig:
    values: dict[str, object] = {
        "vocab_size": 10,
        "image_height": 8,
        "channels": (8, 16, 32),
        "rnn_hidden": 16,
        "rnn_layers": 2,
        "dropout": 0.0,
        "decoder_layers": 1,
        "decoder_dim": 32,
        "decoder_heads": 4,
        "decoder_ffn": 64,
        "max_decoder_len": 16,
    }
    values.update(overrides)
    return ModelConfig(**values)  # type: ignore[arg-type]


def _charset() -> CharSet:
    return CharSet(("a", "b", "c", "d", "e", "f", "g", "h", " "))


def test_model_config_validates_blocks_and_heads() -> None:
    with pytest.raises(ValueError, match="conv blocks"):
        ModelConfig(vocab_size=10, image_height=32, channels=(8, 16))
    with pytest.raises(ValueError, match="divisible"):
        ModelConfig(vocab_size=10, decoder_dim=30, decoder_heads=4)


def test_encoder_time_axis_is_width_over_two() -> None:
    model = CTCSeq2Seq(_config())
    images = torch.zeros(2, 1, 8, 160)
    widths = torch.tensor([160, 80])
    memory, lengths = model.encode(images, widths)
    assert memory.shape == (2, 80, 32)
    assert lengths.tolist() == [80, 40]


def test_encoder_uses_no_batch_norm() -> None:
    model = CTCSeq2Seq(_config())
    assert not any(
        isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
        for module in model.encoder.modules()
    )


def test_encoder_output_does_not_depend_on_batch_companions() -> None:
    """GroupNorm keeps a sample's features independent of the rest of the batch.

    BatchNorm leaked batch composition into the features, which let the model
    memorize fixture batches instead of glyphs.
    """
    torch.manual_seed(0)
    model = CTCSeq2Seq(_config())
    model.train()
    alone = torch.randn(1, 1, 8, 64)
    companions = torch.randn(3, 1, 8, 64) * 5.0
    with torch.no_grad():
        single, _ = model.encode(alone, torch.tensor([64]))
        mixed, _ = model.encode(
            torch.cat([alone, companions]), torch.tensor([64, 64, 64, 64])
        )
    assert torch.allclose(single[0], mixed[0], atol=1e-5)


def test_forward_shapes() -> None:
    model = CTCSeq2Seq(_config())
    images = torch.zeros(2, 1, 8, 100)
    widths = torch.tensor([100, 60])
    tokens = torch.zeros(2, 5, dtype=torch.long)
    ctc_logits, decoder_logits, lengths = model(images, widths, tokens)
    assert ctc_logits.shape == (2, 50, 10)
    assert decoder_logits is not None
    assert decoder_logits.shape == (2, 5, 11)  # charset + EOS
    assert lengths.tolist() == [50, 30]


def test_lengths_are_clamped_to_memory() -> None:
    model = CTCSeq2Seq(_config())
    images = torch.zeros(1, 1, 8, 32)
    _, lengths = model.encode(images, torch.tensor([4096]))
    assert int(lengths.item()) == 16


def test_memorizes_one_batch() -> None:
    torch.manual_seed(0)
    model = CTCSeq2Seq(_config())
    images = torch.rand(2, 1, 8, 64)
    widths = torch.tensor([64, 48])
    targets = torch.tensor([1, 2, 3, 4])
    target_lengths = torch.tensor([2, 2])
    tokens = torch.tensor([[0, 1, 2], [0, 3, 4]])
    decoder_targets = torch.tensor([[1, 2, 10], [3, 4, 10]])
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

    def step() -> float:
        optimizer.zero_grad()
        ctc_logits, decoder_logits, lengths = model(images, widths, tokens)
        log_probs = ctc_logits.log_softmax(dim=-1).transpose(0, 1)
        ctc_loss = torch.nn.functional.ctc_loss(
            log_probs, targets, lengths, target_lengths, blank=0
        )
        ce_loss = torch.nn.functional.cross_entropy(
            decoder_logits.reshape(-1, 11), decoder_targets.reshape(-1)
        )
        loss = 0.5 * ctc_loss + 0.5 * ce_loss
        loss.backward()
        optimizer.step()
        return float(loss.detach())

    first = step()
    for _ in range(60):
        last = step()
    assert last < first * 0.5


def test_greedy_decode_returns_index_lists() -> None:
    model = CTCSeq2Seq(_config()).eval()
    images = torch.zeros(2, 1, 8, 64)
    widths = torch.tensor([64, 40])
    generated = model.greedy_decode(images, widths)
    assert len(generated) == 2
    assert all(isinstance(row, list) for row in generated)
    assert all(token < 11 for row in generated for token in row)


def test_decoder_attends_to_encoder_memory() -> None:
    """Different memories must produce different decoder logits.

    Guards against a cross-attention wiring bug where the decoder degenerates
    into a position-only language model and ignores the image.
    """
    torch.manual_seed(0)
    model = CTCSeq2Seq(_config()).eval()
    tokens = torch.zeros(2, 4, dtype=torch.long)
    first = torch.randn(2, 20, model.config.encoder_dim)
    second = torch.randn(2, 20, model.config.encoder_dim)
    with torch.no_grad():
        logits_first = model.decoder(tokens, first)
        logits_second = model.decoder(tokens, second)
    assert not torch.allclose(logits_first, logits_second, atol=1e-4)


def test_decoder_loss_reaches_the_encoder() -> None:
    """The cross-entropy term must train the shared encoder, not just the head."""
    torch.manual_seed(0)
    model = CTCSeq2Seq(_config())
    images = torch.rand(1, 1, 8, 32)
    widths = torch.tensor([32])
    tokens = torch.zeros(1, 3, dtype=torch.long)
    targets = torch.tensor([[1, 2, 10]])
    _, decoder_logits, _ = model(images, widths, tokens)
    loss = torch.nn.functional.cross_entropy(
        decoder_logits.reshape(-1, 11), targets.reshape(-1)
    )
    loss.backward()
    first_conv = model.encoder.blocks[0][0]
    assert first_conv.weight.grad is not None
    assert float(first_conv.weight.grad.abs().sum()) > 0.0
    # The CTC head is only trained by the CTC term.
    assert model.ctc_head.projection.weight.grad is None


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    charset = _charset()
    model = CTCSeq2Seq(_config(vocab_size=charset.size))
    path = tmp_path / "model.pt"
    save_checkpoint(
        path, model, charset=charset, epoch=3, step=120, metrics={"ctc_cer": 0.2}
    )

    payload = load_checkpoint(path)
    assert payload["charset"] == list(charset.chars)
    assert describe_checkpoint(path)["epoch"] == 3

    restored, restored_charset = build_model(payload)
    assert restored_charset == charset
    for key, value in model.state_dict().items():
        assert torch.equal(restored.state_dict()[key], value)


def test_checkpoint_rejects_charset_mismatch(tmp_path: Path) -> None:
    charset = _charset()
    model = CTCSeq2Seq(_config(vocab_size=charset.size))
    path = tmp_path / "model.pt"
    save_checkpoint(path, model, charset=charset)
    payload = load_checkpoint(path)
    payload["charset"] = ["a", "b"]
    with pytest.raises(ValueError, match="vocab_size"):
        build_model(payload)


# --- dataset -----------------------------------------------------------------


def _record(
    tmp_path: Path, name: str, text: str, size: tuple[int, int] = (160, 32)
) -> LineRecord:
    path = tmp_path / f"{name}.png"
    Image.new("L", size, 255).save(path)
    return LineRecord(
        image=path,
        text=text,
        text_norm=text,
        script="latin",
        split="train",
        source="fake",
        orig_id=name,
        width=size[0],
        height=size[1],
    )


def test_resize_keeps_aspect_ratio() -> None:
    image = np.zeros((64, 320), dtype=np.uint8)
    resized = resize_to_height(image, 32)
    assert resized.shape == (32, 160)
    with pytest.raises(ValueError, match="empty image"):
        resize_to_height(np.zeros((0, 10), dtype=np.uint8), 32)


def test_dataset_returns_normalized_crop(tmp_path: Path) -> None:
    charset = CharSet(("a", "b", " "))
    dataset = LineCropDataset(
        [_record(tmp_path, "one", "ab", size=(200, 40))],
        charset,
        LineDatasetConfig(image_height=32, augment=False),
    )
    sample = dataset[0]
    assert sample.image.shape == (1, 32, 160)
    assert sample.image.min() >= -1.0
    assert sample.image.max() <= 1.0
    assert sample.target.tolist() == charset.encode("ab")
    assert sample.width == 160
    assert sample.text == "ab"


def test_dataset_rejects_oversized_and_long_labels(tmp_path: Path) -> None:
    charset = CharSet(("a", "b"))
    wide = LineCropDataset(
        [_record(tmp_path, "wide", "ab", size=(4000, 32))],
        charset,
        LineDatasetConfig(image_height=32, max_width=1280),
    )
    with pytest.raises(ValueError, match="max_width"):
        _ = wide[0]

    long = LineCropDataset(
        [_record(tmp_path, "long", "ab" * 10, size=(400, 32))],
        charset,
        LineDatasetConfig(image_height=32),
        max_decoder_len=5,
    )
    with pytest.raises(ValueError, match="max_decoder_len"):
        _ = long[0]


def test_collate_pads_with_white_and_adds_eos(tmp_path: Path) -> None:
    charset = CharSet(("a", "b"))
    dataset = LineCropDataset(
        [
            _record(tmp_path, "one", "ab", size=(64, 32)),
            _record(tmp_path, "two", "a", size=(32, 32)),
        ],
        charset,
        LineDatasetConfig(image_height=32, augment=False),
    )
    batch = collate_lines([dataset[0], dataset[1]], eos_index=charset.size)

    assert batch.images.shape == (2, 1, 32, 64)
    assert batch.widths.tolist() == [64, 32]
    assert torch.allclose(batch.images[1, :, :, 32:], torch.ones(1, 32, 32))
    assert batch.ctc_targets.tolist() == [1, 2, 1]
    assert batch.target_lengths.tolist() == [2, 1]
    assert batch.decoder_inputs.tolist() == [[0, 1, 2], [0, 1, 0]]
    assert batch.decoder_targets.tolist() == [[1, 2, 3], [1, 3, -100]]
    assert batch.texts == ("ab", "a")


def test_bucket_sampler_covers_every_index_once() -> None:
    widths = [10.0, 1.0, 9.0, 2.0, 8.0, 3.0]
    sampler = BucketBatchSampler(widths, batch_size=2, shuffle=False)
    batches = list(sampler)
    assert len(sampler) == 3
    assert sorted(index for batch in batches for index in batch) == list(range(6))
    narrowest = max(widths[index] for index in batches[0])
    widest = min(widths[index] for index in batches[-1])
    assert narrowest < widest


def test_bucket_sampler_rejects_bad_batch_size() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        BucketBatchSampler([1.0], batch_size=0)
