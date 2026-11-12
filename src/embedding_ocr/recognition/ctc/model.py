"""CRNN encoder with a CTC head and a small transformer decoder.

The encoder downsamples height by 32 (``image_height=32`` becomes one row) and
width by **2**. Width stride 4, the classic CRNN choice, cannot represent long
Arabic lines: on 400 KHATT lines only 37.5% satisfy the CTC constraint
``T >= 2L - 1`` at stride 4, versus 84.8% at stride 2 (see
``docs/decisions.md``).

Shapes:
    images          (B, 1, H, W)  float32, white background = +1.0
    memory          (B, T, D)     D = 2 * rnn_hidden
    ctc_logits      (B, T, V)     V = charset size, index 0 is blank
    decoder_logits  (B, L, V)     teacher-forced next-token logits
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

BLANK_INDEX = 0


@dataclass(frozen=True)
class ModelConfig:
    """Architecture hyperparameters, persisted in every checkpoint."""

    vocab_size: int
    image_height: int = 32
    channels: tuple[int, ...] = (32, 64, 128, 256, 256)
    norm_groups: int = 8
    rnn_hidden: int = 256
    rnn_layers: int = 2
    dropout: float = 0.1
    decoder_layers: int = 3
    decoder_dim: int = 256
    decoder_heads: int = 4
    decoder_ffn: int = 1024
    max_decoder_len: int = 128

    def __post_init__(self) -> None:
        if 2 ** len(self.channels) != self.image_height:
            message = (
                f"image_height={self.image_height} needs {math.log2(self.image_height):.0f} "
                f"conv blocks, got {len(self.channels)}"
            )
            raise ValueError(message)
        if self.decoder_dim % self.decoder_heads:
            message = "decoder_dim must be divisible by decoder_heads"
            raise ValueError(message)

    @property
    def encoder_dim(self) -> int:
        return 2 * self.rnn_hidden

    @property
    def decoder_vocab_size(self) -> int:
        """Decoder vocabulary: every charset symbol plus EOS."""
        return self.vocab_size + 1

    @property
    def eos_index(self) -> int:
        """Index of the end-of-sequence token."""
        return self.vocab_size


class CRNNEncoder(nn.Module):
    """Conv stack (height /32, width /2) followed by a BiLSTM.

    The conv blocks use :class:`torch.nn.GroupNorm`, not BatchNorm. The
    encoder collapses height to a single row, so BatchNorm's per-channel
    statistics are computed over ``B * T`` values and leak information about
    the *other samples in the batch*. On a small memorization fixture the
    model exploited that: it emitted confident characters over the blank
    right edge of the crop and its loss changed by 10x with batch
    composition (see ``docs/decisions.md``). GroupNorm normalizes per sample,
    so training and evaluation agree and the loss is batch-size independent.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        in_channels = 1
        for index, channels in enumerate(config.channels):
            stride = (2, 2) if index == 0 else (2, 1)
            blocks.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels, channels, 3, stride=stride, padding=1, bias=False
                    ),
                    nn.GroupNorm(min(config.norm_groups, channels), channels),
                    nn.ReLU(inplace=True),
                )
            )
            in_channels = channels
        self.blocks = nn.Sequential(*blocks)
        self.rnn = nn.LSTM(
            in_channels,
            config.rnn_hidden,
            num_layers=config.rnn_layers,
            bidirectional=True,
            batch_first=True,
            dropout=config.dropout if config.rnn_layers > 1 else 0.0,
        )

    def forward(self, images: Tensor) -> Tensor:
        features = self.blocks(images)  # (B, C, 1, T)
        features = features.squeeze(2).transpose(1, 2)  # (B, T, C)
        output, _ = self.rnn(features)
        return output


class CTCHead(nn.Module):
    """Linear projection from encoder features to character logits."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.projection = nn.Linear(config.encoder_dim, config.vocab_size)

    def forward(self, memory: Tensor) -> Tensor:
        return self.projection(memory)


class TransformerDecoder(nn.Module):
    """Small autoregressive decoder with cross-attention over encoder memory."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.decoder_vocab_size, config.decoder_dim)
        self.position = nn.Embedding(config.max_decoder_len, config.decoder_dim)
        self.memory_projection = nn.Linear(config.encoder_dim, config.decoder_dim)
        layer = nn.TransformerDecoderLayer(
            d_model=config.decoder_dim,
            nhead=config.decoder_heads,
            dim_feedforward=config.decoder_ffn,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.layers = nn.TransformerDecoder(layer, num_layers=config.decoder_layers)
        self.norm = nn.LayerNorm(config.decoder_dim)
        self.output = nn.Linear(config.decoder_dim, config.decoder_vocab_size)

    def forward(
        self,
        tokens: Tensor,
        memory: Tensor,
        memory_key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        """Decode ``tokens`` (B, L) against ``memory`` (B, T, D)."""
        length = tokens.shape[1]
        if length > self.config.max_decoder_len:
            message = (
                f"decoder length {length} exceeds max_decoder_len="
                f"{self.config.max_decoder_len}"
            )
            raise ValueError(message)
        positions = torch.arange(length, device=tokens.device)
        hidden = self.embedding(tokens) + self.position(positions)
        causal = nn.Transformer.generate_square_subsequent_mask(
            length, device=tokens.device
        )
        decoded = self.layers(
            hidden,
            self.memory_projection(memory),
            tgt_mask=causal,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return self.output(self.norm(decoded))


class CTCSeq2Seq(nn.Module):
    """Encoder + CTC head + optional transformer decoder."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = CRNNEncoder(config)
        self.ctc_head = CTCHead(config)
        self.decoder = TransformerDecoder(config)

    def encode(self, images: Tensor, widths: Tensor) -> tuple[Tensor, Tensor]:
        """Return encoder memory and the valid time steps per sample.

        Args:
            images: ``(B, 1, H, W)`` padded crops.
            widths: ``(B,)`` unpadded crop widths before batch padding.
        """
        memory = self.encoder(images)
        lengths = torch.div(widths + 1, 2, rounding_mode="floor").clamp(
            max=memory.shape[1]
        )
        return memory, lengths

    def forward(
        self,
        images: Tensor,
        widths: Tensor,
        decoder_tokens: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None, Tensor]:
        """Run the encoder and both heads.

        Args:
            images: ``(B, 1, H, W)`` crops.
            widths: ``(B,)`` valid widths.
            decoder_tokens: ``(B, L)`` teacher-forcing inputs, or ``None``.

        Returns:
            ``(ctc_logits (B,T,V), decoder_logits (B,L,V) or None, lengths)``.
        """
        memory, lengths = self.encode(images, widths)
        ctc_logits = self.ctc_head(memory)
        if decoder_tokens is None:
            return ctc_logits, None, lengths
        positions = torch.arange(memory.shape[1], device=memory.device)
        memory_mask = positions[None, :] >= lengths[:, None]
        decoder_logits = self.decoder(decoder_tokens, memory, memory_mask)
        return ctc_logits, decoder_logits, lengths

    @torch.no_grad()
    def greedy_decode(self, images: Tensor, widths: Tensor) -> list[list[int]]:
        """Autoregressive decoder output, greedy, stopping at EOS.

        Returns the generated tokens with EOS and everything after it removed.
        """
        memory, lengths = self.encode(images, widths)
        positions = torch.arange(memory.shape[1], device=memory.device)
        memory_mask = positions[None, :] >= lengths[:, None]
        tokens = torch.full(
            (images.shape[0], 1), BLANK_INDEX, dtype=torch.long, device=images.device
        )
        finished = torch.zeros(images.shape[0], dtype=torch.bool, device=images.device)
        for _ in range(self.config.max_decoder_len - 1):
            logits = self.decoder(tokens, memory, memory_mask)
            next_token = logits[:, -1].argmax(dim=-1)
            tokens = torch.cat([tokens, next_token[:, None]], dim=1)
            finished |= next_token == self.config.eos_index
            if bool(finished.all()):
                break
        eos = self.config.eos_index
        return [[token for token in row[1:].tolist() if token != eos] for row in tokens]

    def parameter_count(self) -> int:
        """Number of trainable parameters."""
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )
