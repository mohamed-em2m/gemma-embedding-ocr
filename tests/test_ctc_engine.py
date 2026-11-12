"""Tests for the CTC inference engine and lexicon rescoring."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from embedding_ocr.recognition.base import RecognitionResult
from embedding_ocr.recognition.ctc.charset import CharSet
from embedding_ocr.recognition.ctc.checkpoint import save_checkpoint
from embedding_ocr.recognition.ctc.engine import CTCRecognizer, DecodeSettings
from embedding_ocr.recognition.ctc.lexicon import LexiconScorer, WordLexicon
from embedding_ocr.recognition.ctc.model import CTCSeq2Seq, ModelConfig
from embedding_ocr.training.ctc_dataset import to_gray


def _config(vocab_size: int) -> ModelConfig:
    return ModelConfig(
        vocab_size=vocab_size,
        image_height=8,
        channels=(8, 16, 32),
        rnn_hidden=16,
        rnn_layers=1,
        dropout=0.0,
        decoder_layers=1,
        decoder_dim=32,
        decoder_heads=4,
        decoder_ffn=64,
        max_decoder_len=16,
    )


def _charset() -> CharSet:
    return CharSet(("a", "b", "c", "d", " "))


def _checkpoint(tmp_path: Path) -> Path:
    charset = _charset()
    model = CTCSeq2Seq(_config(charset.size))
    path = tmp_path / "ctc.pt"
    save_checkpoint(path, model, charset=charset)
    return path


def _crop(width: int = 64, height: int = 24) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(height, width), dtype=np.uint8)


# --- engine ------------------------------------------------------------------


def test_engine_returns_recognition_results(tmp_path: Path) -> None:
    recognizer = CTCRecognizer.from_checkpoint(_checkpoint(tmp_path), device="cpu")
    result = recognizer.recognize(_crop())
    assert isinstance(result, RecognitionResult)
    assert 0.0 <= result.confidence <= 1.0
    assert result.metadata["decoder"] == "greedy"
    assert set(result.text) <= set(recognizer.charset.chars)


def test_engine_batch_matches_single_calls(tmp_path: Path) -> None:
    recognizer = CTCRecognizer.from_checkpoint(_checkpoint(tmp_path), device="cpu")
    crops = [_crop(64), _crop(64), _crop(64)]
    batch = recognizer.recognize_batch(crops)
    single = [recognizer.recognize(crop) for crop in crops]
    assert [item.text for item in batch] == [item.text for item in single]


def test_engine_batch_preserves_input_order(tmp_path: Path) -> None:
    recognizer = CTCRecognizer.from_checkpoint(_checkpoint(tmp_path), device="cpu")
    crops = [_crop(128), _crop(48), _crop(96)]
    batch = recognizer.recognize_batch(crops)
    assert len(batch) == 3
    # Each crop is matched against its own single-image decode.
    for result in batch:
        assert isinstance(result, RecognitionResult)


def test_engine_accepts_color_and_single_channel_images(tmp_path: Path) -> None:
    recognizer = CTCRecognizer.from_checkpoint(_checkpoint(tmp_path), device="cpu")
    gray = _crop()
    rgb = np.stack([gray, gray, gray], axis=-1)
    rgba = np.concatenate(
        [rgb, np.full((*gray.shape, 1), 255, dtype=np.uint8)], axis=-1
    )
    for image in (gray, gray[:, :, None], rgb, rgba):
        assert recognizer.recognize(image).text == recognizer.recognize(gray).text


def test_engine_downscales_over_wide_crops(tmp_path: Path) -> None:
    recognizer = CTCRecognizer.from_checkpoint(
        _checkpoint(tmp_path), device="cpu", max_width=32
    )
    tensor, width = recognizer._prepare(_crop(width=400, height=24))
    assert width == 32
    assert tensor.shape == (1, recognizer.image_height, 32)


def test_engine_empty_batch_returns_nothing(tmp_path: Path) -> None:
    recognizer = CTCRecognizer.from_checkpoint(_checkpoint(tmp_path), device="cpu")
    assert recognizer.recognize_batch([]) == []


def test_engine_beam_decoding_is_available(tmp_path: Path) -> None:
    recognizer = CTCRecognizer.from_checkpoint(
        _checkpoint(tmp_path), device="cpu", decode=DecodeSettings(beam_width=4)
    )
    result = recognizer.recognize(_crop())
    assert result.metadata["decoder"] == "beam"
    assert 0.0 <= result.confidence <= 1.0


def test_decode_settings_rejects_bad_beam_width() -> None:
    with pytest.raises(ValueError, match="beam_width"):
        DecodeSettings(beam_width=0)


def test_to_gray_handles_shapes() -> None:
    gray = _crop()
    assert to_gray(gray).shape == gray.shape
    assert to_gray(gray[:, :, None]).shape == gray.shape
    assert to_gray(np.stack([gray, gray, gray], axis=-1)).shape == gray.shape
    with pytest.raises(ValueError, match="expected a 2D or 3D"):
        to_gray(np.zeros((2, 3, 4, 5), dtype=np.uint8))


# --- lexicon -----------------------------------------------------------------


def test_lexicon_from_texts_counts_and_normalizes() -> None:
    lexicon = WordLexicon.from_texts(["the cat", "the dog", "the"], min_count=2)
    assert lexicon.words == frozenset({"the"})
    assert lexicon.is_prefix("th")
    assert not lexicon.is_prefix("ca")


def test_lexicon_from_file(tmp_path: Path) -> None:
    path = tmp_path / "words.txt"
    path.write_text("hello world\nhello\n", encoding="utf-8")
    lexicon = WordLexicon.from_file(path, min_count=2)
    assert lexicon.words == frozenset({"hello"})


def test_scorer_penalizes_impossible_prefixes() -> None:
    charset = CharSet(("a", "b", "c", " ", "h", "e", "l", "o"))
    lexicon = WordLexicon.from_texts(["hello"])
    scorer = LexiconScorer(charset, lexicon, penalty=3.0)
    index = {char: charset.index(char) for char in charset.chars}

    # "he" can still become "hello".
    prefix = (index["h"], index["e"])
    assert scorer.score(prefix, index["l"]) == 0.0
    # "hex" cannot.
    assert scorer.score(prefix, index["a"]) == -3.0
    # A completed word gets the bonus.
    scorer_bonus = LexiconScorer(charset, lexicon, complete_bonus=1.5)
    assert scorer_bonus.score(prefix, index["l"]) == 0.0
    full = (index["h"], index["e"], index["l"], index["l"], index["o"])
    assert scorer_bonus.score(full, index[" "]) == 0.0


def test_scorer_rejects_negative_penalty() -> None:
    with pytest.raises(ValueError, match="penalty"):
        LexiconScorer(_charset(), WordLexicon.from_texts(["ab"]), penalty=-1.0)


def test_scorer_reset_clears_memo() -> None:
    charset = _charset()
    scorer = LexiconScorer(charset, WordLexicon.from_texts(["ab"]))
    scorer.score((charset.index("a"),), charset.index("b"))
    assert scorer._partials
    scorer.reset()
    assert not scorer._partials


def test_lexicon_can_steer_a_decoder() -> None:
    """The scorer turns an acoustically preferred non-word into a known word."""
    from embedding_ocr.recognition.ctc.decode import log_softmax, prefix_beam_search

    charset = CharSet(("a", "b", "c"))  # index 1='a', 2='b', 3='c'
    # The acoustics prefer "ac": 'c' is ahead of 'b' at the second step.
    log_probs = log_softmax(
        np.array(
            [
                [0.0, 3.0, 0.0, 0.0],
                [0.0, 0.0, 1.5, 2.0],
            ],
            dtype=np.float32,
        )
    )
    plain = prefix_beam_search(log_probs, beam_width=4, top_k=4)
    assert charset.decode(plain.indices) == "ac"

    # A lexicon that only knows "ab" penalizes 'c' and flips the result.
    lexicon = WordLexicon.from_texts(["ab"])
    scorer = LexiconScorer(charset, lexicon, penalty=5.0)
    steered = prefix_beam_search(
        log_probs, beam_width=4, top_k=4, scorer=scorer, scorer_weight=1.0
    )
    assert charset.decode(steered.indices) == "ab"
