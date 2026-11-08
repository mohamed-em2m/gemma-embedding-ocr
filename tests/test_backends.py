"""Unit tests for optional adapters and embedder helpers (no model downloads)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from embedding_ocr.config import AppConfig
from embedding_ocr.recognition.embedding.embedder import (
    _default_rope_parameters,
    _feature_tensor,
    _truncate_and_normalize,
)
from embedding_ocr.recognition.factory import build_recognizer
from embedding_ocr.recognition.oxr import (
    OXRRecognizer,
    _encode_multipart,
    _extract_parse_text,
)
from embedding_ocr.recognition.surya_ocr import _html_to_text


def test_truncate_and_normalize_shape_and_norm() -> None:
    vectors = np.random.default_rng(0).standard_normal((3, 16)).astype(np.float32)
    result = _truncate_and_normalize(vectors, 8)
    assert result.shape == (3, 8)
    norms = np.linalg.norm(result, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-6)


def test_truncate_and_normalize_preserves_full_dim() -> None:
    vectors = np.array([[3.0, 4.0]], dtype=np.float32)
    result = _truncate_and_normalize(vectors, 2)
    assert result.shape == (1, 2)
    np.testing.assert_allclose(result, [[0.6, 0.8]], atol=1e-6)


def test_feature_tensor_unwraps_pooler_output() -> None:
    expected = np.array([[1.0, 2.0]], dtype=np.float32)
    output = SimpleNamespace(pooler_output=expected, last_hidden_state=None)
    np.testing.assert_array_equal(_feature_tensor(output), expected)


def test_feature_tensor_reads_dict_embedding_fields() -> None:
    expected = np.array([[3.0, 4.0]], dtype=np.float32)
    assert _feature_tensor({"image_embeds": expected}) is expected


def test_default_rope_parameters_matches_standard_rope() -> None:
    config = SimpleNamespace(
        rope_parameters={"rope_theta": 10000.0},
        hidden_size=8,
        num_attention_heads=2,
        head_dim=4,
    )
    inv_freq, scaling = _default_rope_parameters(config)
    expected = 1.0 / (10000.0 ** (np.arange(0, 4, 2, dtype=np.float32) / 4))
    np.testing.assert_allclose(inv_freq.numpy(), expected, rtol=1e-6)
    assert scaling == 1.0


def test_oxr_auto_backend_selection() -> None:
    assert (
        OXRRecognizer("https://free-oxr.somark.ai/api/v1")._resolved_backend()
        == "parse"
    )
    assert OXRRecognizer("http://127.0.0.1:8000/v1")._resolved_backend() == "parse"
    assert OXRRecognizer("http://127.0.0.1:8080/v1")._resolved_backend() == "chat"
    assert (
        OXRRecognizer("http://127.0.0.1:8080/v1", backend="parse")._resolved_backend()
        == "parse"
    )


def test_oxr_extract_markdown_response() -> None:
    payload = {"code": 0, "data": {"outputs": {"markdown": "<p>A</p>"}}}
    assert _extract_parse_text(payload) == "<p>A</p>"


def test_oxr_extract_json_response_orders_blocks() -> None:
    payload = {
        "code": 0,
        "data": {
            "outputs": {
                "json": {
                    "pages": [
                        {
                            "blocks": [
                                {"idx": 1, "content": "B"},
                                {"idx": 0, "content": "A"},
                            ]
                        }
                    ]
                }
            }
        },
    }
    assert _extract_parse_text(payload) == "A\nB"


def test_multipart_contains_fields_and_file() -> None:
    body, content_type = _encode_multipart(
        fields=[("api_key", "dummy"), ("output_formats", "markdown")],
        file_field="file",
        filename="crop.png",
        file_bytes=b"png-bytes",
        file_content_type="image/png",
    )
    assert content_type.startswith("multipart/form-data; boundary=")
    assert b'name="api_key"' in body
    assert b"dummy" in body
    assert b'filename="crop.png"' in body
    assert b"png-bytes" in body


def test_surya_html_to_text() -> None:
    assert _html_to_text("<p>Hello <b>world</b></p>") == "Hello world"
    assert _html_to_text("A &amp; B") == "A & B"


def test_build_dtlr_recognizer_forwards_settings(monkeypatch, tmp_path) -> None:
    import embedding_ocr.recognition.dtlr as dtlr_module

    captured: dict[str, object] = {}

    class FakeDTLR:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(dtlr_module, "DTLRRecognizer", FakeDTLR)
    config = AppConfig()
    config.recognition.backend = "dtlr"
    config.recognition.dtlr_root = tmp_path
    config.recognition.dtlr_config = tmp_path / "config.py"
    config.recognition.dtlr_weights = tmp_path / "weights.pth"
    config.recognition.dtlr_charset = tmp_path / "charset.json"
    config.recognition.dtlr_device = "cuda:0"

    recognizer = build_recognizer(config)

    assert isinstance(recognizer, FakeDTLR)
    assert captured == {
        "root": tmp_path,
        "config_path": tmp_path / "config.py",
        "weights": tmp_path / "weights.pth",
        "charset_path": tmp_path / "charset.json",
        "device": "cuda:0",
    }
