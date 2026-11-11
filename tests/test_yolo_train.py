"""Tests for the YOLO word-training CLI helpers."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from scripts.train_yolo_words import resolve_ultralytics_device


def _fake_torch(*, available: bool, count: int) -> SimpleNamespace:
    return SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: available,
            device_count=lambda: count,
        )
    )


def test_resolve_auto_uses_all_visible_gpus(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(available=True, count=2))
    assert resolve_ultralytics_device("auto") == "0,1"


def test_resolve_auto_falls_back_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(available=False, count=0))
    assert resolve_ultralytics_device("auto") == "cpu"
    assert resolve_ultralytics_device("cuda") == "cpu"


def test_resolve_explicit_device_is_unchanged() -> None:
    assert resolve_ultralytics_device(None) is None
    assert resolve_ultralytics_device("cpu") == "cpu"
    assert resolve_ultralytics_device("0,1") == "0,1"
