"""Settings loading and env override behaviour (the laptop/CI duality)."""

from __future__ import annotations

from pathlib import Path

import pytest

from watari.config import Settings


def test_defaults_point_at_ollama() -> None:
    s = Settings()
    assert s.llm_base_url.endswith("/v1")
    assert s.chat_model == "qwen3.5:4b"


def test_env_overrides_model_role(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WATARI_CHAT_MODEL", "qwen2.5:0.5b")
    monkeypatch.setenv("WATARI_JUDGE_MODEL", "qwen2.5:0.5b")
    s = Settings()
    assert s.chat_model == "qwen2.5:0.5b"
    assert s.judge_model == "qwen2.5:0.5b"


def test_db_path_is_under_data_dir(tmp_path: Path) -> None:
    s = Settings(data_dir=tmp_path)
    assert s.db_path == tmp_path / "watari.db"


def test_binds_loopback_by_default() -> None:
    assert Settings().host == "127.0.0.1"
