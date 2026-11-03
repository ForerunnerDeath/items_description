from pathlib import Path

import pytest

from app.config import Settings


def test_llm_settings_read_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("LLM_MAX_TOKENS", "4096")
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "12.5")

    settings = Settings()

    assert settings.model == "test-model"
    assert settings.api_key == "test-key"
    assert settings.base_url == "http://localhost:11434/v1"
    assert settings.max_tokens == 4096
    assert settings.timeout_seconds == 12.5
