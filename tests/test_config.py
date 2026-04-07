import pytest
from pathlib import Path
from core.config import Config

def test_config_loads_settings():
    cfg = Config()
    assert cfg.wake_word == "jarvis"
    assert cfg.stt.model == "medium"
    assert cfg.llm.ollama_model == "llama3.1:8b"
    assert cfg.memory.context_history_count == 5

def test_config_env_override(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-123")
    cfg = Config()
    assert cfg.gemini_api_key == "test-key-123"
