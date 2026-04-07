import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from core.config import Config
from core.engine import Engine

@pytest.mark.asyncio
async def test_engine_handle_command_full_flow(tmp_path, httpx_mock):
    """Integration: mock mic + LLM → verify skill called + history logged."""
    fake_pcm = b"\x00" * 3200

    mock_vector_store = MagicMock()
    mock_vector_store.search.return_value = []
    mock_vector_store.add.return_value = None

    with patch("stt.audio_capture.AudioCapture.record_until_silence", new_callable=AsyncMock, return_value=fake_pcm), \
         patch("stt.transcriber.Transcriber.transcribe", new_callable=AsyncMock, return_value="what is the time"), \
         patch("stt.audio_capture.AudioCapture.start"), \
         patch("stt.audio_capture.AudioCapture.stop"), \
         patch("stt.hotword.HotwordDetector.start"), \
         patch("stt.hotword.HotwordDetector.stop"), \
         patch("tts.speaker.Speaker.speak", new_callable=AsyncMock) as mock_speak, \
         patch("core.engine.VectorStore", return_value=mock_vector_store):

        # Router picks general_chat
        httpx_mock.add_response(
            url="http://localhost:11434/api/chat",
            json={"message": {"content": '{"skill": "general_chat", "params": {"message": "what is the time"}}'}},
        )
        # general_chat LLM response
        httpx_mock.add_response(
            url="http://localhost:11434/api/chat",
            json={"message": {"content": "It's time to get things done."}},
        )

        cfg = Config(settings_path="settings.yaml")
        cfg.memory.db_path = tmp_path / "test.db"
        cfg.memory.chroma_path = tmp_path / "chroma"

        engine = Engine(cfg)
        await engine._handle_command()

        # TTS was called
        assert mock_speak.call_count >= 1

        # History was written
        history = engine._history.recent(count=5)
        roles = [h["role"] for h in history]
        assert "user" in roles
        assert "assistant" in roles
