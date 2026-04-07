from __future__ import annotations
import asyncio
import io
import threading
import re
import edge_tts
import sounddevice as sd
import soundfile as sf
from core.logger import get_logger

logger = get_logger(__name__)

_CS_PATTERN = re.compile(r"[áčďéěíňóřšťúůýžÁČĎÉĚÍŇÓŘŠŤÚŮÝŽ]")

class Speaker:
    def __init__(
        self,
        voice_en: str = "en-US-JennyNeural",
        voice_cs: str = "cs-CZ-VlastaNeural",
        rate: str = "+0%",
        volume: str = "+0%",
    ) -> None:
        self.voice_en = voice_en
        self.voice_cs = voice_cs
        self.rate = rate
        self.volume = volume
        self._cancel_event = asyncio.Event()
        self.is_speaking = False

    def _detect_language(self, text: str) -> str:
        cs_chars = len(_CS_PATTERN.findall(text))
        return "cs" if cs_chars >= 2 else "en"

    def interrupt(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        """Thread-safe: can be called from non-asyncio threads (e.g. hotword daemon)."""
        if loop and loop.is_running():
            loop.call_soon_threadsafe(self._cancel_event.set)
        else:
            self._cancel_event.set()

    async def speak(self, text: str) -> None:
        self._cancel_event.clear()
        voice = self.voice_cs if self._detect_language(text) == "cs" else self.voice_en

        communicate = edge_tts.Communicate(text, voice, rate=self.rate, volume=self.volume)
        audio_buf = io.BytesIO()

        async for chunk in communicate.stream():
            if self._cancel_event.is_set():
                logger.info("TTS interrupted before playback")
                return
            if chunk["type"] == "audio":
                audio_buf.write(chunk["data"])

        audio_buf.seek(0)
        if self._cancel_event.is_set():
            return

        await self._play(audio_buf)

    async def _play(self, audio_buf: io.BytesIO) -> None:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._play_sync, audio_buf)

    def _play_sync(self, audio_buf: io.BytesIO) -> None:
        self.is_speaking = True
        try:
            data, samplerate = sf.read(audio_buf, dtype="float32")
            sd.play(data, samplerate)
            duration = len(data) / samplerate
            elapsed = 0.0
            chunk = 0.05
            while elapsed < duration:
                if self._cancel_event.is_set():
                    sd.stop()
                    return
                threading.Event().wait(chunk)
                elapsed += chunk
            sd.wait()
        finally:
            self.is_speaking = False
