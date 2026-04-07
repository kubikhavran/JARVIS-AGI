from __future__ import annotations
import asyncio
import io
import wave
from faster_whisper import WhisperModel
from core.logger import get_logger

logger = get_logger(__name__)

class Transcriber:
    def __init__(
        self,
        model_size: str = "medium",
        device: str = "cuda",
        compute_type: str = "float16",
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._model: WhisperModel | None = None
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        logger.info(f"Loading Whisper {self.model_size} on {self.device}...")
        self._model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type,
        )
        self._loaded = True
        logger.info("Whisper loaded")

    async def transcribe(self, pcm_bytes: bytes, sample_rate: int = 16000) -> str:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._transcribe_sync, pcm_bytes, sample_rate)

    def _transcribe_sync(self, pcm_bytes: bytes, sample_rate: int) -> str:
        self._load()
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm_bytes)
        buf.seek(0)

        segments, info = self._model.transcribe(buf, beam_size=5)
        text = " ".join(seg.text for seg in segments).strip()
        logger.info(f"Transcribed [{info.language}]: {text!r}")
        return text
