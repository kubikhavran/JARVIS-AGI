from __future__ import annotations
import asyncio
import threading
from collections import deque
from typing import AsyncIterator
import numpy as np
import sounddevice as sd
import webrtcvad
from core.logger import get_logger

logger = get_logger(__name__)

class AudioCapture:
    def __init__(
        self,
        sample_rate: int = 16000,
        frame_duration_ms: int = 30,
        channels: int = 1,
        vad_aggressiveness: int = 2,
    ) -> None:
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.channels = channels
        self.frame_size = int(sample_rate * frame_duration_ms / 1000)
        self.frame_size_bytes = self.frame_size * 2  # int16 = 2 bytes
        self._vad = webrtcvad.Vad(vad_aggressiveness)
        self._buffer: deque[bytes] = deque()
        self._lock = threading.Lock()
        self._stream: sd.RawInputStream | None = None

    def is_speech(self, frame: bytes) -> bool:
        try:
            return self._vad.is_speech(frame, self.sample_rate)
        except Exception:
            return False

    def start(self) -> None:
        def callback(indata, frames, time, status):
            with self._lock:
                self._buffer.append(bytes(indata))

        self._stream = sd.RawInputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="int16",
            blocksize=self.frame_size,
            callback=callback,
        )
        self._stream.start()
        logger.info("Audio capture started")

    def stop(self) -> None:
        if self._stream:
            self._stream.stop()
            self._stream.close()
            logger.info("Audio capture stopped")

    def read_frame(self) -> bytes | None:
        with self._lock:
            return self._buffer.popleft() if self._buffer else None

    async def record_until_silence(
        self,
        silence_threshold_sec: float = 1.5,
        max_duration_sec: float = 30.0,
    ) -> bytes:
        """Record audio until silence is detected. Returns raw PCM bytes."""
        frames: list[bytes] = []
        silence_frames = 0
        max_silence = int(silence_threshold_sec * 1000 / self.frame_duration_ms)
        max_frames = int(max_duration_sec * 1000 / self.frame_duration_ms)

        while len(frames) < max_frames:
            frame = self.read_frame()
            if frame is None:
                await asyncio.sleep(0.005)
                continue
            frames.append(frame)
            if self.is_speech(frame):
                silence_frames = 0
            else:
                silence_frames += 1
                if silence_frames >= max_silence and len(frames) > 10:
                    break

        return b"".join(frames)
