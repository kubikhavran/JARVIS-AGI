from __future__ import annotations
import threading
from typing import Callable
import pvporcupine
import sounddevice as sd
import numpy as np
from core.logger import get_logger

logger = get_logger(__name__)

class HotwordDetector:
    def __init__(self, access_key: str, keyword: str = "jarvis") -> None:
        self.access_key = access_key
        self.keyword = keyword
        self._porcupine: pvporcupine.Porcupine | None = None
        self._thread: threading.Thread | None = None
        self._running = False
        self._callback: Callable | None = None

    def on_detect(self, callback: Callable) -> None:
        self._callback = callback

    def start(self) -> None:
        if not self.access_key:
            logger.warning("No Porcupine access key — using keyboard fallback mode")
            self._start_keyboard_fallback()
            return

        self._porcupine = pvporcupine.create(
            access_key=self.access_key,
            keywords=[self.keyword],
        )
        self._running = True
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()
        logger.info(f"Hotword detector started: listening for '{self.keyword}'")

    def _listen_loop(self) -> None:
        frame_length = self._porcupine.frame_length
        with sd.RawInputStream(
            samplerate=self._porcupine.sample_rate,
            channels=1,
            dtype="int16",
            blocksize=frame_length,
        ) as stream:
            while self._running:
                pcm, _ = stream.read(frame_length)
                pcm_array = np.frombuffer(pcm, dtype=np.int16)
                result = self._porcupine.process(pcm_array)
                if result >= 0 and self._callback:
                    logger.info("Wake word detected!")
                    self._callback()

    def _start_keyboard_fallback(self) -> None:
        def wait_for_enter():
            while True:
                input("\n[FALLBACK] Press Enter to activate Jarvis...\n")
                if self._callback:
                    self._callback()
        self._thread = threading.Thread(target=wait_for_enter, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._porcupine:
            self._porcupine.delete()
