from __future__ import annotations
import asyncio
import uuid
from core.config import Config
from core.logger import get_logger
from stt.audio_capture import AudioCapture
from stt.hotword import HotwordDetector
from stt.transcriber import Transcriber
from brain.router import IntentRouter
from tts.speaker import Speaker
from memory.chat_history import ChatHistory
from skills.registry import SkillRegistry

logger = get_logger(__name__)

class Engine:
    def __init__(self, config: Config) -> None:
        self._cfg = config
        self._session_id = str(uuid.uuid4())

        self._audio = AudioCapture()
        self._hotword = HotwordDetector(
            access_key=config.porcupine_access_key,
            keyword=config.wake_word,
        )
        self._transcriber = Transcriber(
            model_size=config.stt.model,
            device=config.stt.device,
            compute_type=config.stt.compute_type,
        )
        self._speaker = Speaker(
            voice_en=config.tts.voice_en,
            voice_cs=config.tts.voice_cs,
            rate=config.tts.rate,
            volume=config.tts.volume,
        )
        self._history = ChatHistory(db_path=config.memory.db_path)
        self._registry = SkillRegistry()
        self._registry.auto_discover()
        self._router = IntentRouter(
            registry=self._registry,
            ollama_url=config.llm.ollama_url,
            model=config.llm.ollama_model,
            gemini_api_key=config.gemini_api_key,
            timeout=config.llm.timeout_sec,
        )

        self._wake_event = asyncio.Event()
        self._running = False

    def _on_wake(self) -> None:
        loop = asyncio.get_event_loop()
        self._speaker.interrupt(loop=loop)
        loop.call_soon_threadsafe(self._wake_event.set)

    async def run(self) -> None:
        self._running = True
        self._audio.start()
        self._hotword.on_detect(self._on_wake)
        self._hotword.start()
        logger.info("J.A.R.V.I.S. is online. Say 'Jarvis' to activate.")

        try:
            while self._running:
                await self._wake_event.wait()
                self._wake_event.clear()
                await self._handle_command()
        finally:
            self._audio.stop()
            self._hotword.stop()

    async def _handle_command(self) -> None:
        logger.info("Listening for command...")
        await self._speaker.speak("Yes?")

        pcm = await self._audio.record_until_silence(
            max_duration_sec=self._cfg.timeouts.stt_max_recording_sec
        )
        if not pcm:
            return

        text = await self._transcriber.transcribe(pcm)
        if not text.strip():
            return

        logger.info(f"User: {text}")
        self._history.log("user", text, self._session_id)

        context = self._history.recent(
            count=self._cfg.memory.context_history_count,
            session_id=self._session_id,
        )

        try:
            skill_name, params = await self._router.route(text, context=context[:-1])
            skill = self._registry.get(skill_name)
            response = await asyncio.wait_for(
                skill.execute(**params),
                timeout=self._cfg.timeouts.skill_execute_sec,
            )
        except Exception as e:
            logger.error(f"Skill execution failed: {e}")
            response = "I couldn't do that, sorry."

        logger.info(f"Jarvis: {response}")
        self._history.log("assistant", response, self._session_id)
        await self._speaker.speak(response)

    def stop(self) -> None:
        self._running = False
