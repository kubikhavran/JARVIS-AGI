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
from memory.vector_store import VectorStore
from memory.user_facts import UserFacts
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
        self._vector_store = VectorStore(chroma_path=config.memory.chroma_path)
        self._user_facts = UserFacts(db_path=config.memory.db_path)
        self._exchange_count = 0
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
        # Augment with semantic memories if any exist
        semantic = self._vector_store.search(text, top_k=3)
        if semantic:
            context = [{"role": "system", "content": "Relevant memories:\n" + "\n".join(semantic)}] + context

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
        # Store exchange in vector memory
        self._exchange_count += 1
        import uuid as _uuid
        exchange_text = f"User: {text}\nAssistant: {response}"
        self._vector_store.add(str(_uuid.uuid4()), exchange_text, {"session": self._session_id})
        # Background user fact extraction every 5 exchanges
        if self._exchange_count % 5 == 0:
            asyncio.create_task(self._extract_user_facts(exchange_text))
        await self._speaker.speak(response)

    async def _extract_user_facts(self, context_text: str) -> None:
        from brain.ollama_client import OllamaClient
        import json
        client = OllamaClient(base_url=self._cfg.llm.ollama_url, model=self._cfg.llm.ollama_model)
        prompt = (
            "Extract any personal facts about the user from this conversation. "
            "Return a JSON array of {\"key\": \"...\", \"value\": \"...\"} objects, or empty array [].\n"
            f"Conversation:\n{context_text}"
        )
        try:
            raw = await client.chat([{"role": "user", "content": prompt}], json_mode=True)
            facts = json.loads(raw) if isinstance(raw, str) else raw
            if isinstance(facts, list):
                for f in facts:
                    if "key" in f and "value" in f:
                        self._user_facts.upsert(f["key"], f["value"])
        except Exception as e:
            logger.debug(f"User fact extraction failed (non-critical): {e}")

    def stop(self) -> None:
        self._running = False
