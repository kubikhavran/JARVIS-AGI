from __future__ import annotations
import json
from brain.ollama_client import OllamaClient
from brain.gemini_client import GeminiClient
from brain.prompts import build_router_prompt
from skills.registry import SkillRegistry
from core.logger import get_logger

logger = get_logger(__name__)

class IntentRouter:
    def __init__(
        self,
        registry: SkillRegistry,
        ollama_url: str,
        model: str,
        gemini_api_key: str = "",
        timeout: int = 30,
    ) -> None:
        self._registry = registry
        self._ollama = OllamaClient(base_url=ollama_url, model=model, timeout=timeout)
        self._gemini = GeminiClient(api_key=gemini_api_key) if gemini_api_key else None

    async def _call_llm(self, messages: list[dict]) -> str:
        try:
            return await self._ollama.chat(messages, json_mode=True)
        except Exception as e:
            logger.warning(f"Ollama failed: {e}. Trying Gemini...")
            if self._gemini:
                return await self._gemini.chat(messages)
            raise

    async def route(
        self, user_text: str, context: list[dict]
    ) -> tuple[str, dict]:
        system_prompt = build_router_prompt(self._registry.function_definitions())
        messages = [
            {"role": "system", "content": system_prompt},
            *context,
            {"role": "user", "content": user_text},
        ]

        raw = await self._call_llm(messages)
        result = self._parse(raw, user_text)
        if result is None:
            # Retry once with explicit format reminder
            retry_messages = messages + [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": 'Respond with ONLY JSON: {"skill": "...", "params": {...}}'},
            ]
            raw2 = await self._call_llm(retry_messages)
            result = self._parse(raw2, user_text)

        return result or ("general_chat", {"message": user_text})

    def _parse(self, raw: str, user_text: str) -> tuple[str, dict] | None:
        try:
            data = json.loads(raw)
            skill_name = data.get("skill", "general_chat")
            params = data.get("params", {})
            if skill_name not in self._registry.skill_names:
                logger.warning(f"Unknown skill '{skill_name}', falling back to general_chat")
                skill_name = "general_chat"
                params = {"message": user_text}
            return skill_name, params
        except json.JSONDecodeError:
            logger.warning(f"Router parse failed on: {raw!r}. Will retry.")
            return None
