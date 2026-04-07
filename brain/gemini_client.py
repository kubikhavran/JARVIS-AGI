from __future__ import annotations
import asyncio
import google.generativeai as genai
from core.logger import get_logger

logger = get_logger(__name__)

class GeminiClient:
    def __init__(self, api_key: str, model: str = "gemini-1.5-flash") -> None:
        self.api_key = api_key
        self.model = model

    async def chat(self, messages: list[dict]) -> str:
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not set")
        genai.configure(api_key=self.api_key)
        model = genai.GenerativeModel(self.model)

        history = []
        for msg in messages[:-1]:
            role = "user" if msg["role"] == "user" else "model"
            history.append({"role": role, "parts": [msg["content"]]})

        chat = model.start_chat(history=history)
        last = messages[-1]["content"]

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, chat.send_message, last)
        return response.text
