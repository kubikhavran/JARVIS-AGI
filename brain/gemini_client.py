from __future__ import annotations
import asyncio
from core.logger import get_logger

logger = get_logger(__name__)

class GeminiClient:
    def __init__(self, api_key: str, model: str = "gemini-2.0-flash") -> None:
        self.api_key = api_key
        self.model = model

    async def chat(self, messages: list[dict]) -> str:
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not set")
        try:
            from google import genai
            client = genai.Client(api_key=self.api_key)
            # Convert messages to single prompt (google-genai simple path)
            contents = [
                {"role": "user" if m["role"] == "user" else "model", "parts": [{"text": m["content"]}]}
                for m in messages if m["role"] != "system"
            ]
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: client.models.generate_content(model=self.model, contents=contents),
            )
            return response.text
        except ImportError:
            # Fallback to legacy google-generativeai if google-genai not installed
            import google.generativeai as genai_legacy
            genai_legacy.configure(api_key=self.api_key)
            model_obj = genai_legacy.GenerativeModel(self.model)
            history = []
            for msg in messages[:-1]:
                role = "user" if msg["role"] == "user" else "model"
                history.append({"role": role, "parts": [msg["content"]]})
            chat = model_obj.start_chat(history=history)
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, chat.send_message, messages[-1]["content"])
            return response.text
