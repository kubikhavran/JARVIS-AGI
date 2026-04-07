from skills.base import BaseSkill
from core.logger import get_logger

logger = get_logger(__name__)

class GeneralChatSkill(BaseSkill):
    name = "general_chat"
    description = "Have a general conversation or answer questions"
    parameters = {
        "type": "object",
        "properties": {"message": {"type": "string"}},
        "required": ["message"],
    }

    def __init__(self, ollama_url: str = "http://localhost:11434", model: str = "qwen3:8b") -> None:
        self.ollama_url = ollama_url
        self.model = model

    async def execute(self, message: str, **kwargs) -> str:
        from brain.ollama_client import OllamaClient
        client = OllamaClient(base_url=self.ollama_url, model=self.model)
        messages = [
            {"role": "system", "content": "You are J.A.R.V.I.S., a helpful AI assistant."},
            {"role": "user", "content": message},
        ]
        try:
            return await client.chat(messages)
        except Exception as e:
            logger.error(f"general_chat failed: {e}")
            return "I'm having trouble thinking right now."
