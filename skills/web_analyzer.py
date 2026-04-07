from playwright.async_api import async_playwright
from brain.ollama_client import OllamaClient
from skills.base import BaseSkill
from core.logger import get_logger

logger = get_logger(__name__)

class WebAnalyzerSkill(BaseSkill):
    name = "web_analyzer"
    description = "Fetch a webpage and summarize its content"
    parameters = {
        "type": "object",
        "properties": {
            "url": {"type": "string"},
            "question": {"type": "string", "description": "Optional question to answer from page content"},
        },
        "required": ["url"],
    }

    def __init__(self, ollama_url: str = "http://localhost:11434", model: str = "qwen3:8b") -> None:
        self._client = OllamaClient(base_url=ollama_url, model=model)

    async def execute(self, url: str, question: str = "", **kwargs) -> str:
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()
                await page.goto(url, timeout=15000)
                text = await page.inner_text("body")
                await browser.close()

            text = text[:4000]
            prompt = question if question else "Summarize this webpage content in 3 sentences."
            messages = [
                {"role": "system", "content": "You are a helpful assistant that analyzes web pages."},
                {"role": "user", "content": f"{prompt}\n\nPage content:\n{text}"},
            ]
            return await self._client.chat(messages)
        except Exception as e:
            logger.error(f"Web analysis failed: {e}")
            return f"I couldn't analyze that page: {e}"
