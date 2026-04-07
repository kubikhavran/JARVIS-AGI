import asyncio
import pyautogui
from datetime import datetime
from skills.base import BaseSkill
from core.logger import get_logger

logger = get_logger(__name__)

class ScreenshotSkill(BaseSkill):
    name = "screenshot"
    description = "Take a screenshot of the current screen"
    parameters = {
        "type": "object",
        "properties": {
            "save_path": {"type": "string", "description": "Optional file path to save screenshot"},
        },
    }

    async def execute(self, save_path: str = "", **kwargs) -> str:
        loop = asyncio.get_event_loop()
        path = save_path or f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        await loop.run_in_executor(None, lambda: pyautogui.screenshot(path))
        return f"Screenshot saved to {path}."
