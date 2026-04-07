import httpx
from skills.base import BaseSkill
from core.logger import get_logger

logger = get_logger(__name__)

class WeatherSkill(BaseSkill):
    name = "weather"
    description = "Get current weather for a location"
    parameters = {
        "type": "object",
        "properties": {
            "location": {"type": "string", "description": "City name or location"},
        },
        "required": ["location"],
    }

    async def execute(self, location: str, **kwargs) -> str:
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.get(f"https://wttr.in/{location}?format=3")
                r.raise_for_status()
                return r.text.strip()
        except Exception as e:
            logger.error(f"Weather failed: {e}")
            return f"Couldn't get weather for {location}."
