import asyncio
from skills.base import BaseSkill
from core.logger import get_logger

logger = get_logger(__name__)

class TimerAlarmSkill(BaseSkill):
    name = "timer_alarm"
    description = "Set a countdown timer"
    parameters = {
        "type": "object",
        "properties": {
            "seconds": {"type": "integer", "description": "Timer duration in seconds"},
            "label": {"type": "string", "description": "Timer label"},
        },
        "required": ["seconds"],
    }

    async def execute(self, seconds: int, label: str = "Timer", **kwargs) -> str:
        async def _fire():
            await asyncio.sleep(seconds)
            logger.info(f"TIMER DONE: {label}")
            print(f"\n⏰ {label} is done!\n")

        asyncio.create_task(_fire())
        mins, secs = divmod(seconds, 60)
        if mins:
            return f"{label} set for {mins} minute{'s' if mins != 1 else ''} and {secs} seconds."
        return f"{label} set for {seconds} seconds."
