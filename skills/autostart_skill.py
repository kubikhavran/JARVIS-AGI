from skills.base import BaseSkill
from core.logger import get_logger

logger = get_logger(__name__)

class AutostartSkill(BaseSkill):
    name = "autostart"
    description = "Enable or disable JARVIS starting automatically on Windows login"
    parameters = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["enable", "disable", "status"]},
        },
        "required": ["action"],
    }

    async def execute(self, action: str, **kwargs) -> str:
        from core.autostart import install_autostart, remove_autostart, autostart_status
        if action == "enable":
            return install_autostart()
        elif action == "disable":
            return remove_autostart()
        elif action == "status":
            active = autostart_status()
            return "Autostart is enabled." if active else "Autostart is disabled."
        return "Unknown action."
