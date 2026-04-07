import subprocess
import psutil
from skills.base import BaseSkill
from core.logger import get_logger

logger = get_logger(__name__)

class AppLauncherSkill(BaseSkill):
    name = "app_launcher"
    description = "Open or close a Windows application by name"
    parameters = {
        "type": "object",
        "properties": {
            "app_name": {"type": "string", "description": "Application name, e.g. 'notepad', 'chrome'"},
            "action": {"type": "string", "enum": ["open", "close"], "default": "open"},
        },
        "required": ["app_name"],
    }

    _APP_MAP = {
        "notepad": "notepad.exe",
        "calculator": "calc.exe",
        "chrome": "chrome.exe",
        "firefox": "firefox.exe",
        "explorer": "explorer.exe",
        "spotify": "spotify.exe",
        "vscode": "code",
        "terminal": "wt.exe",
    }

    async def execute(self, app_name: str, action: str = "open", **kwargs) -> str:
        exe = self._APP_MAP.get(app_name.lower(), f"{app_name}.exe")
        if action == "open":
            try:
                subprocess.Popen([exe], shell=True)
                return f"Opening {app_name}."
            except Exception as e:
                logger.error(f"Failed to open {app_name}: {e}")
                return f"I couldn't open {app_name}."
        else:
            for proc in psutil.process_iter(["name"]):
                if app_name.lower() in proc.info["name"].lower():
                    proc.kill()
                    return f"Closed {app_name}."
            return f"{app_name} doesn't seem to be running."
