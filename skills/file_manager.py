import subprocess
from pathlib import Path
from skills.base import BaseSkill
from core.logger import get_logger

logger = get_logger(__name__)

class FileManagerSkill(BaseSkill):
    name = "file_manager"
    description = "List, open, or get info about files and folders"
    parameters = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["list", "open", "info"]},
            "path": {"type": "string", "description": "File or directory path"},
        },
        "required": ["action", "path"],
    }

    async def execute(self, action: str, path: str, **kwargs) -> str:
        p = Path(path).expanduser()
        if action == "list":
            if not p.exists():
                return f"Path not found: {path}"
            items = list(p.iterdir())[:20]
            return "\n".join(f"{'[DIR]' if i.is_dir() else '[FILE]'} {i.name}" for i in items)
        elif action == "open":
            subprocess.Popen(["explorer", str(p)], shell=True)
            return f"Opening {path}."
        elif action == "info":
            if not p.exists():
                return f"Not found: {path}"
            stat = p.stat()
            return f"{p.name}: {'directory' if p.is_dir() else 'file'}, {stat.st_size} bytes"
        return "Unknown action."
