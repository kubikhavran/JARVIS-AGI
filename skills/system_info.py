import os
import psutil
from skills.base import BaseSkill

class SystemInfoSkill(BaseSkill):
    name = "system_info"
    description = "Get CPU, RAM, and disk usage stats"
    parameters = {"type": "object", "properties": {}}

    async def execute(self, **kwargs) -> str:
        cpu = psutil.cpu_percent(interval=1)
        ram = psutil.virtual_memory()
        root = os.path.splitdrive(os.getcwd())[0] + "/" or "/"
        disk = psutil.disk_usage(root)
        return (
            f"CPU: {cpu}% | "
            f"RAM: {ram.percent}% ({ram.used // 1024**3}GB / {ram.total // 1024**3}GB) | "
            f"Disk: {disk.percent}% used"
        )
