from __future__ import annotations
import os
import sys
import subprocess
from pathlib import Path
from core.logger import get_logger

logger = get_logger(__name__)

TASK_NAME = "JarvisAGI"

def _python_exe() -> str:
    return sys.executable

def _main_script() -> str:
    return str(Path(__file__).parent.parent / "main.py")

def install_autostart() -> str:
    """Register JARVIS as a Windows Task Scheduler task on login."""
    python = _python_exe()
    script = _main_script()
    cmd = [
        "schtasks", "/create", "/tn", TASK_NAME,
        "/tr", f'"{python}" "{script}"',
        "/sc", "onlogon",
        "/rl", "highest",
        "/f",  # overwrite if exists
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            logger.info(f"Autostart registered: {TASK_NAME}")
            return f"Autostart enabled. JARVIS will start on login."
        else:
            logger.error(f"schtasks failed: {result.stderr}")
            return f"Failed to register autostart: {result.stderr.strip()}"
    except Exception as e:
        return f"Autostart setup failed: {e}"

def remove_autostart() -> str:
    """Remove the Task Scheduler entry."""
    cmd = ["schtasks", "/delete", "/tn", TASK_NAME, "/f"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            return "Autostart disabled."
        return f"Could not remove autostart: {result.stderr.strip()}"
    except Exception as e:
        return f"Failed: {e}"

def autostart_status() -> bool:
    """Returns True if the task exists."""
    result = subprocess.run(
        ["schtasks", "/query", "/tn", TASK_NAME],
        capture_output=True, text=True
    )
    return result.returncode == 0
