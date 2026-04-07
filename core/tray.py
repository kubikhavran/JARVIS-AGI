from __future__ import annotations
import asyncio
import threading
import sys
from pathlib import Path
from core.logger import get_logger

logger = get_logger(__name__)

def run_tray(engine_stop_fn: callable) -> None:
    """Run system tray icon in a background thread. Requires pystray + Pillow."""
    try:
        import pystray
        from PIL import Image, ImageDraw

        def _make_icon() -> Image.Image:
            img = Image.new("RGB", (64, 64), color=(0, 20, 0))
            draw = ImageDraw.Draw(img)
            draw.ellipse([8, 8, 56, 56], fill=(0, 200, 100))
            draw.text((20, 22), "J", fill=(0, 0, 0))
            return img

        def _on_quit(icon, item):
            logger.info("Tray: quit requested")
            icon.stop()
            engine_stop_fn()

        def _on_status(icon, item):
            pass  # placeholder

        menu = pystray.Menu(
            pystray.MenuItem("J.A.R.V.I.S AGI", _on_status, enabled=False),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Quit", _on_quit),
        )
        icon = pystray.Icon("jarvis", _make_icon(), "J.A.R.V.I.S AGI", menu)
        icon.run()
    except ImportError:
        logger.warning("pystray not installed — no system tray icon")
    except Exception as e:
        logger.warning(f"Tray failed: {e}")
