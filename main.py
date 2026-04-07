import os
import sys

# Inject CUDA DLL paths before any CUDA-dependent imports
def _inject_cuda_paths() -> None:
    from dotenv import load_dotenv
    load_dotenv()
    cuda_dll = os.getenv("CUDA_DLL_PATH", "")
    if cuda_dll and os.path.isdir(cuda_dll):
        os.add_dll_directory(cuda_dll)
    # Also add Ollama's bundled CUDA as fallback
    ollama_cuda = os.path.expandvars(
        r"%LOCALAPPDATA%\Programs\Ollama\lib\ollama\cuda_v12"
    )
    if os.path.isdir(ollama_cuda):
        os.add_dll_directory(ollama_cuda)

_inject_cuda_paths()

import asyncio
import signal
import threading
import argparse
from core.config import Config
from core.engine import Engine
from core.logger import get_logger

logger = get_logger("main")

def parse_args():
    parser = argparse.ArgumentParser(description="J.A.R.V.I.S AGI")
    parser.add_argument("--no-tray", action="store_true", help="Disable system tray icon")
    parser.add_argument("--install-autostart", action="store_true", help="Register Windows autostart task and exit")
    parser.add_argument("--remove-autostart", action="store_true", help="Remove Windows autostart task and exit")
    return parser.parse_args()

async def main() -> None:
    args = parse_args()

    if args.install_autostart:
        from core.autostart import install_autostart
        print(install_autostart())
        return

    if args.remove_autostart:
        from core.autostart import remove_autostart
        print(remove_autostart())
        return

    config = Config()
    engine = Engine(config)

    # System tray in background thread
    if not args.no_tray:
        tray_thread = threading.Thread(
            target=lambda: __import__("core.tray", fromlist=["run_tray"]).run_tray(engine.stop),
            daemon=True,
        )
        tray_thread.start()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, engine.stop)
        except NotImplementedError:
            pass  # Windows doesn't support add_signal_handler for all signals

    await engine.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutting down")
        sys.exit(0)
