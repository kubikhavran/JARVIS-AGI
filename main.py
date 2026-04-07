import asyncio
import signal
import sys
from core.config import Config
from core.engine import Engine
from core.logger import get_logger

logger = get_logger("main")

async def main() -> None:
    config = Config()
    engine = Engine(config)

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, engine.stop)

    await engine.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutting down")
        sys.exit(0)
