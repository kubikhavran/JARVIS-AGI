import asyncio
import subprocess
from skills.base import BaseSkill
from core.logger import get_logger

logger = get_logger(__name__)

class MusicPlayerSkill(BaseSkill):
    name = "music_player"
    description = "Play, pause, or stop music from YouTube"
    parameters = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["play", "stop"]},
            "query": {"type": "string", "description": "Song or artist to play"},
        },
        "required": ["action"],
    }

    def __init__(self) -> None:
        self._process: subprocess.Popen | None = None

    async def execute(self, action: str, query: str = "", **kwargs) -> str:
        if action == "stop":
            if self._process:
                self._process.terminate()
                self._process = None
            return "Music stopped."

        if not query:
            return "Please specify what to play."

        try:
            loop = asyncio.get_event_loop()
            url = await loop.run_in_executor(None, self._get_audio_url, query)
            self._process = subprocess.Popen(
                ["ffplay", "-nodisp", "-autoexit", url],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return f"Playing {query}."
        except Exception as e:
            logger.error(f"Music playback failed: {e}")
            return f"Couldn't play {query}: {e}"

    def _get_audio_url(self, query: str) -> str:
        import yt_dlp
        ydl_opts = {"format": "bestaudio", "quiet": True, "noplaylist": True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(f"ytsearch1:{query}", download=False)
            return info["entries"][0]["url"]
