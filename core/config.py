from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv
import yaml

load_dotenv()

@dataclass
class STTConfig:
    model: str
    device: str
    compute_type: str

@dataclass
class LLMConfig:
    primary: str
    ollama_model: str
    ollama_url: str
    fallback: str
    timeout_sec: int

@dataclass
class TTSConfig:
    voice_en: str
    voice_cs: str
    rate: str
    volume: str

@dataclass
class MemoryConfig:
    db_path: Path
    chroma_path: Path
    context_history_count: int

@dataclass
class TimeoutsConfig:
    stt_max_recording_sec: int
    llm_response_sec: int
    skill_execute_sec: int

@dataclass
class LanguageConfig:
    primary: str
    secondary: str

class Config:
    def __init__(self, settings_path: str = "settings.yaml") -> None:
        with open(settings_path, "r") as f:
            raw = yaml.safe_load(f)

        self.wake_word: str = raw["wake_word"]
        self.language = LanguageConfig(**raw["language"])
        self.stt = STTConfig(**raw["stt"])
        self.llm = LLMConfig(**raw["llm"])
        self.tts = TTSConfig(**raw["tts"])

        mem = raw["memory"]
        self.memory = MemoryConfig(
            db_path=Path(mem["db_path"]),
            chroma_path=Path(mem["chroma_path"]),
            context_history_count=mem["context_history_count"],
        )
        self.timeouts = TimeoutsConfig(**raw["timeouts"])

        self.gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
        self.porcupine_access_key: str = os.getenv("PORCUPINE_ACCESS_KEY", "")
