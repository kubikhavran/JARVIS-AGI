# J.A.R.V.I.S AGI Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local-first, Tony Stark-style AI voice assistant for Windows that listens for a wake word, understands EN/CZ commands, routes intent via LLM, executes desktop/web skills, and speaks responses.

**Architecture:** Event-driven state machine — mic → VAD → hotword → STT → LLM router → skill execute → TTS. SQLite stores chat history; ChromaDB adds semantic memory in M7. All models run locally (Ollama + CUDA).

**Tech Stack:** Python 3.11, faster-whisper (CUDA), pvporcupine, Ollama (qwen3:8b), edge-tts, SQLite, ChromaDB, sentence-transformers, playwright, rich, pyautogui, yt-dlp

**Spec:** `docs/superpowers/specs/2026-04-06-jarvis-agi-design.md`

---

## Chunk 1: M1 — Project Scaffolding

### Task 1: Repository Init + Core Config Files

**Files:**
- Create: `.gitignore`
- Create: `.env.example`
- Create: `pyproject.toml`
- Create: `requirements.txt`
- Create: `settings.yaml`

- [ ] **Step 1: Create `.gitignore`**

```
.env
data/
__pycache__/
*.pyc
*.pyo
.pytest_cache/
*.egg-info/
dist/
build/
.venv/
venv/
*.log
```

- [ ] **Step 2: Create `.env.example`**

```
GEMINI_API_KEY=your_gemini_api_key_here
PORCUPINE_ACCESS_KEY=your_porcupine_access_key_here
OPENWEATHERMAP_API_KEY=optional_openweathermap_key
```

- [ ] **Step 3: Create `settings.yaml`**

```yaml
wake_word: jarvis
language:
  primary: en
  secondary: cs
stt:
  model: medium
  device: cuda
  compute_type: float16
llm:
  primary: ollama
  ollama_model: qwen3:8b
  ollama_url: http://localhost:11434
  fallback: gemini
  timeout_sec: 30
tts:
  voice_en: en-US-JennyNeural
  voice_cs: cs-CZ-VlastaNeural
  rate: +0%
  volume: +0%
memory:
  db_path: ./data/jarvis.db
  chroma_path: ./data/chroma
  context_history_count: 5
timeouts:
  stt_max_recording_sec: 30
  llm_response_sec: 30
  skill_execute_sec: 60
```

- [ ] **Step 4: Create `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "jarvis-agi"
version = "0.1.0"
requires-python = ">=3.11"
description = "Local-first AI voice assistant"

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
```

- [ ] **Step 5: Create `requirements.txt`**

```
# Core
pyyaml>=6.0
python-dotenv>=1.0
rich>=13.0

# STT
faster-whisper>=1.0
pvporcupine>=3.0
webrtcvad>=2.0.10
pyaudio>=0.2.13
sounddevice>=0.4.6
numpy>=1.24

# LLM
httpx>=0.27
google-generativeai>=0.7

# TTS
edge-tts>=6.1

# Memory
chromadb>=0.5
sentence-transformers>=3.0

# Skills
playwright>=1.40
pyautogui>=0.9.54
psutil>=5.9
duckduckgo-search>=6.0
yt-dlp>=2024.1
pillow>=10.0
pytesseract>=0.3.10

# Dev
pytest>=8.0
pytest-asyncio>=0.23
pytest-httpx>=0.21
python-json-logger>=2.0
```

- [ ] **Step 6: Git init and first commit**

```bash
cd "f:/APPS/J.A.R.V.I.S AGI"
git init
git add .gitignore .env.example pyproject.toml requirements.txt settings.yaml
git commit -m "M1: add project config files"
```

---

### Task 2: Core Infrastructure (`core/`)

**Files:**
- Create: `core/__init__.py`
- Create: `core/config.py`
- Create: `core/logger.py`
- Create: `tests/__init__.py`

- [ ] **Step 1: Create `core/__init__.py`** (empty)

- [ ] **Step 2: Write failing test for config loading**

```python
# tests/test_config.py
import pytest
from pathlib import Path
from core.config import Config

def test_config_loads_settings():
    cfg = Config()
    assert cfg.wake_word == "jarvis"
    assert cfg.stt.model == "medium"
    assert cfg.llm.ollama_model == "qwen3:8b"
    assert cfg.memory.context_history_count == 5

def test_config_env_override(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-123")
    cfg = Config()
    assert cfg.gemini_api_key == "test-key-123"
```

- [ ] **Step 3: Run test to confirm failure**

```bash
pytest tests/test_config.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'core'`

- [ ] **Step 4: Create `core/config.py`**

```python
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
```

- [ ] **Step 5: Run test to confirm pass**

```bash
pytest tests/test_config.py -v
```
Expected: PASS (2 tests)

- [ ] **Step 6: Create `core/logger.py`**

```python
import logging
import sys
from pythonjsonlogger import jsonlogger

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(jsonlogger.JsonFormatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger
```

- [ ] **Step 7: Commit**

```bash
git add core/ tests/
git commit -m "M1: core config + logger"
```

---

### Task 3: Skill Base + Registry

**Files:**
- Create: `skills/__init__.py`
- Create: `skills/base.py`
- Create: `skills/registry.py`
- Create: `tests/test_skills.py` (base tests only)

- [ ] **Step 1: Write failing test**

```python
# tests/test_skills.py
import pytest
from skills.base import BaseSkill
from skills.registry import SkillRegistry

class EchoSkill(BaseSkill):
    name = "echo"
    description = "Echoes back input"
    parameters = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }

    async def execute(self, text: str) -> str:
        return text

@pytest.mark.asyncio
async def test_base_skill_execute():
    skill = EchoSkill()
    result = await skill.execute(text="hello")
    assert result == "hello"

def test_registry_register_and_get():
    registry = SkillRegistry()
    registry.register(EchoSkill())
    assert "echo" in registry.skill_names
    skill = registry.get("echo")
    assert skill.name == "echo"

def test_registry_function_definitions():
    registry = SkillRegistry()
    registry.register(EchoSkill())
    defs = registry.function_definitions()
    assert len(defs) == 1
    assert defs[0]["name"] == "echo"
    assert "description" in defs[0]
    assert "parameters" in defs[0]
```

- [ ] **Step 2: Run test — confirm failure**

```bash
pytest tests/test_skills.py -v
```

- [ ] **Step 3: Create `skills/base.py`**

```python
from abc import ABC, abstractmethod

class BaseSkill(ABC):
    name: str
    description: str
    parameters: dict

    @abstractmethod
    async def execute(self, **kwargs) -> str: ...
```

- [ ] **Step 4: Create `skills/registry.py`**

```python
from __future__ import annotations
import importlib
import pkgutil
from pathlib import Path
from skills.base import BaseSkill

class SkillRegistry:
    def __init__(self) -> None:
        self._skills: dict[str, BaseSkill] = {}

    def register(self, skill: BaseSkill) -> None:
        self._skills[skill.name] = skill

    def get(self, name: str) -> BaseSkill:
        return self._skills[name]

    @property
    def skill_names(self) -> list[str]:
        return list(self._skills.keys())

    def function_definitions(self) -> list[dict]:
        return [
            {
                "name": s.name,
                "description": s.description,
                "parameters": s.parameters,
            }
            for s in self._skills.values()
        ]

    def auto_discover(self) -> None:
        """Import all skills/*.py modules and register BaseSkill subclasses."""
        skills_path = Path(__file__).parent
        for finder, module_name, _ in pkgutil.iter_modules([str(skills_path)]):
            if module_name in ("base", "registry", "__init__"):
                continue
            module = importlib.import_module(f"skills.{module_name}")
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                try:
                    if (
                        isinstance(attr, type)
                        and issubclass(attr, BaseSkill)
                        and attr is not BaseSkill
                    ):
                        self.register(attr())
                except TypeError:
                    pass
```

- [ ] **Step 5: Run test — confirm pass**

```bash
pytest tests/test_skills.py -v
```

- [ ] **Step 6: Create remaining `__init__.py` stubs**

Create empty `__init__.py` in: `stt/`, `brain/`, `tts/`, `memory/`

- [ ] **Step 7: Create MASTER_PLAN.md and INFOAI.md**

Create `MASTER_PLAN.md`:

```markdown
# J.A.R.V.I.S AGI — Master Plan

## Status

| Milestone | Status | Notes |
|---|---|---|
| M1 | ✅ DONE | Scaffolding, config, logger, skill base/registry |
| M2 | 🔲 NEXT | STT pipeline: VAD, hotword, transcription |
| M3 | 🔲 TODO | LLM brain + intent router |
| M4 | 🔲 TODO | TTS: edge-tts, interruptible |
| M5 | 🔲 TODO | Core engine loop + SQLite memory |
| M6 | 🔲 TODO | 11 skills |
| M7 | 🔲 TODO | ChromaDB semantic memory + user facts |
| M8 | 🔲 TODO | Polish + persistent service |

## Architecture Quick Reference

```
Mic → VAD → Hotword → STT → Brain (Ollama/Gemini) → Skill → TTS
                                    ↕
                            Memory (SQLite + ChromaDB)
```

## Key Files
- Entry point: `main.py`
- Orchestrator: `core/engine.py`
- Skill base: `skills/base.py`
- Config: `settings.yaml` + `.env`
- Plan: `docs/superpowers/plans/2026-04-06-jarvis-agi.md`
- Spec: `docs/superpowers/specs/2026-04-06-jarvis-agi-design.md`
```

Create `INFOAI.md`:

```markdown
# J.A.R.V.I.S AGI — AI Agent Handoff Document

## Project Summary
Local-first Windows desktop AI voice assistant. Wake word "Jarvis" → voice command → LLM routes to skill → spoken response. Runs 24/7 in background.

## Current State
Check `MASTER_PLAN.md` for current milestone status.

## Architecture
- **STT**: faster-whisper (CUDA) + pvporcupine hotword + webrtcvad
- **Brain**: Ollama qwen3:8b (primary) → Gemini API (fallback). LLM receives skill list as function defs, returns `{"skill": "<name>", "params": {...}}`.
- **TTS**: edge-tts async, interruptible. Porcupine runs in daemon thread; `cancel_event` stops TTS.
- **Memory**: SQLite (chat_history + user_facts tables) + ChromaDB (semantic search, added M7)
- **Skills**: 11 skills + general_chat fallback. All extend `BaseSkill`, auto-discovered by `SkillRegistry`.

## Key Conventions
- Python 3.11+, type hints required, async for I/O
- Config via `core/config.py` (loads `settings.yaml` + `.env`)
- Logging via `core/logger.py` → `get_logger(__name__)`
- All skills in `skills/`, extend `BaseSkill`, auto-discovered
- Data stored in `data/` (gitignored): `jarvis.db`, `chroma/`
- Test with: `pytest tests/ -v`

## Environment
- Windows 11, RTX 2070 (8GB VRAM), 32GB RAM
- Ollama running at `http://localhost:11434`
- CUDA available for faster-whisper + sentence-transformers

## Secrets (in .env)
- `PORCUPINE_ACCESS_KEY` — from Picovoice console (free tier)
- `GEMINI_API_KEY` — Google AI Studio
- `OPENWEATHERMAP_API_KEY` — optional, for weather skill

## If You're Starting a New Session
1. Read `MASTER_PLAN.md` to find current milestone
2. Read this file for architecture context
3. Check `docs/superpowers/plans/2026-04-06-jarvis-agi.md` for next tasks
4. Run `pytest tests/ -v` to verify current state
```

- [ ] **Step 8: Final M1 commit**

```bash
git add .
git commit -m "M1: complete scaffolding — config, logger, skill base/registry, docs"
```

---

## Chunk 2: M2 — STT Pipeline

### Task 4: Audio Capture with VAD

**Files:**
- Create: `stt/__init__.py`
- Create: `stt/audio_capture.py`
- Create: `tests/test_stt.py`

- [ ] **Step 1: Write failing test for audio capture**

```python
# tests/test_stt.py
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from stt.audio_capture import AudioCapture

def test_audio_capture_init():
    cap = AudioCapture(sample_rate=16000, frame_duration_ms=30)
    assert cap.sample_rate == 16000
    assert cap.frame_duration_ms == 30

def test_vad_detects_silence():
    cap = AudioCapture()
    silent_frame = np.zeros(480, dtype=np.int16).tobytes()
    # VAD on pure zeros should return False (no speech)
    result = cap.is_speech(silent_frame)
    assert isinstance(result, bool)

def test_frame_size_calculation():
    cap = AudioCapture(sample_rate=16000, frame_duration_ms=30)
    # 16000 samples/sec * 0.030 sec = 480 samples * 2 bytes = 960 bytes
    assert cap.frame_size_bytes == 960
```

- [ ] **Step 2: Run test — confirm failure**

```bash
pytest tests/test_stt.py -v
```

- [ ] **Step 3: Create `stt/audio_capture.py`**

```python
from __future__ import annotations
import asyncio
import threading
from collections import deque
from typing import AsyncIterator
import numpy as np
import sounddevice as sd
import webrtcvad
from core.logger import get_logger

logger = get_logger(__name__)

class AudioCapture:
    def __init__(
        self,
        sample_rate: int = 16000,
        frame_duration_ms: int = 30,
        channels: int = 1,
        vad_aggressiveness: int = 2,
    ) -> None:
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.channels = channels
        self.frame_size = int(sample_rate * frame_duration_ms / 1000)
        self.frame_size_bytes = self.frame_size * 2  # int16 = 2 bytes
        self._vad = webrtcvad.Vad(vad_aggressiveness)
        self._buffer: deque[bytes] = deque()
        self._lock = threading.Lock()
        self._stream: sd.RawInputStream | None = None

    def is_speech(self, frame: bytes) -> bool:
        try:
            return self._vad.is_speech(frame, self.sample_rate)
        except Exception:
            return False

    def start(self) -> None:
        def callback(indata, frames, time, status):
            with self._lock:
                self._buffer.append(bytes(indata))

        self._stream = sd.RawInputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="int16",
            blocksize=self.frame_size,
            callback=callback,
        )
        self._stream.start()
        logger.info("Audio capture started")

    def stop(self) -> None:
        if self._stream:
            self._stream.stop()
            self._stream.close()
            logger.info("Audio capture stopped")

    def read_frame(self) -> bytes | None:
        with self._lock:
            return self._buffer.popleft() if self._buffer else None

    async def record_until_silence(
        self,
        silence_threshold_sec: float = 1.5,
        max_duration_sec: float = 30.0,
    ) -> bytes:
        """Record audio until silence is detected. Returns raw PCM bytes."""
        frames: list[bytes] = []
        silence_frames = 0
        max_silence = int(silence_threshold_sec * 1000 / self.frame_duration_ms)
        max_frames = int(max_duration_sec * 1000 / self.frame_duration_ms)

        while len(frames) < max_frames:
            frame = self.read_frame()
            if frame is None:
                await asyncio.sleep(0.005)
                continue
            frames.append(frame)
            if self.is_speech(frame):
                silence_frames = 0
            else:
                silence_frames += 1
                if silence_frames >= max_silence and len(frames) > 10:
                    break

        return b"".join(frames)
```

- [ ] **Step 4: Run tests — confirm pass**

```bash
pytest tests/test_stt.py -v
```

---

### Task 5: Wake Word Detection (Porcupine)

**Files:**
- Create: `stt/hotword.py`

- [ ] **Step 1: Add test for hotword detector init**

Append to `tests/test_stt.py`:

```python
def test_hotword_detector_init_no_key():
    """Without a real key, HotwordDetector should raise on start, not on init."""
    from stt.hotword import HotwordDetector
    detector = HotwordDetector(access_key="", keyword="jarvis")
    assert detector.keyword == "jarvis"
    # Don't call start() — no real Porcupine key in test env
```

- [ ] **Step 2: Create `stt/hotword.py`**

```python
from __future__ import annotations
import asyncio
import threading
from typing import Callable
import pvporcupine
import sounddevice as sd
import numpy as np
from core.logger import get_logger

logger = get_logger(__name__)

class HotwordDetector:
    def __init__(self, access_key: str, keyword: str = "jarvis") -> None:
        self.access_key = access_key
        self.keyword = keyword
        self._porcupine: pvporcupine.Porcupine | None = None
        self._thread: threading.Thread | None = None
        self._running = False
        self._callback: Callable | None = None

    def on_detect(self, callback: Callable) -> None:
        self._callback = callback

    def start(self) -> None:
        if not self.access_key:
            logger.warning("No Porcupine access key — using keyboard fallback mode")
            self._start_keyboard_fallback()
            return

        self._porcupine = pvporcupine.create(
            access_key=self.access_key,
            keywords=[self.keyword],
        )
        self._running = True
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()
        logger.info(f"Hotword detector started: listening for '{self.keyword}'")

    def _listen_loop(self) -> None:
        frame_length = self._porcupine.frame_length
        with sd.RawInputStream(
            samplerate=self._porcupine.sample_rate,
            channels=1,
            dtype="int16",
            blocksize=frame_length,
        ) as stream:
            while self._running:
                pcm, _ = stream.read(frame_length)
                pcm_array = np.frombuffer(pcm, dtype=np.int16)
                result = self._porcupine.process(pcm_array)
                if result >= 0 and self._callback:
                    logger.info("Wake word detected!")
                    self._callback()

    def _start_keyboard_fallback(self) -> None:
        def wait_for_enter():
            while True:
                input("\n[FALLBACK] Press Enter to activate Jarvis...\n")
                if self._callback:
                    self._callback()
        self._thread = threading.Thread(target=wait_for_enter, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._porcupine:
            self._porcupine.delete()
```

- [ ] **Step 3: Run tests**

```bash
pytest tests/test_stt.py -v
```

---

### Task 6: Transcription (faster-whisper)

**Files:**
- Create: `stt/transcriber.py`

- [ ] **Step 1: Add transcriber tests**

Append to `tests/test_stt.py`:

```python
def test_transcriber_init():
    from stt.transcriber import Transcriber
    t = Transcriber(model_size="tiny", device="cpu", compute_type="int8")
    assert t.model_size == "tiny"
    assert not t._loaded

@pytest.mark.asyncio
async def test_transcriber_transcribes_audio():
    """Integration test — requires faster-whisper installed. Uses tiny model on CPU."""
    from stt.transcriber import Transcriber
    import wave, struct, math
    # Generate 1 second of 440Hz sine wave at 16kHz
    sample_rate = 16000
    duration = 1
    samples = [int(32767 * math.sin(2 * math.pi * 440 * i / sample_rate))
               for i in range(sample_rate * duration)]
    pcm = struct.pack(f"{len(samples)}h", *samples)
    t = Transcriber(model_size="tiny", device="cpu", compute_type="int8")
    result = await t.transcribe(pcm, sample_rate=sample_rate)
    assert isinstance(result, str)  # may be empty for a sine tone, that's ok
```

- [ ] **Step 2: Create `stt/transcriber.py`**

```python
from __future__ import annotations
import asyncio
import io
import wave
from faster_whisper import WhisperModel
from core.logger import get_logger

logger = get_logger(__name__)

class Transcriber:
    def __init__(
        self,
        model_size: str = "medium",
        device: str = "cuda",
        compute_type: str = "float16",
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._model: WhisperModel | None = None
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        logger.info(f"Loading Whisper {self.model_size} on {self.device}...")
        self._model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type,
        )
        self._loaded = True
        logger.info("Whisper loaded")

    async def transcribe(self, pcm_bytes: bytes, sample_rate: int = 16000) -> str:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._transcribe_sync, pcm_bytes, sample_rate)

    def _transcribe_sync(self, pcm_bytes: bytes, sample_rate: int) -> str:
        self._load()
        # Write PCM to in-memory WAV
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm_bytes)
        buf.seek(0)

        segments, info = self._model.transcribe(buf, beam_size=5)
        text = " ".join(seg.text for seg in segments).strip()
        logger.info(f"Transcribed [{info.language}]: {text!r}")
        return text
```

- [ ] **Step 3: Run STT tests**

```bash
pytest tests/test_stt.py -v
```

- [ ] **Step 4: Commit M2**

```bash
git add stt/ tests/test_stt.py
git commit -m "M2: STT pipeline — audio capture, VAD, hotword, transcription"
```

- [ ] **Step 5: Update MASTER_PLAN.md — mark M2 DONE, M3 NEXT**

---

## Chunk 3: M3 — LLM Brain + Intent Router

### Task 7: Ollama Client

**Files:**
- Create: `brain/__init__.py`
- Create: `brain/ollama_client.py`
- Create: `tests/test_brain.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_brain.py
import pytest
import httpx
from pytest_httpx import HTTPXMock
from brain.ollama_client import OllamaClient

@pytest.mark.asyncio
async def test_ollama_chat_returns_string(httpx_mock: HTTPXMock):
    httpx_mock.add_response(
        url="http://localhost:11434/api/chat",
        json={"message": {"content": "Hello!"}},
    )
    client = OllamaClient(base_url="http://localhost:11434", model="qwen3:8b")
    result = await client.chat([{"role": "user", "content": "hi"}])
    assert result == "Hello!"

@pytest.mark.asyncio
async def test_ollama_chat_timeout_raises(httpx_mock: HTTPXMock):
    httpx_mock.add_exception(httpx.TimeoutException("timeout"))
    client = OllamaClient(base_url="http://localhost:11434", model="qwen3:8b")
    with pytest.raises(httpx.TimeoutException):
        await client.chat([{"role": "user", "content": "hi"}])
```

- [ ] **Step 2: Create `brain/ollama_client.py`**

```python
from __future__ import annotations
import httpx
from core.logger import get_logger

logger = get_logger(__name__)

class OllamaClient:
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen3:8b",
        timeout: int = 30,
    ) -> None:
        self.base_url = base_url
        self.model = model
        self.timeout = timeout

    async def chat(self, messages: list[dict], json_mode: bool = False) -> str:
        payload: dict = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }
        if json_mode:
            payload["format"] = "json"

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/api/chat",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            return data["message"]["content"]
```

- [ ] **Step 3: Run tests**

```bash
pytest tests/test_brain.py -v
```

---

### Task 8: Gemini Client (Fallback)

**Files:**
- Create: `brain/gemini_client.py`

- [ ] **Step 1: Add test**

Append to `tests/test_brain.py`:

```python
@pytest.mark.asyncio
async def test_gemini_client_no_key_raises():
    from brain.gemini_client import GeminiClient
    client = GeminiClient(api_key="")
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        await client.chat([{"role": "user", "content": "hi"}])
```

- [ ] **Step 2: Create `brain/gemini_client.py`**

```python
from __future__ import annotations
import asyncio
import google.generativeai as genai
from core.logger import get_logger

logger = get_logger(__name__)

class GeminiClient:
    def __init__(self, api_key: str, model: str = "gemini-1.5-flash") -> None:
        self.api_key = api_key
        self.model = model

    async def chat(self, messages: list[dict]) -> str:
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not set")
        genai.configure(api_key=self.api_key)
        model = genai.GenerativeModel(self.model)

        # Convert to Gemini format
        history = []
        for msg in messages[:-1]:
            role = "user" if msg["role"] == "user" else "model"
            history.append({"role": role, "parts": [msg["content"]]})

        chat = model.start_chat(history=history)
        last = messages[-1]["content"]

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, chat.send_message, last)
        return response.text
```

- [ ] **Step 3: Run tests**

```bash
pytest tests/test_brain.py -v
```

---

### Task 9: Prompts + Intent Router

**Files:**
- Create: `brain/prompts.py`
- Create: `brain/router.py`

- [ ] **Step 1: Add router tests**

Append to `tests/test_brain.py`:

```python
@pytest.mark.asyncio
async def test_router_returns_skill_and_params(httpx_mock: HTTPXMock):
    from brain.router import IntentRouter
    from skills.registry import SkillRegistry
    from skills.base import BaseSkill

    class WeatherSkill(BaseSkill):
        name = "weather"
        description = "Get weather for a location"
        parameters = {
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"],
        }
        async def execute(self, **kwargs) -> str:
            return "sunny"

    registry = SkillRegistry()
    registry.register(WeatherSkill())

    httpx_mock.add_response(
        url="http://localhost:11434/api/chat",
        json={"message": {"content": '{"skill": "weather", "params": {"location": "Prague"}}'}},
    )
    router = IntentRouter(registry=registry, ollama_url="http://localhost:11434", model="qwen3:8b")
    skill_name, params = await router.route("What's the weather in Prague?", context=[])

    assert skill_name == "weather"
    assert params["location"] == "Prague"

@pytest.mark.asyncio
async def test_router_unknown_skill_falls_back_to_general_chat(httpx_mock: HTTPXMock):
    from brain.router import IntentRouter
    from skills.registry import SkillRegistry
    from skills.general_chat import GeneralChatSkill

    registry = SkillRegistry()
    registry.register(GeneralChatSkill(ollama_url="http://localhost:11434", model="qwen3:8b"))

    # First call returns unknown skill; second call (retry) also bad → fallback
    httpx_mock.add_response(
        url="http://localhost:11434/api/chat",
        json={"message": {"content": '{"skill": "nonexistent_skill", "params": {}}'}},
    )
    router = IntentRouter(registry=registry, ollama_url="http://localhost:11434", model="qwen3:8b")
    skill_name, params = await router.route("something random", context=[])

    assert skill_name == "general_chat"

@pytest.mark.asyncio
async def test_router_retries_on_bad_json(httpx_mock: HTTPXMock):
    from brain.router import IntentRouter
    from skills.registry import SkillRegistry
    from skills.general_chat import GeneralChatSkill

    registry = SkillRegistry()
    registry.register(GeneralChatSkill(ollama_url="http://localhost:11434", model="qwen3:8b"))

    # First response is invalid JSON, second is valid
    httpx_mock.add_response(json={"message": {"content": "oops not json"}})
    httpx_mock.add_response(json={"message": {"content": '{"skill": "general_chat", "params": {"message": "hi"}}'}})

    router = IntentRouter(registry=registry, ollama_url="http://localhost:11434", model="qwen3:8b")
    skill_name, params = await router.route("hi", context=[])
    assert skill_name == "general_chat"
```

- [ ] **Step 2: Create `brain/prompts.py`**

```python
ROUTER_SYSTEM_PROMPT = """You are J.A.R.V.I.S., an AI assistant. Your job is to analyze the user's request and decide which skill to use.

You MUST respond with ONLY a single JSON object in this exact format:
{{"skill": "<skill_name>", "params": {{...}}}}

Available skills:
{skill_definitions}

Rules:
- skill must be one of the listed skill names
- params must match the skill's parameter schema
- if no skill fits, use "general_chat" with {{"message": "<user input>"}}
- respond with JSON ONLY, no other text
"""

def build_router_prompt(skill_definitions: list[dict]) -> str:
    defs_text = "\n".join(
        f"- {s['name']}: {s['description']} | params: {s['parameters']}"
        for s in skill_definitions
    )
    return ROUTER_SYSTEM_PROMPT.format(skill_definitions=defs_text)
```

- [ ] **Step 3: Create `brain/router.py`**

```python
from __future__ import annotations
import json
from brain.ollama_client import OllamaClient
from brain.gemini_client import GeminiClient
from brain.prompts import build_router_prompt
from skills.registry import SkillRegistry
from core.logger import get_logger

logger = get_logger(__name__)

class IntentRouter:
    def __init__(
        self,
        registry: SkillRegistry,
        ollama_url: str,
        model: str,
        gemini_api_key: str = "",
        timeout: int = 30,
    ) -> None:
        self._registry = registry
        self._ollama = OllamaClient(base_url=ollama_url, model=model, timeout=timeout)
        self._gemini = GeminiClient(api_key=gemini_api_key) if gemini_api_key else None

    async def _call_llm(self, messages: list[dict]) -> str:
        try:
            return await self._ollama.chat(messages, json_mode=True)
        except Exception as e:
            logger.warning(f"Ollama failed: {e}. Trying Gemini...")
            if self._gemini:
                return await self._gemini.chat(messages)
            raise

    async def route(
        self, user_text: str, context: list[dict]
    ) -> tuple[str, dict]:
        system_prompt = build_router_prompt(self._registry.function_definitions())
        messages = [
            {"role": "system", "content": system_prompt},
            *context,
            {"role": "user", "content": user_text},
        ]

        raw = await self._call_llm(messages)
        result = self._parse(raw, user_text)
        if result is None:
            # Retry once with explicit format reminder
            retry_messages = messages + [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": 'Respond with ONLY JSON: {"skill": "...", "params": {...}}'},
            ]
            raw2 = await self._call_llm(retry_messages)
            result = self._parse(raw2, user_text)

        return result or ("general_chat", {"message": user_text})

    def _parse(self, raw: str, user_text: str) -> tuple[str, dict] | None:
        try:
            data = json.loads(raw)
            skill_name = data.get("skill", "general_chat")
            params = data.get("params", {})
            if skill_name not in self._registry.skill_names:
                logger.warning(f"Unknown skill '{skill_name}', falling back to general_chat")
                skill_name = "general_chat"
                params = {"message": user_text}
            return skill_name, params
        except json.JSONDecodeError:
            logger.warning(f"Router parse failed on: {raw!r}. Will retry.")
            return None
```

- [ ] **Step 4: Create stub `skills/general_chat.py`** (needed by router tests)

```python
from skills.base import BaseSkill
from core.logger import get_logger

logger = get_logger(__name__)

class GeneralChatSkill(BaseSkill):
    name = "general_chat"
    description = "Have a general conversation or answer questions"
    parameters = {
        "type": "object",
        "properties": {"message": {"type": "string"}},
        "required": ["message"],
    }

    def __init__(self, ollama_url: str = "http://localhost:11434", model: str = "qwen3:8b") -> None:
        self.ollama_url = ollama_url
        self.model = model

    async def execute(self, message: str, **kwargs) -> str:
        from brain.ollama_client import OllamaClient
        client = OllamaClient(base_url=self.ollama_url, model=self.model)
        messages = [
            {"role": "system", "content": "You are J.A.R.V.I.S., a helpful AI assistant."},
            {"role": "user", "content": message},
        ]
        try:
            return await client.chat(messages)
        except Exception as e:
            logger.error(f"general_chat failed: {e}")
            return "I'm having trouble thinking right now."
```

- [ ] **Step 5: Run all tests**

```bash
pytest tests/ -v
```

- [ ] **Step 6: Commit M3**

```bash
git add brain/ skills/general_chat.py tests/test_brain.py
git commit -m "M3: LLM brain — ollama client, gemini fallback, prompts, intent router"
```

- [ ] **Step 7: Update MASTER_PLAN.md — M3 DONE, M4 NEXT**

---

## Chunk 4: M4 — Text-to-Speech

### Task 10: Speaker (edge-tts)

**Files:**
- Create: `tts/__init__.py`
- Create: `tts/speaker.py`
- Create: `tests/test_tts.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_tts.py
import pytest
import asyncio
from unittest.mock import patch, AsyncMock
from tts.speaker import Speaker

def test_speaker_detects_language_en():
    sp = Speaker(voice_en="en-US-JennyNeural", voice_cs="cs-CZ-VlastaNeural")
    assert sp._detect_language("Hello, how are you?") == "en"

def test_speaker_detects_language_cs():
    sp = Speaker(voice_en="en-US-JennyNeural", voice_cs="cs-CZ-VlastaNeural")
    assert sp._detect_language("Ahoj, jak se máš?") == "cs"

@pytest.mark.asyncio
async def test_speaker_can_be_interrupted():
    sp = Speaker(voice_en="en-US-JennyNeural", voice_cs="cs-CZ-VlastaNeural")
    sp.interrupt()  # Should not raise even when not speaking
    assert not sp.is_speaking
```

- [ ] **Step 2: Create `tts/speaker.py`**

```python
from __future__ import annotations
import asyncio
import io
import threading
import re
import edge_tts
import sounddevice as sd
import soundfile as sf
from core.logger import get_logger

logger = get_logger(__name__)

_CS_PATTERN = re.compile(r"[áčďéěíňóřšťúůýžÁČĎÉĚÍŇÓŘŠŤÚŮÝŽ]")

class Speaker:
    def __init__(
        self,
        voice_en: str = "en-US-JennyNeural",
        voice_cs: str = "cs-CZ-VlastaNeural",
        rate: str = "+0%",
        volume: str = "+0%",
    ) -> None:
        self.voice_en = voice_en
        self.voice_cs = voice_cs
        self.rate = rate
        self.volume = volume
        self._cancel_event = asyncio.Event()
        self.is_speaking = False

    def _detect_language(self, text: str) -> str:
        cs_chars = len(_CS_PATTERN.findall(text))
        return "cs" if cs_chars >= 2 else "en"

    def interrupt(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        """Thread-safe: can be called from non-asyncio threads (e.g. hotword daemon)."""
        if loop and loop.is_running():
            loop.call_soon_threadsafe(self._cancel_event.set)
        else:
            self._cancel_event.set()

    async def speak(self, text: str) -> None:
        self._cancel_event.clear()
        voice = self.voice_cs if self._detect_language(text) == "cs" else self.voice_en

        communicate = edge_tts.Communicate(text, voice, rate=self.rate, volume=self.volume)
        audio_buf = io.BytesIO()

        async for chunk in communicate.stream():
            if self._cancel_event.is_set():
                logger.info("TTS interrupted before playback")
                return
            if chunk["type"] == "audio":
                audio_buf.write(chunk["data"])

        audio_buf.seek(0)
        if self._cancel_event.is_set():
            return

        await self._play(audio_buf)

    async def _play(self, audio_buf: io.BytesIO) -> None:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._play_sync, audio_buf)

    def _play_sync(self, audio_buf: io.BytesIO) -> None:
        self.is_speaking = True
        try:
            data, samplerate = sf.read(audio_buf, dtype="float32")
            stream = sd.play(data, samplerate)
            # Poll cancel_event every 50ms while sounddevice plays
            while sd.get_stream().active if hasattr(sd, 'get_stream') else False:
                if self._cancel_event.is_set():
                    sd.stop()
                    return
                threading.Event().wait(0.05)
            # Simpler: sleep in chunks and check cancel
            import numpy as _np
            duration = len(data) / samplerate
            elapsed = 0.0
            chunk = 0.05
            while elapsed < duration:
                if self._cancel_event.is_set():
                    sd.stop()
                    return
                threading.Event().wait(chunk)
                elapsed += chunk
            sd.wait()
        finally:
            self.is_speaking = False
```

- [ ] **Step 3: Run tests**

```bash
pytest tests/test_tts.py -v
```

- [ ] **Step 4: Commit M4**

```bash
git add tts/ tests/test_tts.py
git commit -m "M4: TTS — edge-tts speaker with language detection and interruption"
```

- [ ] **Step 5: Update MASTER_PLAN.md — M4 DONE, M5 NEXT**

---

## Chunk 5: M5 — Core Engine Loop + SQLite Memory

### Task 11: SQLite Memory (Basic)

**Files:**
- Create: `memory/__init__.py`
- Create: `memory/chat_history.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_memory.py
import pytest
import tempfile
from pathlib import Path
from memory.chat_history import ChatHistory

@pytest.fixture
def db():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    history = ChatHistory(db_path=Path(path))
    yield history

def test_log_and_retrieve(db):
    db.log("user", "Hello Jarvis")
    db.log("assistant", "Hello! How can I help?")
    history = db.recent(count=5)
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[1]["role"] == "assistant"

def test_recent_respects_count(db):
    for i in range(10):
        db.log("user", f"message {i}")
    history = db.recent(count=3)
    assert len(history) == 3
```

- [ ] **Step 2: Create `memory/chat_history.py`**

```python
from __future__ import annotations
import sqlite3
from datetime import datetime
from pathlib import Path

class ChatHistory:
    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._create_tables()

    def _create_tables(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                session_id TEXT DEFAULT ''
            )
        """)
        self._conn.commit()

    def log(self, role: str, content: str, session_id: str = "") -> None:
        self._conn.execute(
            "INSERT INTO chat_history (timestamp, role, content, session_id) VALUES (?, ?, ?, ?)",
            (datetime.utcnow().isoformat(), role, content, session_id),
        )
        self._conn.commit()

    def recent(self, count: int = 5, session_id: str = "") -> list[dict]:
        if session_id:
            rows = self._conn.execute(
                "SELECT role, content FROM chat_history WHERE session_id=? ORDER BY id DESC LIMIT ?",
                (session_id, count),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT role, content FROM chat_history ORDER BY id DESC LIMIT ?",
                (count,),
            ).fetchall()
        return [{"role": r, "content": c} for r, c in reversed(rows)]
```

- [ ] **Step 3: Run tests**

```bash
pytest tests/test_memory.py -v
```

---

### Task 12: Core Engine

**Files:**
- Create: `core/engine.py`
- Create: `main.py`

- [ ] **Step 1: Create `core/engine.py`**

```python
from __future__ import annotations
import asyncio
import uuid
from core.config import Config
from core.logger import get_logger
from stt.audio_capture import AudioCapture
from stt.hotword import HotwordDetector
from stt.transcriber import Transcriber
from brain.router import IntentRouter
from tts.speaker import Speaker
from memory.chat_history import ChatHistory
from skills.registry import SkillRegistry

logger = get_logger(__name__)

class Engine:
    def __init__(self, config: Config) -> None:
        self._cfg = config
        self._session_id = str(uuid.uuid4())

        self._audio = AudioCapture()
        self._hotword = HotwordDetector(
            access_key=config.porcupine_access_key,
            keyword=config.wake_word,
        )
        self._transcriber = Transcriber(
            model_size=config.stt.model,
            device=config.stt.device,
            compute_type=config.stt.compute_type,
        )
        self._speaker = Speaker(
            voice_en=config.tts.voice_en,
            voice_cs=config.tts.voice_cs,
            rate=config.tts.rate,
            volume=config.tts.volume,
        )
        self._history = ChatHistory(db_path=config.memory.db_path)
        self._registry = SkillRegistry()
        self._registry.auto_discover()
        self._router = IntentRouter(
            registry=self._registry,
            ollama_url=config.llm.ollama_url,
            model=config.llm.ollama_model,
            gemini_api_key=config.gemini_api_key,
            timeout=config.llm.timeout_sec,
        )

        self._wake_event = asyncio.Event()
        self._running = False

    def _on_wake(self) -> None:
        loop = asyncio.get_event_loop()
        self._speaker.interrupt(loop=loop)
        loop.call_soon_threadsafe(self._wake_event.set)

    async def run(self) -> None:
        self._running = True
        self._audio.start()
        self._hotword.on_detect(self._on_wake)
        self._hotword.start()
        logger.info("J.A.R.V.I.S. is online. Say 'Jarvis' to activate.")

        try:
            while self._running:
                await self._wake_event.wait()
                self._wake_event.clear()
                await self._handle_command()
        finally:
            self._audio.stop()
            self._hotword.stop()

    async def _handle_command(self) -> None:
        logger.info("Listening for command...")
        await self._speaker.speak("Yes?")

        pcm = await self._audio.record_until_silence(
            max_duration_sec=self._cfg.timeouts.stt_max_recording_sec
        )
        if not pcm:
            return

        text = await self._transcriber.transcribe(pcm)
        if not text.strip():
            return

        logger.info(f"User: {text}")
        self._history.log("user", text, self._session_id)

        context = self._history.recent(
            count=self._cfg.memory.context_history_count,
            session_id=self._session_id,
        )

        try:
            skill_name, params = await self._router.route(text, context=context[:-1])
            skill = self._registry.get(skill_name)
            response = await asyncio.wait_for(
                skill.execute(**params),
                timeout=self._cfg.timeouts.skill_execute_sec,
            )
        except Exception as e:
            logger.error(f"Skill execution failed: {e}")
            response = "I couldn't do that, sorry."

        logger.info(f"Jarvis: {response}")
        self._history.log("assistant", response, self._session_id)
        await self._speaker.speak(response)

    def stop(self) -> None:
        self._running = False
```

- [ ] **Step 2: Create `main.py`**

```python
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
```

- [ ] **Step 3: Write engine integration test (mocked)**

Append to `tests/test_brain.py` (or create `tests/test_engine.py`):

```python
# tests/test_engine.py
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from core.config import Config
from core.engine import Engine

@pytest.mark.asyncio
async def test_engine_handle_command_full_flow(tmp_path, httpx_mock):
    """Integration: mock mic PCM + LLM response → verify skill called + history logged."""
    from pytest_httpx import HTTPXMock

    # Patch audio to return fake PCM, STT to return "what time is it"
    fake_pcm = b"\x00" * 3200  # 0.1s of silence

    with patch("stt.audio_capture.AudioCapture.record_until_silence", return_value=fake_pcm), \
         patch("stt.transcriber.Transcriber.transcribe", return_value="what time is it"), \
         patch("stt.audio_capture.AudioCapture.start"), \
         patch("stt.audio_capture.AudioCapture.stop"), \
         patch("stt.hotword.HotwordDetector.start"), \
         patch("stt.hotword.HotwordDetector.stop"), \
         patch("tts.speaker.Speaker.speak", new_callable=AsyncMock) as mock_speak:

        httpx_mock.add_response(
            json={"message": {"content": '{"skill": "general_chat", "params": {"message": "what time is it"}}'}},
        )
        httpx_mock.add_response(
            json={"message": {"content": "It's time to get things done."}},
        )

        cfg = Config(settings_path="settings.yaml")
        cfg.memory.db_path = tmp_path / "test.db"
        cfg.memory.chroma_path = tmp_path / "chroma"

        engine = Engine(cfg)
        engine._wake_event.set()  # Trigger one command cycle

        # Run engine for one cycle then stop
        async def run_one():
            await engine._handle_command()

        await asyncio.wait_for(run_one(), timeout=10)

        mock_speak.assert_called()  # TTS was called with some response

        # Verify history was written
        history = engine._history.recent(count=5)
        roles = [h["role"] for h in history]
        assert "user" in roles
        assert "assistant" in roles
```

- [ ] **Step 4: Run engine tests**

```bash
pytest tests/test_engine.py -v
```

- [ ] **Step 5: Smoke test (manual)**

```bash
pip install -r requirements.txt
python main.py
# Press Ctrl+C to stop
```

- [ ] **Step 6: Commit M5**

```bash
git add core/engine.py main.py memory/ tests/test_memory.py tests/test_engine.py
git commit -m "M5: core engine loop + SQLite chat history"
```

- [ ] **Step 5: Update MASTER_PLAN.md — M5 DONE, M6 NEXT**

---

## Chunk 6: M6 — Skills (11 Skills)

### Task 13: App Launcher

**Files:** `skills/app_launcher.py`

- [ ] **Step 1: Create `skills/app_launcher.py`**

```python
import subprocess
from skills.base import BaseSkill
from core.logger import get_logger

logger = get_logger(__name__)

class AppLauncherSkill(BaseSkill):
    name = "app_launcher"
    description = "Open or close a Windows application by name"
    parameters = {
        "type": "object",
        "properties": {
            "app_name": {"type": "string", "description": "Application name, e.g. 'notepad', 'chrome', 'spotify'"},
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
            import psutil
            for proc in psutil.process_iter(["name"]):
                if app_name.lower() in proc.info["name"].lower():
                    proc.kill()
                    return f"Closed {app_name}."
            return f"{app_name} doesn't seem to be running."
```

---

### Task 14: Web Search

**Files:** `skills/web_search.py`

- [ ] **Step 1: Create `skills/web_search.py`**

```python
from duckduckgo_search import DDGS
from skills.base import BaseSkill
from core.logger import get_logger

logger = get_logger(__name__)

class WebSearchSkill(BaseSkill):
    name = "web_search"
    description = "Search the web and return top results"
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "num_results": {"type": "integer", "default": 3},
        },
        "required": ["query"],
    }

    async def execute(self, query: str, num_results: int = 3, **kwargs) -> str:
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=num_results))
            if not results:
                return "No results found."
            lines = [f"{r['title']}: {r['body']}" for r in results]
            return "\n\n".join(lines)
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return f"Search failed: {e}"
```

---

### Task 15: Web Analyzer

**Files:** `skills/web_analyzer.py`

- [ ] **Step 1: Create `skills/web_analyzer.py`**

```python
from playwright.async_api import async_playwright
from brain.ollama_client import OllamaClient
from skills.base import BaseSkill
from core.logger import get_logger

logger = get_logger(__name__)

class WebAnalyzerSkill(BaseSkill):
    name = "web_analyzer"
    description = "Fetch a webpage and summarize its content"
    parameters = {
        "type": "object",
        "properties": {
            "url": {"type": "string"},
            "question": {"type": "string", "description": "Optional question to answer from page content"},
        },
        "required": ["url"],
    }

    def __init__(self, ollama_url: str = "http://localhost:11434", model: str = "qwen3:8b") -> None:
        self._client = OllamaClient(base_url=ollama_url, model=model)

    async def execute(self, url: str, question: str = "", **kwargs) -> str:
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()
                await page.goto(url, timeout=15000)
                text = await page.inner_text("body")
                await browser.close()

            text = text[:4000]
            prompt = question if question else f"Summarize this webpage content in 3 sentences."
            messages = [
                {"role": "system", "content": "You are a helpful assistant that analyzes web pages."},
                {"role": "user", "content": f"{prompt}\n\nPage content:\n{text}"},
            ]
            return await self._client.chat(messages)
        except Exception as e:
            logger.error(f"Web analysis failed: {e}")
            return f"I couldn't analyze that page: {e}"
```

---

### Task 16: Screenshot, System Info, Weather, Note Taker, Timer, File Manager, Music Player

**Files:** One file each in `skills/`

- [ ] **Step 1: Create `skills/screenshot.py`**

```python
import asyncio
import pyautogui
from pathlib import Path
from datetime import datetime
from skills.base import BaseSkill
from core.logger import get_logger

logger = get_logger(__name__)

class ScreenshotSkill(BaseSkill):
    name = "screenshot"
    description = "Take a screenshot of the current screen"
    parameters = {
        "type": "object",
        "properties": {
            "save_path": {"type": "string", "description": "Optional file path to save screenshot"},
        },
    }

    async def execute(self, save_path: str = "", **kwargs) -> str:
        loop = asyncio.get_event_loop()
        path = save_path or f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        await loop.run_in_executor(None, lambda: pyautogui.screenshot(path))
        return f"Screenshot saved to {path}."
```

- [ ] **Step 2: Create `skills/system_info.py`**

```python
import psutil
from skills.base import BaseSkill

class SystemInfoSkill(BaseSkill):
    name = "system_info"
    description = "Get CPU, RAM, GPU usage and system stats"
    parameters = {"type": "object", "properties": {}}

    async def execute(self, **kwargs) -> str:
        import os
        cpu = psutil.cpu_percent(interval=1)
        ram = psutil.virtual_memory()
        root = os.path.splitdrive(os.getcwd())[0] + "/" or "/"
        disk = psutil.disk_usage(root)
        return (
            f"CPU: {cpu}% | "
            f"RAM: {ram.percent}% ({ram.used // 1024**3}GB / {ram.total // 1024**3}GB) | "
            f"Disk: {disk.percent}% used"
        )
```

- [ ] **Step 3: Create `skills/weather.py`**

```python
import httpx
from skills.base import BaseSkill
from core.logger import get_logger

logger = get_logger(__name__)

class WeatherSkill(BaseSkill):
    name = "weather"
    description = "Get current weather for a location"
    parameters = {
        "type": "object",
        "properties": {
            "location": {"type": "string", "description": "City name or location"},
        },
        "required": ["location"],
    }

    async def execute(self, location: str, **kwargs) -> str:
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.get(f"https://wttr.in/{location}?format=3")
                r.raise_for_status()
                return r.text.strip()
        except Exception as e:
            logger.error(f"Weather failed: {e}")
            return f"Couldn't get weather for {location}."
```

- [ ] **Step 4: Create `skills/note_taker.py`**

```python
import sqlite3
from datetime import datetime
from pathlib import Path
from skills.base import BaseSkill

class NoteTakerSkill(BaseSkill):
    name = "note_taker"
    description = "Save or retrieve notes"
    parameters = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["save", "list", "search"]},
            "content": {"type": "string", "description": "Note content (for save)"},
            "query": {"type": "string", "description": "Search query (for search)"},
        },
        "required": ["action"],
    }

    def __init__(self, db_path: str = "./data/jarvis.db") -> None:
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                content TEXT
            )
        """)
        self._conn.commit()

    async def execute(self, action: str, content: str = "", query: str = "", **kwargs) -> str:
        if action == "save":
            self._conn.execute(
                "INSERT INTO notes (timestamp, content) VALUES (?, ?)",
                (datetime.utcnow().isoformat(), content),
            )
            self._conn.commit()
            return "Note saved."
        elif action == "list":
            rows = self._conn.execute(
                "SELECT timestamp, content FROM notes ORDER BY id DESC LIMIT 5"
            ).fetchall()
            if not rows:
                return "No notes found."
            return "\n".join(f"[{ts[:10]}] {c}" for ts, c in rows)
        elif action == "search":
            rows = self._conn.execute(
                "SELECT timestamp, content FROM notes WHERE content LIKE ? LIMIT 5",
                (f"%{query}%",),
            ).fetchall()
            return "\n".join(f"[{ts[:10]}] {c}" for ts, c in rows) or "No matching notes."
        return "Unknown action."
```

- [ ] **Step 5: Create `skills/timer_alarm.py`**

```python
import asyncio
from skills.base import BaseSkill
from core.logger import get_logger

logger = get_logger(__name__)

class TimerAlarmSkill(BaseSkill):
    name = "timer_alarm"
    description = "Set a countdown timer or alarm"
    parameters = {
        "type": "object",
        "properties": {
            "seconds": {"type": "integer", "description": "Timer duration in seconds"},
            "label": {"type": "string", "description": "What to call this timer"},
        },
        "required": ["seconds"],
    }

    async def execute(self, seconds: int, label: str = "Timer", **kwargs) -> str:
        async def _fire():
            await asyncio.sleep(seconds)
            logger.info(f"TIMER DONE: {label}")
            # TTS notification would be injected here via engine callback in future
            print(f"\n⏰ {label} is done!\n")

        asyncio.create_task(_fire())
        mins, secs = divmod(seconds, 60)
        if mins:
            return f"{label} set for {mins} minute{'s' if mins != 1 else ''} and {secs} seconds."
        return f"{label} set for {seconds} seconds."
```

- [ ] **Step 6: Create `skills/file_manager.py`**

```python
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
```

- [ ] **Step 7: Create `skills/music_player.py`**

```python
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
            # yt-dlp to get audio URL, then play with vlc/ffplay
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
```

- [ ] **Step 8: Add skill tests**

Append to `tests/test_skills.py`:

```python
@pytest.mark.asyncio
async def test_system_info_returns_stats():
    from skills.system_info import SystemInfoSkill
    skill = SystemInfoSkill()
    result = await skill.execute()
    assert "CPU" in result
    assert "RAM" in result

@pytest.mark.asyncio
async def test_weather_skill_calls_wttr(httpx_mock):
    from skills.weather import WeatherSkill
    httpx_mock.add_response(text="Prague: ⛅ 12°C")
    skill = WeatherSkill()
    result = await skill.execute(location="Prague")
    assert "Prague" in result

@pytest.mark.asyncio
async def test_note_taker_save_and_list(tmp_path):
    from skills.note_taker import NoteTakerSkill
    skill = NoteTakerSkill(db_path=str(tmp_path / "test.db"))
    await skill.execute(action="save", content="Buy milk")
    result = await skill.execute(action="list")
    assert "Buy milk" in result

@pytest.mark.asyncio
async def test_web_search_returns_results(httpx_mock):
    """DuckDuckGo search uses its own HTTP client; mock at the skill level."""
    from skills.web_search import WebSearchSkill
    from unittest.mock import patch, MagicMock
    mock_results = [{"title": "Test", "body": "Result body", "href": "http://example.com"}]
    with patch("duckduckgo_search.DDGS.text", return_value=iter(mock_results)):
        skill = WebSearchSkill()
        result = await skill.execute(query="test query")
    assert "Result body" in result

@pytest.mark.asyncio
async def test_app_launcher_open(tmp_path):
    from skills.app_launcher import AppLauncherSkill
    from unittest.mock import patch
    with patch("subprocess.Popen") as mock_popen:
        skill = AppLauncherSkill()
        result = await skill.execute(app_name="notepad", action="open")
    assert "Opening" in result
    mock_popen.assert_called_once()

@pytest.mark.asyncio
async def test_timer_alarm_creates_task():
    from skills.timer_alarm import TimerAlarmSkill
    skill = TimerAlarmSkill()
    result = await skill.execute(seconds=90, label="Test Timer")
    assert "1 minute" in result
    assert "30 seconds" in result

@pytest.mark.asyncio
async def test_file_manager_list(tmp_path):
    from skills.file_manager import FileManagerSkill
    (tmp_path / "file.txt").write_text("hello")
    (tmp_path / "subdir").mkdir()
    skill = FileManagerSkill()
    result = await skill.execute(action="list", path=str(tmp_path))
    assert "file.txt" in result
    assert "subdir" in result

@pytest.mark.asyncio
async def test_screenshot_calls_pyautogui(tmp_path):
    from skills.screenshot import ScreenshotSkill
    from unittest.mock import patch
    out_path = str(tmp_path / "shot.png")
    with patch("pyautogui.screenshot") as mock_ss:
        skill = ScreenshotSkill()
        result = await skill.execute(save_path=out_path)
    mock_ss.assert_called_once_with(out_path)
    assert "shot.png" in result
```

- [ ] **Step 9: Run all tests**

```bash
pytest tests/ -v
```

- [ ] **Step 10: Commit M6**

```bash
git add skills/
git commit -m "M6: 11 skills — app launcher, web search/analyzer, music, files, screenshot, weather, notes, timer, system info, general chat"
```

- [ ] **Step 11: Update MASTER_PLAN.md — M6 DONE, M7 NEXT**

---

## Chunk 7: M7 — Memory System (ChromaDB + User Facts)

### Task 17: User Facts + Vector Store

**Files:**
- Create: `memory/user_facts.py`
- Create: `memory/vector_store.py`

- [ ] **Step 1: Create `memory/user_facts.py`**

```python
from __future__ import annotations
import sqlite3
from datetime import datetime
from pathlib import Path

class UserFacts:
    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS user_facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                confidence REAL DEFAULT 1.0,
                updated_at TEXT NOT NULL
            )
        """)
        self._conn.commit()

    def upsert(self, key: str, value: str, confidence: float = 1.0) -> None:
        existing = self._conn.execute(
            "SELECT id FROM user_facts WHERE key=?", (key,)
        ).fetchone()
        now = datetime.utcnow().isoformat()
        if existing:
            self._conn.execute(
                "UPDATE user_facts SET value=?, confidence=?, updated_at=? WHERE key=?",
                (value, confidence, now, key),
            )
        else:
            self._conn.execute(
                "INSERT INTO user_facts (key, value, confidence, updated_at) VALUES (?,?,?,?)",
                (key, value, confidence, now),
            )
        self._conn.commit()

    def get_all(self) -> dict[str, str]:
        rows = self._conn.execute("SELECT key, value FROM user_facts").fetchall()
        return {k: v for k, v in rows}
```

- [ ] **Step 2: Create `memory/vector_store.py`**

```python
from __future__ import annotations
from pathlib import Path
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from core.logger import get_logger

logger = get_logger(__name__)

class VectorStore:
    def __init__(self, chroma_path: Path, model_name: str = "all-MiniLM-L6-v2") -> None:
        chroma_path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(chroma_path),
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection("jarvis_memory")
        self._model: SentenceTransformer | None = None
        self._model_name = model_name
        self._loaded = False

    def _load_model(self) -> None:
        if self._loaded:
            return
        logger.info(f"Loading sentence-transformers model: {self._model_name}")
        self._model = SentenceTransformer(self._model_name, device="cuda")
        self._loaded = True

    def add(self, doc_id: str, text: str, metadata: dict | None = None) -> None:
        self._load_model()
        embedding = self._model.encode(text).tolist()
        self._collection.upsert(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[metadata or {}],
        )

    def search(self, query: str, top_k: int = 5) -> list[str]:
        self._load_model()
        embedding = self._model.encode(query).tolist()
        results = self._collection.query(
            query_embeddings=[embedding],
            n_results=min(top_k, self._collection.count() or 1),
        )
        return results["documents"][0] if results["documents"] else []
```

- [ ] **Step 3: Add tests**

Append to `tests/test_memory.py`:

```python
def test_user_facts_upsert_and_get(tmp_path):
    from memory.user_facts import UserFacts
    facts = UserFacts(db_path=tmp_path / "test.db")
    facts.upsert("name", "Jakub")
    facts.upsert("name", "Jakub Novak")  # update
    all_facts = facts.get_all()
    assert all_facts["name"] == "Jakub Novak"
```

- [ ] **Step 4: Integrate vector store into Engine**

Modify `core/engine.py` — add to `__init__`:

```python
from memory.vector_store import VectorStore
from memory.user_facts import UserFacts
# In __init__:
self._vector_store = VectorStore(chroma_path=config.memory.chroma_path)
self._user_facts = UserFacts(db_path=config.memory.db_path)
self._exchange_count = 0
```

In `_handle_command`, replace `context = self._history.recent(...)` with:

```python
context = self._history.recent(
    count=self._cfg.memory.context_history_count,
    session_id=self._session_id,
)
# Augment with semantic memories
semantic = self._vector_store.search(text, top_k=3)
if semantic:
    context_prefix = [{"role": "system", "content": "Relevant memories:\n" + "\n".join(semantic)}]
    context = context_prefix + context
```

After logging response:

```python
self._exchange_count += 1
exchange_text = f"User: {text}\nAssistant: {response}"
import uuid as _uuid
self._vector_store.add(str(_uuid.uuid4()), exchange_text, {"session": self._session_id})
# Background user fact extraction every 5 exchanges
if self._exchange_count % 5 == 0:
    asyncio.create_task(self._extract_user_facts(exchange_text))
```

Add method:

```python
async def _extract_user_facts(self, context_text: str) -> None:
    from brain.ollama_client import OllamaClient
    client = OllamaClient(base_url=self._cfg.llm.ollama_url, model=self._cfg.llm.ollama_model)
    prompt = f"""Extract any personal facts about the user from this conversation. 
Return JSON array of {{"key": "...", "value": "..."}} objects, or empty array [].
Conversation:
{context_text}"""
    try:
        raw = await client.chat([{"role": "user", "content": prompt}], json_mode=True)
        import json
        facts = json.loads(raw) if isinstance(raw, str) else raw
        if isinstance(facts, list):
            for f in facts:
                if "key" in f and "value" in f:
                    self._user_facts.upsert(f["key"], f["value"])
    except Exception as e:
        logger.debug(f"User fact extraction failed (non-critical): {e}")
```

- [ ] **Step 5: Run tests**

```bash
pytest tests/ -v
```

- [ ] **Step 6: Commit M7**

```bash
git add memory/ core/engine.py tests/test_memory.py
git commit -m "M7: memory system — ChromaDB vector store, user facts, semantic context"
```

- [ ] **Step 7: Update MASTER_PLAN.md — M7 DONE, M8 NEXT**

---

## Chunk 8: M8 — Polish + Persistent Service

### Task 18: Rich Terminal UI

**Files:**
- Modify: `core/engine.py`
- Create: `core/ui.py`

- [ ] **Step 1: Create `core/ui.py`**

```python
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

console = Console()

def print_user(text: str) -> None:
    console.print(Panel(Text(text, style="bold cyan"), title="[cyan]You", border_style="cyan"))

def print_jarvis(text: str) -> None:
    console.print(Panel(Text(text, style="bold green"), title="[green]J.A.R.V.I.S", border_style="green"))

def print_status(msg: str) -> None:
    console.print(f"[dim]{msg}[/dim]")

def print_error(msg: str) -> None:
    console.print(f"[red]ERROR: {msg}[/red]")

def print_banner() -> None:
    console.print(Panel(
        "[bold green]J.A.R.V.I.S AGI[/bold green]\n[dim]Local-first AI Voice Assistant[/dim]",
        border_style="green",
    ))
```

- [ ] **Step 2: Integrate UI into engine — add `core/ui.py` calls**

In `core/engine.py`, import `from core.ui import print_user, print_jarvis, print_status, print_banner` and add calls:
- `print_banner()` at start of `run()`
- `print_status("Listening...")` before recording
- `print_user(text)` after transcription
- `print_jarvis(response)` before speaking

- [ ] **Step 3: Commit M8**

```bash
git add core/ui.py core/engine.py
git commit -m "M8: rich terminal UI — conversation display"
```

- [ ] **Step 4: Update MASTER_PLAN.md — M8 DONE, all milestones complete**

---

## Final Checklist

- [ ] All tests pass: `pytest tests/ -v`
- [ ] End-to-end smoke test: `python main.py`
- [ ] `MASTER_PLAN.md` shows all milestones DONE
- [ ] `INFOAI.md` reflects final architecture
- [ ] `data/` is gitignored and not committed
- [ ] `.env` is gitignored and not committed
