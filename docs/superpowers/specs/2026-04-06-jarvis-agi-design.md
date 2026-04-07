# J.A.R.V.I.S AGI — Design Specification
**Date:** 2026-04-06
**Status:** Approved

---

## 1. Overview

J.A.R.V.I.S AGI is a local-first, Tony Stark-style AI voice assistant for Windows desktop. It runs 24/7 in the background, listens for a wake word, understands English and Czech voice commands, routes intent via an LLM brain, executes real desktop and web actions, speaks responses back, and maintains persistent memory.

**Hardware target:** AMD Ryzen 5 5600X, 32 GB RAM, RTX 2070 (8 GB VRAM), Windows 11 Pro x64.

---

## 2. Architecture

```
Mic → VAD → Hotword (Porcupine "Jarvis")
  → Record until silence
  → STT (faster-whisper, CUDA, EN+CZ)
  → Brain (Ollama primary / Gemini fallback)
      receives: transcribed text + skill definitions as function list
      returns: (skill_name, params)
  → Skill Execute
  → TTS (edge-tts, async, interruptible, EN+CZ auto-detect)
  → Speaker output

         ↕ (read/write at each step)
   Memory Layer:
     SQLite: chat history + user facts
     ChromaDB: vector embeddings (sentence-transformers CUDA)
```

**Key design decisions:**

- **LLM-based routing** — no regex or keyword matching. The brain LLM receives the full skill registry as structured function definitions and returns which skill to call with what parameters.
- **Ollama primary, Gemini fallback** — local-first for privacy and speed; Gemini API used for complex tasks or when Ollama fails.
- **Interruptible TTS** — detecting the wake word during speech immediately stops playback.
- **Lazy model loading** — faster-whisper and sentence-transformers are not loaded until first use, keeping idle footprint minimal.
- **Terminal MVP first** — `rich` library for pretty terminal output; no GUI in initial milestones.

---

## 3. Tech Stack

| Layer | Technology | Notes |
|---|---|---|
| Language | Python 3.11+ | Type hints, async where beneficial |
| STT | faster-whisper (CUDA) | multilingual, medium model |
| Hotword | pvporcupine (free tier) | "Jarvis" built-in keyword |
| VAD | webrtcvad | Voice activity detection for segment boundaries |
| LLM (primary) | Ollama local — qwen3:8b or gemma3:4b | Prefer qwen3:8b for quality, gemma3:4b for speed |
| LLM (fallback) | Google Gemini API | Complex tasks, Ollama unavailable |
| TTS | edge-tts | Free, high quality, EN+CZ voices |
| Memory (structured) | SQLite | Chat history, user facts key-value store |
| Memory (semantic) | ChromaDB + sentence-transformers (CUDA) | Vector search for relevant context |
| Web search | DuckDuckGo API (duckduckgo-search) | No API key required |
| Weather | wttr.in (HTTP, no key) or openweathermap (key optional) | Default: wttr.in |
| Web skills | playwright | Browser automation, page analysis (web_analyzer) |
| Music | yt-dlp + vlc/subprocess | YouTube playback, no OAuth required |
| Desktop control | subprocess, pyautogui, psutil | App launcher, file manager, system info |
| UI | rich (terminal) | MVP; WebSocket GUI is future work |
| Config | settings.yaml + .env | YAML for user config, .env for secrets |

---

## 4. Project Structure

```
J.A.R.V.I.S AGI/
├── MASTER_PLAN.md
├── INFOAI.md
├── settings.yaml
├── .env.example
├── .env                    (gitignored)
├── .gitignore
├── requirements.txt
├── pyproject.toml
├── main.py
├── core/
│   ├── __init__.py
│   ├── engine.py           # Main orchestrator: listen→transcribe→think→act→speak
│   ├── config.py           # Load settings.yaml + .env
│   └── logger.py           # Structured logging
├── stt/
│   ├── __init__.py
│   ├── hotword.py          # Porcupine wake word detection
│   ├── transcriber.py      # faster-whisper CUDA transcription
│   └── audio_capture.py    # Continuous mic + VAD
├── brain/
│   ├── __init__.py
│   ├── router.py           # LLM intent router → (skill_name, params)
│   ├── ollama_client.py    # Async Ollama API wrapper
│   ├── gemini_client.py    # Gemini API fallback
│   └── prompts.py          # System prompts
├── tts/
│   ├── __init__.py
│   └── speaker.py          # edge-tts, language auto-detect, interruptible queue
├── memory/
│   ├── __init__.py
│   ├── chat_history.py     # SQLite conversation log
│   ├── user_facts.py       # SQLite user preferences/facts
│   └── vector_store.py     # ChromaDB semantic memory
├── skills/
│   ├── __init__.py
│   ├── base.py             # BaseSkill ABC
│   ├── registry.py         # Auto-discovery and registration
│   ├── app_launcher.py
│   ├── web_search.py
│   ├── web_analyzer.py
│   ├── music_player.py
│   ├── file_manager.py
│   ├── screenshot.py
│   ├── weather.py
│   ├── system_info.py
│   ├── note_taker.py
│   ├── timer_alarm.py
│   └── general_chat.py     # LLM fallback
└── tests/
    ├── __init__.py
    ├── test_stt.py
    ├── test_brain.py
    ├── test_tts.py
    └── test_skills.py
```

---

## 5. Component Interfaces

### BaseSkill (skills/base.py)
```python
class BaseSkill(ABC):
    name: str           # unique identifier used by router
    description: str    # shown to LLM for routing
    parameters: dict    # JSON Schema for LLM function calling

    @abstractmethod
    async def execute(self, **kwargs) -> str: ...
```

### Brain Router (brain/router.py)
- Input: `user_text: str`, `context: list[dict]` (recent memory)
- Builds prompt: system prompt + skill registry as function definitions + conversation context + user text
- Calls Ollama (or Gemini fallback) with `timeout=30s`
- **Output contract:** LLM must return a single JSON object: `{"skill": "<name>", "params": {...}}`. This is enforced via the system prompt instruction and JSON mode where available. On parse failure: retry once with explicit format reminder. If skill name is not in registry: fall back to `general_chat`.
- **`general_chat` routing:** When `general_chat` is selected, `router.py` calls `ollama_client` directly (not re-entering the router) to avoid circular routing. The router is a one-shot dispatch.
- Returns `(skill_name: str, params: dict)` to engine.

### Engine (core/engine.py)
- State machine: `IDLE → HOTWORD_DETECTED → RECORDING → TRANSCRIBING → ROUTING → EXECUTING → SPEAKING → IDLE`
- **Interruption threading model:** Porcupine hotword runs in a dedicated daemon thread reading from the mic stream continuously, even during SPEAKING state. It signals an `asyncio.Event` (`cancel_event`). The TTS speaker checks this event and stops playback. The engine then transitions back to HOTWORD_DETECTED. Speaker bleed (mic picking up TTS output) is mitigated by a 200ms audio gate after TTS starts.

### Memory Integration
- Before routing: retrieve top-5 relevant memories from SQLite chat history (always available from M5); ChromaDB semantic search added in M7.
- After execution: log exchange to SQLite immediately; user fact extraction runs as a background asyncio task (non-blocking) — triggered after every 5 exchanges or when a fact keyword is detected.
- `data/` directory is gitignored (contains `jarvis.db`, `chroma/` — personal data, potentially large).

---

## 6. Skill Definitions

Each skill exposes `name`, `description`, and `parameters` for the LLM router, plus an async `execute()` method.

| Skill | Description |
|---|---|
| app_launcher | Open or close Windows applications by name |
| web_search | Search the web, return top results as text |
| web_analyzer | Fetch URL, extract content, summarize with LLM |
| music_player | Play/pause/stop music via YouTube (yt-dlp + vlc) |
| file_manager | List, open, move, delete files |
| screenshot | Take screenshot, optionally OCR |
| weather | Get current weather for a location |
| system_info | CPU/RAM/GPU stats, uptime, battery |
| note_taker | Save and retrieve quick notes |
| timer_alarm | Set countdown timers and alarms |
| general_chat | Fallback — send to LLM and return response |

---

## 7. Memory Design

### SQLite tables
- `chat_history`: id, timestamp, role (user/assistant), content, session_id
- `user_facts`: id, key, value, confidence, updated_at

### ChromaDB collection
- Collection: `jarvis_memory`
- Documents: chat exchanges + extracted facts
- Metadata: timestamp, type (exchange/fact), session_id
- Query: top-5 by semantic similarity to current user input

---

## 8. VRAM Budget

RTX 2070 has 8 GB VRAM. Budget breakdown:
- faster-whisper medium (float16): ~1.5 GB
- sentence-transformers (e.g. all-MiniLM-L6-v2): ~0.5 GB
- Ollama qwen3:8b: runs in RAM (offloaded) by default; GPU layers can be configured via `OLLAMA_GPU_LAYERS` if VRAM budget allows after whisper unloads
- **Strategy:** faster-whisper is loaded on demand (lazy), released after transcription. This frees VRAM for any GPU-accelerated Ollama layers if desired. sentence-transformers stays loaded once initialized.
- **pvporcupine note:** Free tier requires a valid AccessKey from Picovoice console (env var `PORCUPINE_ACCESS_KEY`). Key is validated at startup; if invalid or network unreachable, the system logs the error and falls back to a keyboard trigger mode (press Enter to activate).

---

## 9. Configuration (settings.yaml)

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
tts:
  voice_en: en-US-JennyNeural
  voice_cs: cs-CZ-VlastaNeural
  rate: +0%
  volume: +0%
memory:
  db_path: ./data/jarvis.db
  chroma_path: ./data/chroma
  context_history_count: 5   # number of recent exchanges retrieved as context
timeouts:
  stt_max_recording_sec: 30
  llm_response_sec: 30
  skill_execute_sec: 60
```

---

## 10. Milestones

Milestone ordering reflects actual dependencies. SQLite memory is introduced in M5 (basic chat log required by engine loop). ChromaDB semantic memory added in M7.

| # | Milestone | Status | Key dependency |
|---|---|---|---|
| M1 | Project Scaffolding | pending | — |
| M2 | STT Pipeline (hotword + transcription + VAD) | pending | M1 |
| M3 | LLM Brain + Intent Router | pending | M1 |
| M4 | Text-to-Speech | pending | M1 |
| M5 | Core Engine Loop + SQLite memory (basic) | pending | M2, M3, M4 |
| M6 | First Batch of Skills (11 skills) | pending | M5 |
| M7 | Memory System (ChromaDB semantic + user facts) | pending | M5 |
| M8 | Polish + Persistent Service | pending | M6, M7 |

---

## 11. Testing Strategy

- **Unit tests** per component in `tests/`
- **M2:** test hotword detection with audio file playback; test transcription with pre-recorded clips
- **M3:** test router with mock LLM responses; verify correct skill + params parsed
- **M4:** test TTS generates audio file; test interruption logic
- **M5:** end-to-end integration test (mock mic → spoken response)
- **M6:** test each skill independently with mocked external calls

---

## 12. Error Handling & Degradation

- Ollama unavailable → automatic Gemini fallback
- Gemini unavailable → `general_chat` via Ollama with error acknowledgement
- Both Ollama and Gemini unavailable → speak static fallback string ("I'm having trouble thinking right now"), return to IDLE
- STT fails → retry once, then ask user to repeat
- Skill execute fails → log error, speak "I couldn't do that" fallback
- All errors logged via `core/logger.py` (structured JSON)
