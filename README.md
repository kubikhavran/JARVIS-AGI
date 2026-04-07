# J.A.R.V.I.S AGI

A Tony Stark-style local-first AI voice assistant for Windows. Say **"Jarvis"** — it listens, thinks, acts, and talks back.

## Features

- **Wake word detection** — "Jarvis" via pvporcupine (keyboard fallback if no key)
- **Multilingual STT** — faster-whisper on CUDA, English + Czech
- **LLM brain** — Ollama (local, private) with Google Gemini fallback
- **Intent routing** — LLM picks the right skill from a function registry, no regex
- **Natural TTS** — edge-tts with auto language detection (EN/CZ), interruptible
- **Persistent memory** — SQLite chat history + ChromaDB semantic recall
- **12 built-in skills** — web search, weather, app launcher, notes, timers, files, screenshots, music, system info, and more
- **System tray** — runs silently in background, quit from tray
- **Windows autostart** — optional Task Scheduler integration

## Requirements

- Windows 10/11 x64
- Python 3.11+
- [Ollama](https://ollama.ai) running locally
- NVIDIA GPU with CUDA (recommended; CPU fallback works)

## Quick Start

```bash
# 1. Clone
git clone https://github.com/kubikhavran/JARVIS-AGI.git
cd JARVIS-AGI

# 2. Install dependencies
pip install -r requirements.txt
playwright install chromium   # for web_analyzer skill

# 3. Pull an LLM
ollama pull llama3.1:8b       # or qwen3:8b, gemma3:4b, etc.

# 4. Configure
cp .env.example .env
# Edit .env — add PORCUPINE_ACCESS_KEY (free at console.picovoice.ai)
#              and optionally GEMINI_API_KEY

# Edit settings.yaml to set your preferred model:
#   llm.ollama_model: llama3.1:8b

# 5. Run
python main.py
```

Say **"Jarvis"** (or press **Enter** if no Porcupine key) to activate.

## Example Commands

| Say | What happens |
|---|---|
| "Jarvis, what's the weather in Prague?" | Fetches wttr.in weather |
| "Jarvis, open Notepad" | Launches notepad.exe |
| "Jarvis, search for Python async tutorials" | DuckDuckGo web search |
| "Jarvis, take a screenshot" | Saves screenshot to disk |
| "Jarvis, set a timer for 5 minutes" | Background countdown |
| "Jarvis, save a note: buy milk" | Stored in SQLite |
| "Jarvis, what's my CPU usage?" | psutil system stats |
| "Jarvis, play jazz music" | YouTube via yt-dlp + ffplay |
| "Jarvis, enable autostart" | Registers Windows Task Scheduler |

## Project Structure

```
├── main.py              # Entry point (--install-autostart, --no-tray flags)
├── settings.yaml        # All configuration
├── .env.example         # API key template
├── core/
│   ├── engine.py        # Main state machine: wake→STT→route→skill→TTS
│   ├── config.py        # Loads settings.yaml + .env
│   ├── logger.py        # Structured JSON logging
│   ├── ui.py            # Rich terminal display
│   ├── tray.py          # System tray icon
│   └── autostart.py     # Windows Task Scheduler
├── stt/                 # faster-whisper + pvporcupine + webrtcvad
├── brain/               # Ollama client, Gemini fallback, intent router
├── tts/                 # edge-tts speaker with interruption
├── memory/              # SQLite + ChromaDB
└── skills/              # 12 auto-discovered skills
```

## Configuration

Edit `settings.yaml`:

```yaml
llm:
  ollama_model: llama3.1:8b   # any model pulled in Ollama
stt:
  model: medium               # tiny/base/small/medium/large
  device: cuda                # cuda or cpu
tts:
  voice_en: en-US-JennyNeural
  voice_cs: cs-CZ-VlastaNeural
```

## API Keys (optional)

| Key | Where to get | Used for |
|---|---|---|
| `PORCUPINE_ACCESS_KEY` | [console.picovoice.ai](https://console.picovoice.ai) (free) | "Jarvis" wake word |
| `GEMINI_API_KEY` | [aistudio.google.com](https://aistudio.google.com) | LLM fallback when Ollama unavailable |

Without keys: keyboard fallback (Enter) + Ollama-only mode. Everything still works.

## Running as a Background Service

```bash
# Enable autostart on Windows login
python main.py --install-autostart

# Disable autostart
python main.py --remove-autostart

# Run without tray icon
python main.py --no-tray
```

## Tech Stack

| Layer | Technology |
|---|---|
| STT | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) + CUDA |
| Wake word | [pvporcupine](https://picovoice.ai/platform/porcupine/) |
| LLM | [Ollama](https://ollama.ai) (local) + Google Gemini (fallback) |
| TTS | [edge-tts](https://github.com/rany2/edge-tts) |
| Memory | SQLite + [ChromaDB](https://www.trychroma.com) + sentence-transformers |
| Web | [Playwright](https://playwright.dev) + DuckDuckGo |
| UI | [rich](https://github.com/Textualize/rich) + pystray |

## License

MIT
