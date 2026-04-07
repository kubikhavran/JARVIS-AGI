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
