# J.A.R.V.I.S AGI — Master Plan

## Status

| Milestone | Status | Notes |
|---|---|---|
| M1 | ✅ DONE | Scaffolding, config, logger, skill base/registry |
| M2 | ✅ DONE | STT pipeline: VAD, hotword, transcription |
| M3 | ✅ DONE | LLM brain + intent router |
| M4 | 🔲 NEXT | TTS: edge-tts, interruptible |
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
