import sqlite3
from datetime import datetime, timezone
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
                (datetime.now(timezone.utc).isoformat(), content),
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
