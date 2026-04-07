from __future__ import annotations
import sqlite3
from datetime import datetime, timezone
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
            (datetime.now(timezone.utc).isoformat(), role, content, session_id),
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
